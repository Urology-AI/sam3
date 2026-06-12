#!/usr/bin/env python3
"""
extract_multiclass_features_v2.py
=================================
Extract SAM2 ViT spatial FPN features (downsampled) for the v2 11-class
phase classifier (attention-pool + BiGRU).

Two modes:
  --mode train  → 15 intuitive_videos cases, augmentation ON, K variants per
                  frame (variant 0 is clean; variants 1..K-1 are augmented).
                  Labels and sample weights come from the v1 sample plan
                  (events + interphases + safety-gap drop + hard-neg weighting).
  --mode val    → 1 OOD folder case (multi-chunk), augmentation OFF, K=1.
                  Chunks stitched into a continuous surgery timeline; labels
                  derived from events_SUBJ_*.csv by surgery-absolute time.
                  No safety-gap drop in val (we want every frame labelled;
                  scoring uses Viterbi-decoded event windows, not per-frame
                  labels).

Output per case (.npz, compressed)  — channel counts detected at runtime
(for Hiera-S they are C1=64, C2=256):
  features_fpn1  : (N, K, C1, 16, 16) float16  — fpn[1] adaptive-avg-pooled
  features_fpn2  : (N, K, C2,  8,  8) float16  — fpn[2] adaptive-avg-pooled
  labels         : (N,) int32                    — 11-class label
  sample_weights : (N,) float32                  — hard-neg multiplier (train);
                                                    1.0 everywhere (val)
  frame_indices  : (N,) int64                    — source-video frame index
                                                    (intra-chunk in val mode)
  times_s        : (N,) float32                  — surgery-absolute time
                                                    (stitched across chunks in val)
  case_id        : str
  mode           : "train" | "val"
  K              : int
  fps            : float                          — train: source fps
                                                    val:  -1.0 (varies per chunk)
  chunks         : (val) array of chunk basenames
  event_csv      : (val) basename of source events CSV

Augmentation stack (train mode only, variants 1..K-1; variant 0 is clean):
  1. Right-side panel composite, prob 0.70. The surgical view is resized
     (aspect preserved, scale-to-fit) into the left sub-rectangle of the
     canvas — never occluded. A panel from panel_library/ (auto-cropped to
     its non-black content bbox) fills the right sub-rectangle. Panel
     width = U[0.20, 0.40] × canvas width; panel brightness ±15 %.
     This matches dv5 recordings where the panel occupies a right strip
     and the surgical view occupies the rest.
  2. Letterbox OR pillarbox bars, prob 0.30; bar width 5–25 %.
  3. Gaussian blur σ ~ U[0.1, 2.0], prob 0.50.
  4. Unsharp mask (radius 5 px), prob 0.15.
  5. Colour jitter (B/C/S/H), prob 0.70.
  6. Downscale–upscale at scale U[0.40, 0.80], prob 0.30.
  7. JPEG re-encode at quality U[30, 90], prob 0.50.

Usage
-----
  python3 extract_multiclass_features_v2.py --mode train --use_bf16
  python3 extract_multiclass_features_v2.py --mode val --use_bf16 \\
      --video_folder /sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_1b7d93c2_Y2025_DOY143 \\
      --event_csv event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv
"""

import argparse
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
ANNOT_CSV = os.path.join(VIDEO_DIR, "annotate_fine.csv")
PANEL_DIR = os.path.join(SAM3_DIR, "panel_library")

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

# Reuse v1 constants and helpers (v1 is not modified).
from extract_multiclass_features import (  # noqa: E402
    VARIANT_TO_CKPT, VARIANT_TO_CFG,
    IMG_SIZE, IMG_MEAN, IMG_STD,
    NUM_CLASSES, CLASS_NAMES,
    EVENT_TO_CLASS, REQUIRED_EVENTS, VAS_CSV_NAMES,
    parse_all_events, build_sample_plan,
    load_sam2,
)

# Downsampled FPN spatial sizes (Option B in the v2 plan). Channel counts
# are detected at runtime via detect_fpn_channels() — they depend on the
# Hiera variant and on which FPN level got the mem_dim projection. For
# Hiera-S at 1024 input the observed counts are fpn[1]=64, fpn[2]=256.
FPN1_DS = (16, 16)
FPN2_DS = (8, 8)


# ── Augmentation primitives ───────────────────────────────────────────────────

def _crop_to_content(panel_bgr, threshold=15):
    """Tight-crop a panel to its non-near-black bounding box.
    Panel PNGs are captured as full-canvas right-side strips; most of each
    image is black canvas around the actual HUD widgets. Pasting the whole
    canvas dumps that empty region onto training frames. This crop keeps
    only the widget content."""
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not mask.any():
        return panel_bgr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    return panel_bgr[y0:y1 + 1, x0:x1 + 1]


def load_panel_library(panel_dir):
    paths = sorted(glob.glob(os.path.join(panel_dir, "panel_*.png")))
    panels = []
    raw_dims, crop_dims = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        raw_dims.append(img.shape[:2])
        cropped = _crop_to_content(img)
        crop_dims.append(cropped.shape[:2])
        panels.append(cropped)
    if panels:
        raw_avg  = np.mean([h * w for h, w in raw_dims])
        crop_avg = np.mean([h * w for h, w in crop_dims])
        kept_pct = 100.0 * crop_avg / max(raw_avg, 1.0)
        print(f"  panel_library: {len(panels)} panels (auto-cropped to "
              f"non-black bbox; avg kept {kept_pct:.0f}% of source pixels)")
    else:
        print(f"  panel_library: 0 panels from {panel_dir}")
    return panels


def aug_side_panel_layout(frame, panels, rng):
    """Right-side panel composite — matches recorded-video layout.

    The surgical view is resized (aspect preserved, scale-to-fit) into the
    left sub-rectangle of the canvas; the HUD panel fills the right
    sub-rectangle. The surgical view is NEVER occluded by the panel — they
    live in disjoint regions of the output canvas, exactly like dv5
    recordings where the panel occupies the right strip and the surgical
    view occupies the rest.

    Panel width is sampled uniformly in [0.20, 0.40] × canvas width."""
    if not panels:
        return frame
    fh, fw   = frame.shape[:2]
    panel    = panels[rng.randrange(len(panels))]
    ph0, pw0 = panel.shape[:2]

    panel_w = int(fw * rng.uniform(0.20, 0.40))
    surg_w  = fw - panel_w
    if surg_w <= 16 or panel_w <= 16:
        return frame

    # Surgical view: scale-to-fit into (surg_w, fh), aspect preserved,
    # centred (so any letterboxing falls equally above/below).
    s_scale = min(surg_w / fw, fh / fh)
    new_sw  = max(1, int(fw * s_scale))
    new_sh  = max(1, int(fh * s_scale))
    surg    = cv2.resize(frame, (new_sw, new_sh),
                         interpolation=cv2.INTER_AREA)
    sox = (surg_w - new_sw) // 2
    soy = (fh     - new_sh) // 2

    # Panel: scale-to-fit into (panel_w, fh), aspect preserved, brightness
    # jittered, anchored top within its sub-rect (HUD widgets start at the
    # top in real recordings).
    p_scale = min(panel_w / pw0, fh / ph0)
    new_pw  = max(1, int(pw0 * p_scale))
    new_ph  = max(1, int(ph0 * p_scale))
    panel_r = cv2.resize(panel, (new_pw, new_ph),
                         interpolation=cv2.INTER_AREA)
    panel_r = np.clip(panel_r.astype(np.float32) * rng.uniform(0.85, 1.15),
                      0, 255).astype(np.uint8)
    pox = surg_w + (panel_w - new_pw) // 2
    poy = 0

    canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
    canvas[soy:soy + new_sh, sox:sox + new_sw] = surg
    canvas[poy:poy + new_ph, pox:pox + new_pw] = panel_r
    return canvas


def aug_letterbox(frame, rng):
    fh, fw = frame.shape[:2]
    if rng.random() < 0.5:
        bt = int(fh * rng.uniform(0.05, 0.25))
        bb = int(fh * rng.uniform(0.05, 0.25))
        frame[:bt] = 0
        if bb > 0:
            frame[-bb:] = 0
    else:
        bl = int(fw * rng.uniform(0.05, 0.25))
        br = int(fw * rng.uniform(0.05, 0.25))
        frame[:, :bl] = 0
        if br > 0:
            frame[:, -br:] = 0
    return frame


def aug_gaussian_blur(frame, rng):
    sigma = rng.uniform(0.1, 2.0)
    ksize = max(3, int(sigma * 4 + 1) | 1)
    return cv2.GaussianBlur(frame, (ksize, ksize), sigma)


def aug_unsharp(frame, rng):
    blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
    amount  = rng.uniform(0.5, 1.5)
    out     = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_colour_jitter(frame, rng):
    # Brightness × contrast in BGR.
    b = rng.uniform(0.85, 1.15)
    f = frame.astype(np.float32) * b
    mean = f.mean(axis=(0, 1), keepdims=True)
    c = rng.uniform(0.85, 1.15)
    f = mean + (f - mean) * c
    f = np.clip(f, 0, 255).astype(np.uint8)
    # Saturation × hue in HSV.
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_downscale_upscale(frame, rng):
    scale = rng.uniform(0.4, 0.8)
    fh, fw = frame.shape[:2]
    small = cv2.resize(frame,
                       (max(1, int(fw * scale)), max(1, int(fh * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (fw, fh), interpolation=cv2.INTER_LINEAR)


def aug_jpeg(frame, rng):
    q = int(rng.uniform(30, 90))
    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return frame
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def augment_frame(frame_bgr, panels, rng):
    out = frame_bgr.copy()
    if rng.random() < 0.70:
        out = aug_side_panel_layout(out, panels, rng)
    if rng.random() < 0.30:
        out = aug_letterbox(out, rng)
    if rng.random() < 0.50:
        out = aug_gaussian_blur(out, rng)
    if rng.random() < 0.15:
        out = aug_unsharp(out, rng)
    if rng.random() < 0.70:
        out = aug_colour_jitter(out, rng)
    if rng.random() < 0.30:
        out = aug_downscale_upscale(out, rng)
    if rng.random() < 0.50:
        out = aug_jpeg(out, rng)
    return out


# ── Preprocess (SBS crop first, then augment, then normalise) ─────────────────

def apply_sbs(frame_bgr, sbs_eye):
    if sbs_eye == "left":
        return frame_bgr[:, :frame_bgr.shape[1] // 2]
    if sbs_eye == "right":
        return frame_bgr[:, frame_bgr.shape[1] // 2:]
    return frame_bgr


def preprocess_to_tensor(frame_bgr, img_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size),
                     interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


# ── Spatial feature extraction ────────────────────────────────────────────────

@torch.no_grad()
def extract_batch_spatial(model, frames_t, device, use_bf16):
    imgs = frames_t.to(device, non_blocking=True)
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(imgs)
    else:
        backbone_out = model.forward_image(imgs)
    fpn  = backbone_out["backbone_fpn"]
    fpn1 = fpn[1].float()
    fpn2 = fpn[2].float()
    fpn1_ds = F.adaptive_avg_pool2d(fpn1, FPN1_DS).half()
    fpn2_ds = F.adaptive_avg_pool2d(fpn2, FPN2_DS).half()
    return fpn1_ds.cpu().numpy(), fpn2_ds.cpu().numpy()


@torch.no_grad()
def detect_fpn_channels(model, device, use_bf16):
    """One dummy forward pass to detect fpn[1]/fpn[2] channel counts.
    Returns (C_fpn1, C_fpn2). Avoids hardcoding channel assumptions that
    vary across Hiera variants and FpnNeck mem_dim projection placement."""
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    f1, f2 = extract_batch_spatial(model, dummy, device, use_bf16)
    return int(f1.shape[1]), int(f2.shape[1])


# ── Train-mode per-case pipeline ──────────────────────────────────────────────

def process_train_case(case_id, sorted_events, model, panels,
                       args, device, rng, fpn_channels):
    out_path = os.path.join(args.out_dir, f"case_{case_id}.npz")
    if os.path.exists(out_path) and not args.overwrite:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  case {case_id}: already extracted → {out_path} "
              f"({size_mb:.1f} MB) — skipping (pass --overwrite to re-extract)")
        return

    video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_end_s  = total_frames / fps if fps > 0 else 0.0
    cap.release()

    frame_indices, labels, weights = build_sample_plan(
        total_frames, fps, sorted_events, video_end_s,
        args.sample_fps, args.safety_gap, args.hard_neg_duration,
        args.hard_neg_weight, args.max_interphase_per_case,
    )
    if len(frame_indices) == 0:
        print(f"  SKIP: empty sample plan")
        return

    K = args.k_aug
    N = len(frame_indices)
    C1, C2 = fpn_channels
    print(f"  case {case_id}: {total_frames} frames @ {fps:.2f} fps  "
          f"({video_end_s/60:.1f} min)  → N={N}  K={K}  "
          f"hard_neg={int((weights > 1.0).sum())}")

    fpn1_out = np.empty((N, K, C1, *FPN1_DS), dtype=np.float16)
    fpn2_out = np.empty((N, K, C2, *FPN2_DS), dtype=np.float16)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    current = int(frame_indices[0])

    pbar = tqdm(total=N, desc=f"  case {case_id}", unit="frame", ncols=80)
    batch_tensors, batch_meta = [], []   # meta = (row, variant)

    def flush():
        if not batch_tensors:
            return
        fpn1_b, fpn2_b = extract_batch_spatial(
            model, torch.stack(batch_tensors), device, args.use_bf16,
        )
        for (r, v), f1, f2 in zip(batch_meta, fpn1_b, fpn2_b):
            fpn1_out[r, v] = f1
            fpn2_out[r, v] = f2
        batch_tensors.clear()
        batch_meta.clear()

    for row, fi in enumerate(frame_indices):
        fi = int(fi)
        while current < fi:
            cap.grab()
            current += 1
        ret, frame_bgr = cap.read()
        current += 1
        if not ret or frame_bgr is None:
            frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        frame_cropped = apply_sbs(frame_bgr, args.sbs_eye)
        for v in range(K):
            f_use = frame_cropped if v == 0 \
                    else augment_frame(frame_cropped, panels, rng)
            batch_tensors.append(preprocess_to_tensor(f_use, IMG_SIZE))
            batch_meta.append((row, v))
            if len(batch_tensors) >= args.batch_size:
                flush()
        pbar.update(1)
    pbar.close()
    cap.release()
    flush()

    times_s = (frame_indices.astype(np.float64) / max(fps, 1e-6)).astype(np.float32)

    np.savez_compressed(
        out_path,
        features_fpn1  = fpn1_out,
        features_fpn2  = fpn2_out,
        labels         = labels,
        sample_weights = weights,
        frame_indices  = frame_indices,
        times_s        = times_s,
        fps            = np.float32(fps),
        case_id        = case_id,
        mode           = "train",
        K              = np.int32(K),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")


# ── Val-mode helpers ──────────────────────────────────────────────────────────

def parse_val_events(event_csv):
    rows = []
    with open(event_csv, newline="") as f:
        for r in csv.DictReader(f):
            name = r["event"].strip()
            if name in VAS_CSV_NAMES:
                name = "vas_cut"
            if name not in EVENT_TO_CLASS:
                continue
            rows.append((name, float(r["start_sec"]), float(r["end_sec"])))
    rows.sort(key=lambda x: x[1])
    return rows


def label_by_absolute_time(t_s, events_sorted):
    for name, s, e in events_sorted:
        if s <= t_s <= e:
            return EVENT_TO_CLASS[name]
    last = -1
    for i, (_, s, e) in enumerate(events_sorted):
        if t_s > e:
            last = i
        else:
            break
    if last == -1:
        return 0
    return EVENT_TO_CLASS[events_sorted[last][0]] + 1


def discover_chunks(folder, pattern):
    matches = set()
    for pat in (pattern, pattern.lower(), pattern.upper()):
        matches.update(glob.glob(os.path.join(folder, pat)))
    if not matches:
        raise FileNotFoundError(f"no videos matching {pattern} in {folder}")
    return sorted(matches, key=os.path.basename)


def process_val_folder(folder, event_csv, model, args, device):
    chunks = discover_chunks(folder, args.video_pattern)
    print(f"  {len(chunks)} chunks in {folder}")
    for c in chunks:
        print(f"    {os.path.basename(c)}")

    events_sorted = parse_val_events(event_csv)
    print(f"  Events ({len(events_sorted)}):")
    for n, s, e in events_sorted:
        print(f"    {n:<14}  {s:>8.1f}s – {e:>8.1f}s  "
              f"({s/60:.1f}–{e/60:.1f} min)")

    all_fpn1, all_fpn2 = [], []
    all_labels, all_times, all_frames = [], [], []
    batch_tensors, batch_meta = [], []

    def flush():
        if not batch_tensors:
            return
        fpn1_b, fpn2_b = extract_batch_spatial(
            model, torch.stack(batch_tensors), device, args.use_bf16,
        )
        for (fi, t_abs), f1, f2 in zip(batch_meta, fpn1_b, fpn2_b):
            all_fpn1.append(f1)
            all_fpn2.append(f2)
            all_labels.append(label_by_absolute_time(t_abs, events_sorted))
            all_times.append(t_abs)
            all_frames.append(fi)
        batch_tensors.clear()
        batch_meta.clear()

    surgery_offset = 0.0
    for ci, vp in enumerate(chunks):
        cap          = cv2.VideoCapture(vp)
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chunk_dur    = total_frames / fps if fps > 0 else 0.0
        stride       = max(1, int(round(fps / args.sample_fps)))
        indices      = list(range(0, total_frames, stride))
        cap.release()

        print(f"\n  Chunk {ci+1}/{len(chunks)}  fps={fps:.2f}  "
              f"frames={total_frames}  dur={chunk_dur:.1f}s  "
              f"sample={len(indices)}  offset={surgery_offset:.1f}s")

        cap = cv2.VideoCapture(vp)
        if indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        current = indices[0] if indices else 0
        pbar = tqdm(total=len(indices), desc=f"  chunk {ci+1}",
                    unit="frame", ncols=80)
        for fi in indices:
            while current < fi:
                cap.grab()
                current += 1
            ret, frame_bgr = cap.read()
            current += 1
            if not ret or frame_bgr is None:
                frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            t_abs = surgery_offset + fi / max(fps, 1e-6)
            frame_cropped = apply_sbs(frame_bgr, args.sbs_eye)
            batch_tensors.append(preprocess_to_tensor(frame_cropped, IMG_SIZE))
            batch_meta.append((fi, t_abs))
            if len(batch_tensors) >= args.batch_size:
                flush()
            pbar.update(1)
        pbar.close()
        cap.release()
        surgery_offset += chunk_dur
    flush()

    N = len(all_fpn1)
    if N == 0:
        print("  SKIP: no frames sampled")
        return

    fpn1_arr = np.stack(all_fpn1, axis=0)[:, None]   # (N, 1, 256, 16, 16)
    fpn2_arr = np.stack(all_fpn2, axis=0)[:, None]   # (N, 1,  64,  8,  8)
    labels   = np.array(all_labels, dtype=np.int32)
    weights  = np.ones(N, dtype=np.float32)
    times_s  = np.array(all_times,  dtype=np.float32)
    frames   = np.array(all_frames, dtype=np.int64)

    print(f"\n  Total val samples: {N}  duration: {times_s[-1]:.1f}s "
          f"({times_s[-1]/60:.1f} min)")
    for cls in range(NUM_CLASSES):
        n = int((labels == cls).sum())
        if n > 0:
            print(f"    cls {cls:>2} {CLASS_NAMES[cls]:<22}  n={n}")

    out_name = os.path.basename(folder.rstrip("/"))
    out_path = os.path.join(args.out_dir, f"{out_name}.npz")
    np.savez_compressed(
        out_path,
        features_fpn1  = fpn1_arr,
        features_fpn2  = fpn2_arr,
        labels         = labels,
        sample_weights = weights,
        frame_indices  = frames,
        times_s        = times_s,
        fps            = np.float32(-1.0),
        case_id        = out_name,
        mode           = "val",
        K              = np.int32(1),
        chunks         = np.array([os.path.basename(c) for c in chunks]),
        event_csv      = os.path.basename(event_csv),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "val"])
    p.add_argument("--sample_fps",       type=float, default=1.0)
    p.add_argument("--safety_gap",       type=float, default=3.0)
    p.add_argument("--hard_neg_duration",type=float, default=10.0)
    p.add_argument("--hard_neg_weight",  type=float, default=5.0)
    p.add_argument("--max_interphase_per_case", type=int, default=None)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--sbs_eye",          default="none",
                   choices=["left", "right", "none"])
    p.add_argument("--sam2_variant",     default="small",
                   choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--use_bf16",         action="store_true")
    p.add_argument("--seed",             type=int, default=42)

    p.add_argument("--annot_csv",        default=ANNOT_CSV)
    p.add_argument("--cases",            default=None,
                   help="Comma-separated case IDs (default: all in CSV).")
    p.add_argument("--k_aug",            type=int, default=4,
                   help="Variants per frame (variant 0 clean, rest augmented). "
                        "Train mode only.")
    p.add_argument("--panel_dir",        default=PANEL_DIR)

    p.add_argument("--video_folder",     default=None,
                   help="OOD folder for val mode.")
    p.add_argument("--event_csv",        default=None,
                   help="event_annotations/events_SUBJ_*.csv for val mode.")
    p.add_argument("--video_pattern",    default="*.MP4")

    p.add_argument("--out_dir",          default=None)
    p.add_argument("--overwrite",        action="store_true",
                   help="Re-extract cases even if the output .npz already exists. "
                        "Default: skip already-extracted cases (train mode only).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(SAM3_DIR, "multiclass_features_v2", args.mode)
    os.makedirs(args.out_dir, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {args.mode}  sample_fps: {args.sample_fps}  "
          f"batch: {args.batch_size}  variant: {args.sam2_variant}  "
          f"bf16: {args.use_bf16}")
    print(f"FPN downsample: fpn1 → {FPN1_DS}, fpn2 → {FPN2_DS}")
    print(f"Out: {args.out_dir}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_sam2(args.sam2_variant, device)

    fpn_channels = detect_fpn_channels(model, device, args.use_bf16)
    print(f"FPN channels detected: fpn[1]={fpn_channels[0]}  "
          f"fpn[2]={fpn_channels[1]}")

    if args.mode == "train":
        by_case = parse_all_events(args.annot_csv)
        if args.cases:
            keep = set(args.cases.split(","))
            by_case = {k: v for k, v in by_case.items() if k in keep}
        complete = {}
        for cid, evs in sorted(by_case.items()):
            if REQUIRED_EVENTS.issubset({n for n, _, _ in evs}):
                complete[cid] = evs
        print(f"\nTrain cases ({len(complete)}): {sorted(complete.keys())}\n")
        panels = load_panel_library(args.panel_dir)
        if not panels:
            print("  [warn] no panels — HUD-paste augmentation is a no-op")
        for case_id, sorted_events in tqdm(sorted(complete.items()),
                                           desc="Cases", unit="case", ncols=80):
            print(f"\n[case {case_id}]  events: "
                  f"{[(n, int(s), int(e)) for n, s, e in sorted_events]}")
            process_train_case(case_id, sorted_events, model, panels,
                               args, device, rng, fpn_channels)
    else:
        if args.video_folder is None or args.event_csv is None:
            raise SystemExit("val mode requires --video_folder and --event_csv")
        process_val_folder(args.video_folder, args.event_csv,
                           model, args, device)

    print(f"\nDone. Features → {args.out_dir}/")


if __name__ == "__main__":
    main()
