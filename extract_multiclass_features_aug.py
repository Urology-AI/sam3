#!/usr/bin/env python3
"""
extract_multiclass_features_aug.py
==================================
v1 multi-class extractor (`extract_multiclass_features.py`) + v2 augmentation
recipe.

This is the v1 multi-class pipeline -- 11-class softmax sample plan with
hard-negative mining (post-event 10s @ 5x sample weight), 3s safety gap
around every event boundary, vas_cut_1/vas_cut_2 merged into class 5, etc.
-- with augmentation hooks added at the train-side encoder pass and an
extra OOD folder mode for inference-side feature extraction.

Two modes
---------
  --mode train   15 intuitive_videos cases. Per-case sample plan = v1
                 multi-class build_sample_plan (events + interphases +
                 safety-gap drop + hard-neg weighting). Each kept frame
                 is encoded K times (variant 0 clean, variants 1..K-1
                 augmented). Output is flat (N*K, D) features with
                 labels and sample_weights replicated K times; schema is
                 compatible with v1 multiclass train_multiclass_classifier.py
                 and localize_multiclass.py.

  --mode val     1 OOD folder case (multi-chunk). No augmentation, K=1.
                 Chunks stitched into a continuous surgery-absolute timeline
                 so the OOD event CSV (already in surgery-absolute seconds)
                 maps directly. No labels saved -- scoring lives in
                 localize_multiclass_aug.py.

The augmentation primitives come verbatim from extract_multiclass_features_v2.py
so the recipe is identical to what produced aug_test_samples/.

Usage
-----
  # Train extraction (augmented intuitive cases)
  python3 extract_multiclass_features_aug.py --mode train --k_aug 4 --use_bf16

  # OOD folder extraction (no aug)
  python3 extract_multiclass_features_aug.py --mode val --use_bf16 \\
      --video_folder /sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_1b7d93c2_Y2025_DOY143
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
ANNOT_CSV = os.path.join(VIDEO_DIR, "annotate_fine.csv")
PANEL_DIR = os.path.join(SAM3_DIR, "panel_library")

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

from extract_multiclass_features import (  # noqa: E402
    VARIANT_TO_CKPT, VARIANT_TO_CFG,
    IMG_SIZE, IMG_MEAN, IMG_STD,
    NUM_CLASSES, CLASS_NAMES,
    REQUIRED_EVENTS,
    parse_all_events,
    build_sample_plan,
    load_sam2,
)
from extract_multiclass_features_v2 import (  # noqa: E402
    load_panel_library,
    augment_frame,
)


# ── Preprocess ────────────────────────────────────────────────────────────────

def apply_sbs(frame_bgr, sbs_eye):
    if sbs_eye == "left":
        return frame_bgr[:, :frame_bgr.shape[1] // 2]
    if sbs_eye == "right":
        return frame_bgr[:, frame_bgr.shape[1] // 2:]
    return frame_bgr


def preprocess_to_tensor(frame_bgr, img_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


# ── Encoder (flat 1024-d / 640-d, same layout as v1 multiclass) ───────────────

@torch.no_grad()
def encode_flat(model, frames_t, device, use_bf16):
    imgs = frames_t.to(device, non_blocking=True)
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(imgs)
    else:
        backbone_out = model.forward_image(imgs)
    fpn = backbone_out["backbone_fpn"]
    pooled = []
    for f in fpn[1:]:                       # drop fpn[0] (finest, local texture)
        ff = f.float()
        pooled.append(ff.mean(dim=[2, 3]))   # avg-pool
        pooled.append(ff.amax(dim=[2, 3]))   # max-pool
    return torch.cat(pooled, dim=1).cpu().numpy()


# ── Train mode (augmented intuitive cases) ────────────────────────────────────

def process_train_case(case_id, sorted_events, model, panels, args, device, rng):
    out_path = os.path.join(args.out_dir, f"case_{case_id}.npz")
    if os.path.exists(out_path) and not args.overwrite:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  case {case_id}: already extracted ({size_mb:.1f} MB) -- skipping")
        return

    video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found -> {video_path}")
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

    K     = args.k_aug
    N     = len(frame_indices)
    n_hn  = int((weights > 1.0).sum())
    print(f"  case {case_id}: {total_frames}f @ {fps:.2f}  "
          f"({video_end_s/60:.1f} min)  N={N}  K={K}  hard_neg={n_hn}  -> {N*K} samples")
    for cls in range(NUM_CLASSES):
        n = int((labels == cls).sum())
        if n > 0:
            n_hn_cls = int(((labels == cls) & (weights > 1.0)).sum())
            print(f"      cls {cls:>2} {CLASS_NAMES[cls]:<22}  n={n:<5}  hard_neg={n_hn_cls}")

    feat_dim  = None
    feats_all = None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    current = int(frame_indices[0])

    pbar = tqdm(total=N, desc=f"  case {case_id}", unit="frame", ncols=80)
    batch_tensors, batch_meta = [], []   # meta = (row, variant)

    def flush():
        nonlocal feats_all, feat_dim
        if not batch_tensors:
            return
        feats = encode_flat(model, torch.stack(batch_tensors), device, args.use_bf16)
        if feats_all is None:
            feat_dim  = feats.shape[1]
            feats_all = np.empty((N, K, feat_dim), dtype=np.float32)
        for (r, v), f in zip(batch_meta, feats):
            feats_all[r, v] = f
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
            f_use = frame_cropped if v == 0 else augment_frame(frame_cropped, panels, rng)
            batch_tensors.append(preprocess_to_tensor(f_use, IMG_SIZE))
            batch_meta.append((row, v))
            if len(batch_tensors) >= args.batch_size:
                flush()
        pbar.update(1)
    pbar.close()
    cap.release()
    flush()

    # Flatten (N, K, D) -> (N*K, D); replicate labels/weights/frame_indices K times.
    feats_flat   = feats_all.reshape(N * K, feat_dim)
    labels_flat  = np.repeat(labels.astype(np.int32),  K)
    weights_flat = np.repeat(weights.astype(np.float32), K)
    frames_flat  = np.repeat(frame_indices.astype(np.int64), K)
    variant_flat = np.tile(np.arange(K, dtype=np.int32), N)

    np.savez_compressed(
        out_path,
        features       = feats_flat,
        labels         = labels_flat,
        sample_weights = weights_flat,
        frame_indices  = frames_flat,
        variant        = variant_flat,
        fps            = np.float32(fps),
        case_id        = case_id,
        k_aug          = np.int32(K),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved -> {out_path}  shape={feats_flat.shape}  ({size_mb:.1f} MB)")


# ── Val mode (OOD multi-chunk folder) ─────────────────────────────────────────

def discover_chunks(folder, pattern):
    matches = set()
    for pat in (pattern, pattern.lower(), pattern.upper()):
        matches.update(glob.glob(os.path.join(folder, pat)))
    if not matches:
        raise FileNotFoundError(f"no videos matching {pattern} in {folder}")
    return sorted(matches, key=os.path.basename)


def process_val_folder(folder, model, args, device):
    chunks = discover_chunks(folder, args.video_pattern)
    print(f"  {len(chunks)} chunks in {folder}")
    for c in chunks:
        print(f"    {os.path.basename(c)}")

    feat_dim   = None
    all_feats  = []
    all_times  = []
    all_frames = []
    all_chunks = []

    batch_tensors, batch_meta = [], []   # meta = (ci, fi, t_abs)

    def flush():
        nonlocal feat_dim
        if not batch_tensors:
            return
        feats = encode_flat(model, torch.stack(batch_tensors), device, args.use_bf16)
        if feat_dim is None:
            feat_dim = feats.shape[1]
        for (ci, fi, t_abs), f in zip(batch_meta, feats):
            all_feats.append(f)
            all_times.append(t_abs)
            all_frames.append(fi)
            all_chunks.append(ci)
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
            batch_meta.append((ci, fi, t_abs))
            if len(batch_tensors) >= args.batch_size:
                flush()
            pbar.update(1)
        pbar.close()
        cap.release()
        surgery_offset += chunk_dur
    flush()

    N = len(all_feats)
    if N == 0:
        print("  SKIP: no frames sampled")
        return

    feats   = np.stack(all_feats, axis=0)
    times_s = np.array(all_times,  dtype=np.float32)
    frames  = np.array(all_frames, dtype=np.int64)
    chunks_ = np.array(all_chunks, dtype=np.int32)

    print(f"\n  Total val samples: {N}  duration: {times_s[-1]:.1f}s "
          f"({times_s[-1]/60:.1f} min)  feat_dim={feats.shape[1]}")

    out_name = os.path.basename(folder.rstrip("/"))
    out_path = os.path.join(args.out_dir, f"{out_name}.npz")
    np.savez_compressed(
        out_path,
        features      = feats,
        times_s       = times_s,
        frame_indices = frames,
        chunk_indices = chunks_,
        case_id       = out_name,
        chunks        = np.array([os.path.basename(c) for c in chunks]),
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "val"])

    # train (mirrors extract_multiclass_features.py defaults)
    p.add_argument("--annot_csv",                default=ANNOT_CSV)
    p.add_argument("--safety_gap",               type=float, default=3.0)
    p.add_argument("--hard_neg_duration",        type=float, default=10.0)
    p.add_argument("--hard_neg_weight",          type=float, default=5.0)
    p.add_argument("--max_interphase_per_case",  type=int,   default=None)
    p.add_argument("--cases",                    default=None,
                   help="Comma-separated case IDs (default: all in CSV).")
    p.add_argument("--k_aug",                    type=int,   default=4,
                   help="Variants per kept frame; variant 0 clean, 1..K-1 augmented.")
    p.add_argument("--panel_dir",                default=PANEL_DIR)

    # val (OOD folder)
    p.add_argument("--video_folder",  default=None,
                   help="OOD multi-chunk folder for val mode.")
    p.add_argument("--video_pattern", default="*.MP4")

    # shared
    p.add_argument("--sample_fps",   type=float, default=1.0)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--sbs_eye",      default="none", choices=["left", "right", "none"])
    p.add_argument("--sam2_variant", default="small",
                   choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--use_bf16",     action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out_dir",      default=None)
    p.add_argument("--overwrite",    action="store_true",
                   help="Re-extract cases even if the output .npz already exists "
                        "(train mode only).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        if args.mode == "train":
            args.out_dir = os.path.join(SAM3_DIR, "multiclass_features_aug", "train")
        else:
            args.out_dir = os.path.join(SAM3_DIR, "multiclass_features_aug", "val")
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
    if args.mode == "train":
        print(f"safety_gap: {args.safety_gap}s  "
              f"hard_neg: {args.hard_neg_duration}s x {args.hard_neg_weight}  "
              f"k_aug: {args.k_aug}")
    print(f"Out: {args.out_dir}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_sam2(args.sam2_variant, device)

    if args.mode == "train":
        by_case = parse_all_events(args.annot_csv)
        if not by_case:
            print(f"\nNo usable annotations in {args.annot_csv}.")
            return
        if args.cases:
            keep = set(args.cases.split(","))
            by_case = {k: v for k, v in by_case.items() if k in keep}

        complete = {}
        skipped  = []
        for cid, evs in sorted(by_case.items()):
            present = {n for n, _, _ in evs}
            missing = REQUIRED_EVENTS - present
            if missing:
                skipped.append((cid, sorted(missing)))
            else:
                complete[cid] = evs
        if skipped:
            print("\n[skip] cases missing required events:")
            for cid, miss in skipped:
                print(f"        case {cid}: missing {miss}")
        by_case = complete

        print(f"\nTrain cases ({len(by_case)}): {sorted(by_case.keys())}\n")

        panels = load_panel_library(args.panel_dir)
        if not panels:
            print("  [warn] no panels found -- HUD-paste augmentation is a no-op")

        for case_id, sorted_events in tqdm(sorted(by_case.items()),
                                           desc="Cases", unit="case", ncols=80):
            print(f"\n[case {case_id}]  events: "
                  f"{[(n, int(s), int(e)) for n, s, e in sorted_events]}")
            process_train_case(case_id, sorted_events, model, panels,
                               args, device, rng)
    else:
        if args.video_folder is None:
            raise SystemExit("val mode requires --video_folder")
        process_val_folder(args.video_folder, model, args, device)

    print(f"\nDone. Features -> {args.out_dir}/")


if __name__ == "__main__":
    main()
