#!/usr/bin/env python3
"""
extract_vas_features.py
=======================
Extract SAM2 ViT (Hiera) backbone features for the VAS classifier.

For each case in untitled.txt:
  - Samples frames at --sample_fps across the full video
  - Labels frames inside the VAS window as positive (1), everything else as negative (0)
  - Subsamples negatives so the ratio is --neg_ratio × positives (balanced enough)
  - Runs SAM2 forward_image in batches → global-average-pools each FPN scale → concatenates
  - Saves a per-case .npz to --out_dir

Why SAM2 and not SAM3:
  forward_image() runs only the Hiera ViT backbone + FPN neck — identical in both.
  SAM3's CLIP/language conditioning only runs inside the detector head, not here.
  SAM2 is simpler to load (no BPE tokenizer).

Usage
-----
  python3 extract_vas_features.py
  python3 extract_vas_features.py --sample_fps 0.5 --sbs_eye none
  python3 extract_vas_features.py --out_dir custom_features/
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR   = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
CLIP_LIST   = os.path.join(VIDEO_DIR, "untitled.txt")
OUT_DIR_DEFAULT = os.path.join(SAM3_DIR, "vas_features")

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

IMG_SIZE  = 1024
IMG_MEAN  = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD   = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Parsing ────────────────────────────────────────────────────────────────────

def to_seconds(t):
    parts = t.strip().split(":")
    s = 0
    for p in parts:
        s = s * 60 + int(p)
    return s


def parse_clip_list(path):
    """Returns dict: case_id (str) → list of (start_s, end_s) tuples."""
    windows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, case_id, timerange = line.split()
            start_s = to_seconds(timerange.split("-")[0])
            end_s   = to_seconds(timerange.split("-")[1])
            windows.setdefault(case_id, []).append((start_s, end_s))
    return windows


# ── Feature extraction ─────────────────────────────────────────────────────────

def load_sam2(device):
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    print(f"  SAM2-large loaded")
    return model


def preprocess_frame(frame_bgr, img_size, sbs_eye):
    """Crop SBS if needed, resize, normalise → [1, 3, H, W] float32 tensor."""
    if sbs_eye == "left":
        frame_bgr = frame_bgr[:, :frame_bgr.shape[1] // 2]
    elif sbs_eye == "right":
        frame_bgr = frame_bgr[:, frame_bgr.shape[1] // 2:]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = (t - IMG_MEAN) / IMG_STD
    return t  # [3, H, W]


@torch.no_grad()
def extract_batch(model, frames_t, device):
    """
    frames_t: [B, 3, H, W]
    Returns: [B, feat_dim] — global-avg-pool over all FPN scales, concatenated.
    """
    imgs = frames_t.to(device)
    backbone_out = model.forward_image(imgs)
    fpn = backbone_out["backbone_fpn"]          # list of [B, C, H, W]
    pooled = [f.mean(dim=[2, 3]) for f in fpn]  # list of [B, C]
    feat = torch.cat(pooled, dim=1)             # [B, C*n_scales]
    return feat.cpu().float().numpy()


# ── Sampling ───────────────────────────────────────────────────────────────────

def build_sample_plan(total_frames, fps, vas_windows, sample_fps, neg_ratio):
    """
    Returns (frame_indices, labels) both np.int32/float32.
    Positives: all frames inside any VAS window at sample_fps.
    Negatives: uniformly spread frames outside VAS windows, capped at neg_ratio × n_pos.
    """
    stride = max(1, int(round(fps / sample_fps)))

    def in_vas(frame_idx):
        t = frame_idx / fps
        return any(s <= t <= e for s, e in vas_windows)

    all_frames = np.arange(0, total_frames, stride)
    pos_frames = [fi for fi in all_frames if     in_vas(fi)]
    neg_frames = [fi for fi in all_frames if not in_vas(fi)]

    n_neg_target = int(len(pos_frames) * neg_ratio)
    if len(neg_frames) > n_neg_target:
        idx = np.linspace(0, len(neg_frames) - 1, n_neg_target, dtype=int)
        neg_frames = [neg_frames[i] for i in idx]

    frames = np.array(pos_frames + neg_frames, dtype=np.int64)
    labels = np.array([1]*len(pos_frames) + [0]*len(neg_frames), dtype=np.int32)
    order  = np.argsort(frames)
    return frames[order], labels[order]


# ── Per-case pipeline ──────────────────────────────────────────────────────────

def process_case(case_id, vas_windows, model, args, device):
    video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices, labels = build_sample_plan(
        total_frames, fps, vas_windows, args.sample_fps, args.neg_ratio
    )
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    print(f"  case {case_id}: {total_frames} frames  "
          f"pos={n_pos}  neg={n_neg}  total={len(frame_indices)}")

    cap = cv2.VideoCapture(video_path)
    frame_tensors = []
    valid_indices = []
    valid_labels  = []
    all_feat_chunks = []

    pbar = tqdm(total=len(frame_indices), desc=f"  case {case_id}",
                unit="frame", ncols=80)

    for fi, lab in zip(frame_indices, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame_bgr = cap.read()
        pbar.update(1)
        if not ret:
            continue
        t = preprocess_frame(frame_bgr, IMG_SIZE, args.sbs_eye)
        frame_tensors.append(t)
        valid_indices.append(fi)
        valid_labels.append(lab)

        if len(frame_tensors) == args.batch_size:
            batch = torch.stack(frame_tensors)
            feats = extract_batch(model, batch, device)
            all_feat_chunks.append(feats)
            frame_tensors = []

    pbar.close()
    cap.release()

    # Flush remaining
    if frame_tensors:
        batch = torch.stack(frame_tensors)
        feats = extract_batch(model, batch, device)
        all_feat_chunks.append(feats)

    if not all_feat_chunks:
        print(f"  SKIP: no features extracted for case {case_id}")
        return

    all_feats = np.concatenate(all_feat_chunks, axis=0)

    out_path = os.path.join(args.out_dir, f"case_{case_id}.npz")
    np.savez_compressed(
        out_path,
        features      = all_feats,
        labels        = np.array(valid_labels, dtype=np.int32),
        frame_indices = np.array(valid_indices, dtype=np.int64),
        fps           = np.float32(fps),
        case_id       = case_id,
    )
    print(f"  Saved → {out_path}  feat_dim={all_feats.shape[1]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_list",   default=CLIP_LIST)
    p.add_argument("--out_dir",     default=OUT_DIR_DEFAULT)
    p.add_argument("--sample_fps",  type=float, default=1.0,
                   help="Frames per second to sample (default 1.0)")
    p.add_argument("--neg_ratio",   type=float, default=4.0,
                   help="Negative-to-positive ratio (default 4 — slight imbalance ok)")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--sbs_eye",     default="left",
                   choices=["left", "right", "none"],
                   help="SBS 3D crop: left/right half or none for 2D (default left)")
    p.add_argument("--cases",       default=None,
                   help="Comma-separated case IDs to process (default: all in clip_list)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    windows = parse_clip_list(args.clip_list)
    if args.cases:
        keep = set(args.cases.split(","))
        windows = {k: v for k, v in windows.items() if k in keep}

    print(f"\nCases to process: {sorted(windows.keys())}")
    print(f"Sample rate: {args.sample_fps} fps  |  neg_ratio: {args.neg_ratio}  "
          f"|  sbs_eye: {args.sbs_eye}\n")

    model = load_sam2(device)

    case_items = sorted(windows.items())
    for case_id, vas_windows in tqdm(case_items, desc="Cases", unit="case", ncols=80):
        print(f"\n[case {case_id}]  VAS windows: {vas_windows}")
        process_case(case_id, vas_windows, model, args, device)

    print(f"\nDone. Features saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
