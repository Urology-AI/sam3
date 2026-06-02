#!/usr/bin/env python3
"""
extract_endobag_features.py
===========================
Extract SAM2 ViT (Hiera) backbone features for the endobagging classifier.

For each case in annotate_fine.csv (event == "endobag"):
  - Samples frames at --sample_fps across the full video
  - Labels frames inside the endobagging window as positive (1), everything else as negative (0)
  - Subsamples negatives so the ratio is --neg_ratio × positives
  - Runs SAM2 forward_image in batches → global-average-pools each FPN scale → concatenates
  - Saves a per-case .npz to --out_dir

Usage
-----
  python3 extract_endobag_features.py
  python3 extract_endobag_features.py --sample_fps 0.5 --sbs_eye none
  python3 extract_endobag_features.py --out_dir custom_features/
"""

import argparse
import os
import sys
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR   = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
ANNOT_CSV   = os.path.join(VIDEO_DIR, "annotate_fine.csv")
OUT_DIR_DEFAULT = os.path.join(SAM3_DIR, "endobag_features")

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

VARIANT_TO_CKPT = {
    "tiny":  "checkpoints/sam2.1_hiera_tiny.pt",
    "small": "checkpoints/sam2.1_hiera_small.pt",
    "base":  "checkpoints/sam2.1_hiera_base_plus.pt",
    "large": "checkpoints/sam2.1_hiera_large.pt",
}
VARIANT_TO_CFG = {
    "tiny":  "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

IMG_SIZE  = 1024
IMG_MEAN  = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD   = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_endobag_annotations(path):
    """
    Returns dict: case_id (str) → list of (start_s, end_s) tuples.
    Reads annotate_fine.csv and filters rows where event == "endobag".
    case_id is extracted from filename: "case_213_clipped.mp4" → "213".
    start_sec / end_sec columns are already in seconds (integers).
    """
    windows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"].strip() != "endobag":
                continue
            # "case_213_clipped.mp4" → "213"
            fname = row["filename"].strip()
            case_id = fname.replace("case_", "").replace("_clipped.mp4", "")
            start_s = int(row["start_sec"])
            end_s   = int(row["end_sec"])
            windows.setdefault(case_id, []).append((start_s, end_s))
    return windows


# ── Feature extraction ─────────────────────────────────────────────────────────

def load_sam2(variant, device):
    from sam2.build_sam import build_sam2
    ckpt = os.path.join(SAM2_DIR, VARIANT_TO_CKPT[variant])
    cfg  = VARIANT_TO_CFG[variant]
    model = build_sam2(cfg, ckpt, device=device)
    model.eval()
    print(f"  SAM2-{variant} loaded  ({ckpt})")
    return model


def preprocess_frame(frame_bgr, img_size, sbs_eye):
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
    imgs = frames_t.to(device)
    backbone_out = model.forward_image(imgs)
    fpn = backbone_out["backbone_fpn"]
    # skip finest scale (fpn[0]) — local texture, not useful for scene-level phase detection
    # avg+max per remaining scale: avg captures mean activation, max captures presence anywhere
    pooled = []
    for f in fpn[1:]:
        pooled.append(f.mean(dim=[2, 3]))
        pooled.append(f.amax(dim=[2, 3]))
    feat = torch.cat(pooled, dim=1)
    return feat.cpu().float().numpy()


# ── Sampling ───────────────────────────────────────────────────────────────────

def build_sample_plan(total_frames, fps, endobag_windows, sample_fps, neg_ratio):
    stride = max(1, int(round(fps / sample_fps)))

    def in_endobag(frame_idx):
        t = frame_idx / fps
        return any(s <= t <= e for s, e in endobag_windows)

    all_frames = np.arange(0, total_frames, stride)
    pos_frames = [fi for fi in all_frames if     in_endobag(fi)]
    neg_frames = [fi for fi in all_frames if not in_endobag(fi)]

    n_neg_target = int(len(pos_frames) * neg_ratio)
    if len(neg_frames) > n_neg_target:
        idx = np.linspace(0, len(neg_frames) - 1, n_neg_target, dtype=int)
        neg_frames = [neg_frames[i] for i in idx]

    frames = np.array(pos_frames + neg_frames, dtype=np.int64)
    labels = np.array([1]*len(pos_frames) + [0]*len(neg_frames), dtype=np.int32)
    order  = np.argsort(frames)
    return frames[order], labels[order]


# ── Per-case pipeline ──────────────────────────────────────────────────────────

def process_case(case_id, endobag_windows, model, args, device):
    video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices, labels = build_sample_plan(
        total_frames, fps, endobag_windows, args.sample_fps, args.neg_ratio
    )
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    print(f"  case {case_id}: {total_frames} frames  "
          f"pos={n_pos}  neg={n_neg}  total={len(frame_indices)}")

    cap = cv2.VideoCapture(video_path)
    frame_tensors    = []
    valid_indices    = []
    valid_labels     = []
    all_feat_chunks  = []

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
    p.add_argument("--annot_csv",   default=ANNOT_CSV)
    p.add_argument("--out_dir",     default=OUT_DIR_DEFAULT)
    p.add_argument("--sample_fps",  type=float, default=1.0)
    p.add_argument("--neg_ratio",   type=float, default=4.0)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--sbs_eye",     default="none",
                   choices=["left", "right", "none"])
    p.add_argument("--sam2_variant", default="large",
                   choices=["tiny", "small", "base", "large"],
                   help="Which Hiera backbone variant to extract features with")
    p.add_argument("--cases",       default=None,
                   help="Comma-separated case IDs to process (default: all in csv)")
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

    windows = parse_endobag_annotations(args.annot_csv)
    if args.cases:
        keep = set(args.cases.split(","))
        windows = {k: v for k, v in windows.items() if k in keep}

    print(f"\nCases to process: {sorted(windows.keys())}")
    print(f"Sample rate: {args.sample_fps} fps  |  neg_ratio: {args.neg_ratio}  "
          f"|  sbs_eye: {args.sbs_eye}  |  variant: {args.sam2_variant}\n")

    model = load_sam2(args.sam2_variant, device)

    for case_id, endobag_windows in tqdm(sorted(windows.items()),
                                         desc="Cases", unit="case", ncols=80):
        print(f"\n[case {case_id}]  endobag windows: {endobag_windows}")
        process_case(case_id, endobag_windows, model, args, device)

    print(f"\nDone. Features saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
