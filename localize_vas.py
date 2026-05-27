#!/usr/bin/env python3
"""
localize_vas.py
===============
Train on N-1 cases, run full-video inference on the held-out case,
and temporally localise the VAS dissection event.

Inference uses sequential frame reading (grab/retrieve) instead of random
seeks — skipping non-sampled frames with cap.grab() costs almost nothing
since the frame is not decoded, making uniform subsampling fast regardless
of video length.

Post-processing:
  rolling mean smoothing → threshold → contiguous segment detection

Output:
  - probability trace plot (PNG) with ground-truth window shaded
  - predicted (start_s, end_s) printed to stdout

Usage
-----
  python3 localize_vas.py --hold_out 219
  python3 localize_vas.py --hold_out 234 --sample_fps 0.5 --smooth_window 20
  python3 localize_vas.py --all          # run for every case in turn
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR     = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR     = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR    = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
CLIP_LIST    = os.path.join(VIDEO_DIR, "untitled.txt")
FEATURES_DIR = os.path.join(SAM3_DIR, "vas_features")
OUT_DIR      = os.path.join(SAM3_DIR, "vas_localization")

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


# ── Feature loading ────────────────────────────────────────────────────────────

def load_all_cases(features_dir):
    cases = {}
    for path in sorted(glob.glob(os.path.join(features_dir, "case_*.npz"))):
        d = np.load(path, allow_pickle=True)
        case_id  = str(d["case_id"])
        cases[case_id] = {
            "features": d["features"].astype(np.float32),
            "labels":   d["labels"].astype(np.int32),
        }
    return cases


# ── Classifier ────────────────────────────────────────────────────────────────

def train_mlp(cases_dict, hold_out_id):
    from sklearn.neural_network import MLPClassifier

    train_ids = [cid for cid in cases_dict if cid != hold_out_id]
    X_train = np.concatenate([cases_dict[cid]["features"] for cid in train_ids])
    y_train = np.concatenate([cases_dict[cid]["labels"]   for cid in train_ids])

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std

    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                        early_stopping=True, validation_fraction=0.1,
                        random_state=42, learning_rate_init=1e-3)
    clf.fit(X_train_s, y_train)
    print(f"  Trained on {len(train_ids)} cases  ({len(y_train)} frames  "
          f"pos={y_train.sum()}  neg={(1-y_train).sum()})")
    return clf, mean, std


# ── Full-video inference ───────────────────────────────────────────────────────

def load_sam2(device):
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    return model


@torch.no_grad()
def extract_batch(model, frames_t, device):
    imgs = frames_t.to(device)
    backbone_out = model.forward_image(imgs)
    fpn    = backbone_out["backbone_fpn"]
    pooled = [f.mean(dim=[2, 3]) for f in fpn]
    return torch.cat(pooled, dim=1).cpu().float().numpy()


def infer_full_video(video_path, model, clf, scaler_mean, scaler_std,
                     sample_fps, batch_size, sbs_eye, device):
    """
    Sequential read with grab() for fast uniform subsampling.
    Returns (times_s, probabilities) as 1-D arrays.
    """
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride       = max(1, int(round(fps / sample_fps)))
    n_samples    = total_frames // stride
    cap.release()

    cap = cv2.VideoCapture(video_path)
    frame_tensors = []
    times_s       = []
    all_probs     = []
    frame_num     = 0

    pbar = tqdm(total=n_samples, desc="  inference", unit="frame", ncols=80)

    while True:
        if frame_num % stride == 0:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            t_s = frame_num / fps

            if sbs_eye == "left":
                frame_bgr = frame_bgr[:, :frame_bgr.shape[1] // 2]
            elif sbs_eye == "right":
                frame_bgr = frame_bgr[:, frame_bgr.shape[1] // 2:]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            t = (t - IMG_MEAN) / IMG_STD
            frame_tensors.append(t)
            times_s.append(t_s)
            pbar.update(1)

            if len(frame_tensors) == batch_size:
                feats = extract_batch(model, torch.stack(frame_tensors), device)
                feats_s = (feats - scaler_mean) / scaler_std
                probs = clf.predict_proba(feats_s)[:, 1]
                all_probs.extend(probs.tolist())
                frame_tensors = []
        else:
            ret = cap.grab()   # fast skip — no decode
            if not ret:
                break

        frame_num += 1

    # flush remainder
    if frame_tensors:
        feats = extract_batch(model, torch.stack(frame_tensors), device)
        feats_s = (feats - scaler_mean) / scaler_std
        probs = clf.predict_proba(feats_s)[:, 1]
        all_probs.extend(probs.tolist())

    pbar.close()
    cap.release()

    return np.array(times_s), np.array(all_probs)


# ── Temporal post-processing ───────────────────────────────────────────────────

def rolling_mean(x, window):
    """Symmetric rolling mean with edge padding."""
    pad = window // 2
    x_pad = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")[:len(x)]


def find_segments(times_s, probs_smooth, threshold):
    """Return list of (start_s, end_s) for contiguous regions above threshold."""
    above = probs_smooth >= threshold
    segments = []
    in_seg   = False
    seg_start = 0.0
    for i, (t, a) in enumerate(zip(times_s, above)):
        if a and not in_seg:
            in_seg = True
            seg_start = t
        elif not a and in_seg:
            in_seg = False
            segments.append((seg_start, times_s[i - 1]))
    if in_seg:
        segments.append((seg_start, times_s[-1]))
    return segments


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_plot(times_s, probs_raw, probs_smooth, gt_windows, pred_segments,
              threshold, case_id, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(16, 4))

    # Ground-truth windows
    for i, (s, e) in enumerate(gt_windows):
        ax.axvspan(s, e, color="green", alpha=0.20,
                   label="Ground truth" if i == 0 else None)

    # Predicted segments
    for i, (s, e) in enumerate(pred_segments):
        ax.axvspan(s, e, color="red", alpha=0.12,
                   label="Predicted" if i == 0 else None)

    ax.plot(times_s, probs_raw,    color="steelblue", alpha=0.35,
            linewidth=0.8, label="Raw prob")
    ax.plot(times_s, probs_smooth, color="steelblue", linewidth=1.8,
            label="Smoothed")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.9,
               label=f"Threshold {threshold:.2f}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VAS probability")
    ax.set_title(f"VAS localisation — case {case_id}")
    ax.set_ylim(0, 1)
    ax.set_xlim(times_s[0], times_s[-1])
    ax.legend(fontsize=8, loc="upper right")

    # Secondary x-axis in minutes
    ax2 = ax.secondary_xaxis("top",
                              functions=(lambda x: x/60, lambda x: x*60))
    ax2.set_xlabel("Time (min)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Plot → {out_png}")


# ── Per-case run ───────────────────────────────────────────────────────────────

def run_case(hold_out_id, cases_dict, vas_windows, sam2_model, args, device):
    video_path = os.path.join(VIDEO_DIR, f"case_{hold_out_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return

    gt_windows = vas_windows.get(hold_out_id, [])
    if not gt_windows:
        print(f"  SKIP: no VAS annotations for case {hold_out_id}")
        return

    print(f"\n{'='*60}")
    print(f"  Hold-out: case {hold_out_id}")
    print(f"  GT windows: {gt_windows}")
    print(f"{'='*60}")

    # Train
    clf, scaler_mean, scaler_std = train_mlp(cases_dict, hold_out_id)

    # Full-video inference
    print(f"\n  Running full-video inference at {args.sample_fps} fps ...")
    times_s, probs_raw = infer_full_video(
        video_path, sam2_model, clf, scaler_mean, scaler_std,
        args.sample_fps, args.batch_size, args.sbs_eye, device,
    )

    # Smooth
    window_frames = max(1, int(args.smooth_window * args.sample_fps))
    probs_smooth  = rolling_mean(probs_raw, window_frames)

    # Segment
    pred_segments = find_segments(times_s, probs_smooth, args.threshold)

    # Report
    print(f"\n  Ground truth : {[(f'{s:.0f}s–{e:.0f}s ({s/60:.1f}–{e/60:.1f} min)') for s,e in gt_windows]}")
    print(f"  Predicted    : {[(f'{s:.0f}s–{e:.0f}s ({s/60:.1f}–{e/60:.1f} min)') for s,e in pred_segments]}")

    # Temporal error for best-matching predicted segment per GT window
    for gt_s, gt_e in gt_windows:
        gt_mid = (gt_s + gt_e) / 2
        best = min(pred_segments, key=lambda seg: abs((seg[0]+seg[1])/2 - gt_mid),
                   default=None)
        if best:
            err_start = best[0] - gt_s
            err_end   = best[1] - gt_e
            print(f"  GT {gt_s:.0f}–{gt_e:.0f}s → best pred {best[0]:.0f}–{best[1]:.0f}s  "
                  f"(start err={err_start:+.0f}s  end err={err_end:+.0f}s)")

    # Plot
    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"case_{hold_out_id}_localization.png")
    save_plot(times_s, probs_raw, probs_smooth, gt_windows, pred_segments,
              args.threshold, hold_out_id, out_png)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--hold_out", help="Case ID to hold out (e.g. 219)")
    group.add_argument("--all",      action="store_true",
                       help="Run for every case in turn")
    p.add_argument("--features_dir", default=FEATURES_DIR)
    p.add_argument("--out_dir",      default=OUT_DIR)
    p.add_argument("--sample_fps",   type=float, default=0.5,
                   help="Inference sampling rate (default 0.5 fps)")
    p.add_argument("--smooth_window",type=float, default=20.0,
                   help="Rolling mean window in seconds (default 20)")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Decision threshold (default 0.5)")
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--sbs_eye",      default="left",
                   choices=["left", "right", "none"])
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading features from {args.features_dir}/")
    cases_dict  = load_all_cases(args.features_dir)
    vas_windows = parse_clip_list(CLIP_LIST)

    print(f"\nLoading SAM2 backbone...")
    sam2_model = load_sam2(device)

    hold_outs = list(cases_dict.keys()) if args.all else [args.hold_out]

    for hold_out_id in hold_outs:
        if hold_out_id not in cases_dict:
            print(f"  SKIP: no features for case {hold_out_id} — "
                  f"run extract_vas_features.py first")
            continue
        run_case(hold_out_id, cases_dict, vas_windows, sam2_model, args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
