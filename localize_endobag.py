#!/usr/bin/env python3
"""
localize_endobag.py
===================
Train on N-1 cases, run full-video inference on the held-out case,
and temporally localise the endobagging event.

Post-processing:
  rolling mean smoothing → threshold → contiguous segment detection

Output:
  - probability trace plot (PNG) with ground-truth window shaded
  - predicted (start_s, end_s) printed to stdout

Usage
-----
  python3 localize_endobag.py --hold_out 219
  python3 localize_endobag.py --hold_out 213 --sample_fps 0.5 --smooth_window 20
  python3 localize_endobag.py --all
"""

import argparse
import os
import sys
import glob
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR     = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR     = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR    = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
ANNOT_CSV    = os.path.join(VIDEO_DIR, "annotate_fine.csv")
FEATURES_DIR = os.path.join(SAM3_DIR, "endobag_features")
OUT_DIR      = os.path.join(SAM3_DIR, "endobag_localization")

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMG_SIZE  = 1024
IMG_MEAN  = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD   = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_endobag_annotations(path):
    """
    Returns dict: case_id (str) → list of (start_s, end_s) tuples.
    Reads annotate_fine.csv, filters event == "endobag".
    start_sec/end_sec columns are already integers in seconds.
    """
    windows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"].strip() != "endobag":
                continue
            fname   = row["filename"].strip()
            case_id = fname.replace("case_", "").replace("_clipped.mp4", "")
            start_s = int(row["start_sec"])
            end_s   = int(row["end_sec"])
            windows.setdefault(case_id, []).append((start_s, end_s))
    return windows


# ── Feature loading ────────────────────────────────────────────────────────────

def load_all_cases(features_dir):
    cases = {}
    for path in sorted(glob.glob(os.path.join(features_dir, "case_*.npz"))):
        d = np.load(path, allow_pickle=True)
        case_id = str(d["case_id"])
        cases[case_id] = {
            "features": d["features"].astype(np.float32),
            "labels":   d["labels"].astype(np.int32),
        }
    return cases


# ── Classifier ────────────────────────────────────────────────────────────────

def train_classifier(cases_dict, hold_out_id):
    from sklearn.linear_model import LogisticRegression

    train_ids = [cid for cid in cases_dict if cid != hold_out_id]
    X_train = np.concatenate([cases_dict[cid]["features"] for cid in train_ids])
    y_train = np.concatenate([cases_dict[cid]["labels"]   for cid in train_ids])

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="lbfgs", random_state=42)
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
    fpn = backbone_out["backbone_fpn"]
    pooled = []
    for f in fpn[1:]:
        pooled.append(f.mean(dim=[2, 3]))
        pooled.append(f.amax(dim=[2, 3]))
    return torch.cat(pooled, dim=1).cpu().float().numpy()


def infer_full_video(video_path, model, clf, scaler_mean, scaler_std,
                     sample_fps, batch_size, sbs_eye, device):
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
            ret = cap.grab()
            if not ret:
                break

        frame_num += 1

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
    pad = window // 2
    x_pad = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")[:len(x)]


def find_segments(times_s, probs_smooth, threshold):
    above = probs_smooth >= threshold
    segments = []
    in_seg    = False
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


def pick_best_segment(segments, times_s, probs_smooth):
    """
    Endobagging happens exactly once — return the single segment with the
    highest mean smoothed probability.  This favours sustained confidence
    over a brief spike, which is what we want for a phase-level event.
    """
    if not segments:
        return None
    def mean_prob(seg):
        mask = (times_s >= seg[0]) & (times_s <= seg[1])
        return probs_smooth[mask].mean() if mask.any() else 0.0
    return max(segments, key=mean_prob)


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_plot(times_s, probs_raw, probs_smooth, gt_windows, pred_segments,
              selected_segment, threshold, case_id, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(16, 4))

    for i, (s, e) in enumerate(gt_windows):
        ax.axvspan(s, e, color="green", alpha=0.20,
                   label="Ground truth" if i == 0 else None)

    # all candidates faint, selected one bold
    for i, (s, e) in enumerate(pred_segments):
        ax.axvspan(s, e, color="red", alpha=0.08,
                   label="Candidates" if i == 0 else None)
    if selected_segment:
        s, e = selected_segment
        ax.axvspan(s, e, color="red", alpha=0.40, label="Selected")

    ax.plot(times_s, probs_raw,    color="steelblue", alpha=0.35,
            linewidth=0.8, label="Raw prob")
    ax.plot(times_s, probs_smooth, color="steelblue", linewidth=1.8,
            label="Smoothed")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.9,
               label=f"Threshold {threshold:.2f}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Endobagging probability")
    ax.set_title(f"Endobagging localisation — case {case_id}")
    ax.set_ylim(0, 1)
    ax.set_xlim(times_s[0], times_s[-1])
    ax.legend(fontsize=8, loc="upper right")

    ax2 = ax.secondary_xaxis("top",
                              functions=(lambda x: x/60, lambda x: x*60))
    ax2.set_xlabel("Time (min)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Plot → {out_png}")


# ── Per-case run ───────────────────────────────────────────────────────────────

def run_case(hold_out_id, cases_dict, endobag_windows, sam2_model, args, device):
    video_path = os.path.join(VIDEO_DIR, f"case_{hold_out_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return

    gt_windows = endobag_windows.get(hold_out_id, [])
    if not gt_windows:
        print(f"  SKIP: no endobag annotations for case {hold_out_id}")
        return

    print(f"\n{'='*60}")
    print(f"  Hold-out: case {hold_out_id}")
    print(f"  GT windows: {gt_windows}")
    print(f"{'='*60}")

    clf, scaler_mean, scaler_std = train_classifier(cases_dict, hold_out_id)

    print(f"\n  Running full-video inference at {args.sample_fps} fps ...")
    times_s, probs_raw = infer_full_video(
        video_path, sam2_model, clf, scaler_mean, scaler_std,
        args.sample_fps, args.batch_size, args.sbs_eye, device,
    )

    window_frames = max(1, int(args.smooth_window * args.sample_fps))
    probs_smooth  = rolling_mean(probs_raw, window_frames)

    pred_segments    = find_segments(times_s, probs_smooth, args.threshold)
    selected_segment = pick_best_segment(pred_segments, times_s, probs_smooth)

    gt_s, gt_e = gt_windows[0]

    print(f"\n  Ground truth  : {gt_s:.0f}s–{gt_e:.0f}s ({gt_s/60:.1f}–{gt_e/60:.1f} min)")
    print(f"  All candidates: {[(f'{s:.0f}s–{e:.0f}s ({s/60:.1f}–{e/60:.1f} min)') for s,e in pred_segments]}")

    if selected_segment:
        s, e = selected_segment
        err_start = s - gt_s
        err_end   = e - gt_e
        print(f"  Selected      : {s:.0f}s–{e:.0f}s ({s/60:.1f}–{e/60:.1f} min)  "
              f"(start err={err_start:+.0f}s  end err={err_end:+.0f}s)")
    else:
        err_start = None
        print(f"  Selected      : none (no segments above threshold)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"case_{hold_out_id}_localization.png")
    save_plot(times_s, probs_raw, probs_smooth, gt_windows, pred_segments,
              selected_segment, args.threshold, hold_out_id, out_png)

    return {
        "case_id":   hold_out_id,
        "gt":        (gt_s, gt_e),
        "selected":  selected_segment,
        "err_start": err_start,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

WITHIN_3MIN = 180  # seconds

def print_summary(results, out_dir):
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  SUMMARY  ({len(results)} cases)")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Case':<8} {'GT start':>10} {'Pred start':>12} {'Start err':>11}  {'<3 min?'}")
    lines.append(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*11}  {'-'*7}")

    within = []
    for r in results:
        gt_s, gt_e = r["gt"]
        err        = r["err_start"]
        if r["selected"] is None:
            pred_str = "no prediction"
            err_str  = "—"
            ok       = False
        else:
            ps, pe   = r["selected"]
            pred_str = f"{ps:.0f}s ({ps/60:.1f}m)"
            err_str  = f"{err:+.0f}s"
            ok       = abs(err) <= WITHIN_3MIN
        within.append(ok)
        tick = "✓" if ok else "✗"
        lines.append(f"  {r['case_id']:<8} {gt_s:.0f}s ({gt_s/60:.1f}m) {pred_str:>12}  {err_str:>10}  {tick}")

    n_within = sum(within)
    lines.append(f"\n  Within 3 min: {n_within}/{len(results)}  "
                 f"({100*n_within/len(results):.0f}%)")

    errs = [abs(r["err_start"]) for r in results if r["err_start"] is not None]
    if errs:
        lines.append(f"  Median |start err|: {sorted(errs)[len(errs)//2]:.0f}s")
        lines.append(f"  Mean   |start err|: {sum(errs)/len(errs):.0f}s")
    lines.append(f"{'='*60}")

    text = "\n".join(lines)
    print(f"\n{text}\n")

    out_path = os.path.join(out_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"  Summary → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--hold_out", help="Case ID to hold out (e.g. 219)")
    group.add_argument("--all",      action="store_true")
    p.add_argument("--annot_csv",    default=ANNOT_CSV)
    p.add_argument("--features_dir", default=FEATURES_DIR)
    p.add_argument("--out_dir",      default=OUT_DIR)
    p.add_argument("--sample_fps",   type=float, default=0.5)
    p.add_argument("--smooth_window",type=float, default=20.0,
                   help="Rolling mean window in seconds (default 20)")
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--sbs_eye",      default="none",
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
    cases_dict      = load_all_cases(args.features_dir)
    endobag_windows = parse_endobag_annotations(args.annot_csv)

    print(f"\nLoading SAM2 backbone...")
    sam2_model = load_sam2(device)

    hold_outs = list(cases_dict.keys()) if args.all else [args.hold_out]

    results = []
    for hold_out_id in hold_outs:
        if hold_out_id not in cases_dict:
            print(f"  SKIP: no features for case {hold_out_id} — "
                  f"run extract_endobag_features.py first")
            continue
        r = run_case(hold_out_id, cases_dict, endobag_windows, sam2_model, args, device)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        os.makedirs(args.out_dir, exist_ok=True)
        print_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
