#!/usr/bin/env python3
"""
probe_crop_spike.py
===================
Single-event probe for the "invert the strategy" experiment: instead of
augmenting train frames to look like the Sony recordings, normalize the Sony
frame toward the training distribution (crop to the surgical sub-rectangle,
optionally histogram-match colours to an intuitive reference) and feed that
to the encoder.

For one OOD video chunk and one annotated event window, this script:

  1. Trains (or loads from cache) up to three multinomial LR classifiers:
       v1    : original non-augmented features (multiclass_features/, the
               ones behind the good in-domain LOCO results)
       clean : variant == 0 rows of multiclass_features_aug/train/
       aug   : all variants of multiclass_features_aug/train/
  2. Samples frames at --sample_fps in [event_start - context,
     event_end + context] and encodes each frame under several input kinds:
       full        : whole 1920x1080 canvas (control, known failure)
       crop        : surgical bbox crop
       crop_match  : crop + fixed per-channel histogram-match LUT toward the
                     intuitive reference (+ optional unsharp)   [--hist_match]
  3. Reports, per (weights x input) combo, the target-class probability
     inside vs outside the GT window, plus a PNG trace and a CSV.

The histogram-match LUT is *global*, not per-frame: pooled BGR histograms are
computed once from the Sony chunk (cropped) and once from a few intuitive
reference videos, and a single 256-entry per-channel mapping is applied to
every frame. Per-frame matching would erase real scene-level brightness
variation (cautery, deep pelvis), which may itself be phase signal.

Default video/bbox/event are the catheter_pull of SUBJ_1b7d93c2 chunk 1
(GT 908.61-914.94s, surgery-absolute == chunk-relative for chunk 1).

Run inside the singularity shell (needs GPU):
  python3 probe_crop_spike.py --use_bf16 --hist_match
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

from extract_multiclass_features import (  # noqa: E402
    VARIANT_TO_CKPT, IMG_SIZE, CLASS_NAMES, load_sam2,
)
from extract_multiclass_features_aug import (  # noqa: E402
    preprocess_to_tensor, encode_flat,
)

DEFAULT_VIDEO = ("/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/"
                 "SUBJ_1b7d93c2_Y2025_DOY143/"
                 "M_10152025084201_U013419101506001_2_002_0001-01.MP4")
DEFAULT_AUG_TRAIN_DIR = os.path.join(SAM3_DIR, "multiclass_features_aug", "train")
DEFAULT_V1_TRAIN_DIR  = os.path.join(SAM3_DIR, "multiclass_features")
DEFAULT_OUT_DIR       = os.path.join(SAM3_DIR, "multiclass_crop_probe")
DEFAULT_REF_CASES     = "213,219,245"

WEIGHT_COLORS = {"v1": "tab:green", "clean": "tab:blue", "aug": "tab:orange"}


# ── Classifiers ───────────────────────────────────────────────────────────────

def fit_or_load_clf(which, args):
    """which: 'v1' (multiclass_features/, all rows are clean),
              'clean' (aug dir, variant==0), 'aug' (aug dir, all variants)."""
    import glob
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    stride = args.train_stride
    cache = os.path.join(args.out_dir, f"clf_{which}_stride{stride}.joblib")
    if os.path.exists(cache):
        print(f"  [{which}] loading cached classifier: {cache}")
        return joblib.load(cache)

    train_dir = args.v1_train_dir if which == "v1" else args.aug_train_dir
    files = sorted(glob.glob(os.path.join(train_dir, "case_*.npz")))
    if not files:
        raise SystemExit(f"no train features in {train_dir}")

    Xs, ys, ws = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X, y, w = d["features"], d["labels"], d["sample_weights"]
        if which == "clean":
            keep = d["variant"] == 0
            X, y, w = X[keep], y[keep], w[keep]
        Xs.append(X[::stride]); ys.append(y[::stride]); ws.append(w[::stride])
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    print(f"  [{which}] fitting LR on {X.shape[0]} x {X.shape[1]} "
          f"(stride={stride}, {len(files)} cases, {train_dir})")

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
    clf.fit(scaler.transform(X), y, sample_weight=w)
    joblib.dump((scaler, clf), cache)
    print(f"  [{which}] cached -> {cache}")
    return scaler, clf


# ── Histogram-match LUT ───────────────────────────────────────────────────────

def pooled_hist(frames):
    """Pooled per-channel 256-bin histogram over a list of BGR frames."""
    h = np.zeros((3, 256), dtype=np.float64)
    for fr in frames:
        for c in range(3):
            h[c] += cv2.calcHist([fr], [c], None, [256], [0, 256]).ravel()
    return h / h.sum(axis=1, keepdims=True)


def sample_video_frames(path, n, crop=None):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for fi in np.linspace(total * 0.05, total * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, fr = cap.read()
        if ret and fr is not None:
            out.append(crop_bbox(fr, crop) if crop else fr)
    cap.release()
    return out


def build_match_lut(args, box):
    """256-entry per-channel LUT mapping Sony-crop colours -> intuitive ref."""
    cache = os.path.join(args.out_dir, "hist_match_lut.npz")
    if os.path.exists(cache) and not args.rebuild_lut:
        d = np.load(cache)
        print(f"  LUT loaded from cache: {cache}")
        return d["lut"].astype(np.uint8)

    print(f"  building reference histogram from intuitive cases "
          f"{args.ref_cases} ({args.ref_frames_per_video} frames each) ...")
    ref_frames = []
    for cid in args.ref_cases.split(","):
        vp = os.path.join(VIDEO_DIR, f"case_{cid.strip()}_clipped.mp4")
        ref_frames += sample_video_frames(vp, args.ref_frames_per_video)
    print(f"  building source histogram from {os.path.basename(args.video)} "
          f"({args.src_hist_frames} cropped frames) ...")
    src_frames = sample_video_frames(args.video, args.src_hist_frames, crop=box)

    ref_cdf = np.cumsum(pooled_hist(ref_frames), axis=1)
    src_cdf = np.cumsum(pooled_hist(src_frames), axis=1)

    lut = np.zeros((3, 256), dtype=np.uint8)
    levels = np.arange(256)
    for c in range(3):
        # map v -> ref_cdf^{-1}(src_cdf(v))
        lut[c] = np.clip(np.interp(src_cdf[c], ref_cdf[c], levels), 0, 255)

    np.savez(cache, lut=lut)
    print(f"  LUT cached -> {cache}")
    return lut


def apply_match(frame_bgr, lut, unsharp_amount):
    out = np.empty_like(frame_bgr)
    for c in range(3):
        out[..., c] = cv2.LUT(frame_bgr[..., c], lut[c])
    if unsharp_amount > 0:
        blurred = cv2.GaussianBlur(out, (5, 5), 1.0)
        out = cv2.addWeighted(out, 1 + unsharp_amount, blurred, -unsharp_amount, 0)
    return out


# ── Frame sampling + encoding ─────────────────────────────────────────────────

def crop_bbox(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return frame[y0:y1, x0:x1]


@torch.no_grad()
def encode_window(video, box, t0, t1, model, device, args, lut):
    cap = cv2.VideoCapture(video)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur          = total_frames / max(fps, 1e-6)
    if t1 > dur:
        print(f"  [warn] window end {t1:.0f}s exceeds chunk duration {dur:.0f}s; clipping")
        t1 = dur
    times   = np.arange(t0, t1, 1.0 / args.sample_fps)
    indices = (times * fps).astype(int)
    print(f"  video: fps={fps:.2f} frames={total_frames} dur={dur:.0f}s  "
          f"window=[{t0:.0f},{t1:.0f}]s  n={len(indices)}")

    kinds = ["full", "crop"] + (["crop_match"] if lut is not None else [])
    feats = {k: [] for k in kinds}
    times_ok = []
    batch, meta = [], []
    dbg_saved = False

    def flush():
        if not batch:
            return
        out = encode_flat(model, torch.stack(batch), device, args.use_bf16)
        for kind, f in zip(meta, out):
            feats[kind].append(f)
        batch.clear(); meta.clear()

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    current = int(indices[0])
    for t, fi in zip(times, indices):
        fi = int(fi)
        while current < fi:
            cap.grab(); current += 1
        ret, frame = cap.read(); current += 1
        if not ret or frame is None:
            print(f"  [warn] failed read at t={t:.1f}s; skipping")
            continue
        times_ok.append(t)
        cropped = crop_bbox(frame, box)
        variants = {"full": frame, "crop": cropped}
        if lut is not None:
            variants["crop_match"] = apply_match(cropped, lut, args.unsharp)
            if not dbg_saved:
                cv2.imwrite(os.path.join(args.out_dir, "debug_crop.jpg"), cropped)
                cv2.imwrite(os.path.join(args.out_dir, "debug_crop_match.jpg"),
                            variants["crop_match"])
                print(f"  debug images -> {args.out_dir}/debug_crop[_match].jpg")
                dbg_saved = True
        for kind in kinds:
            batch.append(preprocess_to_tensor(variants[kind], IMG_SIZE))
            meta.append(kind)
            if len(batch) >= args.batch_size:
                flush()
    flush()
    cap.release()
    return np.array(times_ok), {k: np.stack(v) for k, v in feats.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",       default=DEFAULT_VIDEO)
    p.add_argument("--crop_box",    default="27,-1,1318,1068",
                   help="x0,y0,x1,y1 of the surgical view (native resolution)")
    p.add_argument("--event",       default="catheter_pull", choices=CLASS_NAMES)
    p.add_argument("--event_start", type=float, default=908.61,
                   help="GT event start, seconds within this chunk")
    p.add_argument("--event_end",   type=float, default=914.94)
    p.add_argument("--context",     type=float, default=120.0,
                   help="seconds of context on each side of the GT window")
    p.add_argument("--sample_fps",  type=float, default=1.0)

    p.add_argument("--weights",     default="v1,clean,aug",
                   help="comma list of classifier weight sets to evaluate")
    p.add_argument("--v1_train_dir",  default=DEFAULT_V1_TRAIN_DIR)
    p.add_argument("--aug_train_dir", default=DEFAULT_AUG_TRAIN_DIR)
    p.add_argument("--train_stride", type=int, default=2,
                   help="row subsample stride when fitting the LRs")
    p.add_argument("--max_iter",     type=int, default=300)

    p.add_argument("--hist_match",  action="store_true",
                   help="add crop_match input kind: crop + global histogram-"
                        "match LUT toward the intuitive reference")
    p.add_argument("--ref_cases",   default=DEFAULT_REF_CASES,
                   help="intuitive case ids for the reference histogram")
    p.add_argument("--ref_frames_per_video", type=int, default=30)
    p.add_argument("--src_hist_frames",      type=int, default=60)
    p.add_argument("--unsharp",     type=float, default=0.0,
                   help="unsharp amount applied after the LUT (e.g. 0.5); "
                        "0 disables")
    p.add_argument("--rebuild_lut", action="store_true")

    p.add_argument("--sam2_variant", default="small",
                   choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--use_bf16",    action="store_true")
    p.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    box = [int(v) for v in args.crop_box.split(",")]
    assert len(box) == 4, "--crop_box must be x0,y0,x1,y1"
    cls = CLASS_NAMES.index(args.event)
    weight_names = [w.strip() for w in args.weights.split(",")]

    print(f"video      : {args.video}")
    print(f"crop box   : {box}")
    print(f"event      : {args.event} (class {cls})  "
          f"GT [{args.event_start:.2f}, {args.event_end:.2f}]s  "
          f"context ±{args.context:.0f}s")
    print(f"weights    : {weight_names}   hist_match: {args.hist_match}"
          + (f" (unsharp={args.unsharp})" if args.hist_match else ""))

    # 1) classifiers (CPU; cached after first fit)
    print("\n[1/3] classifiers")
    clfs = {w: fit_or_load_clf(w, args) for w in weight_names}

    # 2) LUT (if requested) + encode probe window
    print("\n[2/3] encoding probe window")
    lut = build_match_lut(args, box) if args.hist_match else None
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    model  = load_sam2(args.sam2_variant, device)
    t0 = max(0.0, args.event_start - args.context)
    t1 = args.event_end + args.context
    times, feats = encode_window(args.video, box, t0, t1, model, device, args, lut)

    # 3) predict + report
    print("\n[3/3] results")
    inside = (times >= args.event_start) & (times <= args.event_end)
    combos = {}
    for wname, (scaler, clf) in clfs.items():
        for kind, F in feats.items():
            combos[f"{wname}/{kind}"] = clf.predict_proba(scaler.transform(F))

    print(f"\n  {'combo':<18} {'p(in)':>8} {'p(out)':>8} {'ratio':>7} "
          f"{'max_in':>8} {'max_out':>8} {'top class overall':<20}")
    print("  " + "-" * 84)
    for name, P in combos.items():
        pt = P[:, cls]
        p_in, p_out = pt[inside].mean(), pt[~inside].mean()
        top = CLASS_NAMES[np.bincount(P.argmax(1), minlength=len(CLASS_NAMES)).argmax()]
        print(f"  {name:<18} {p_in:8.4f} {p_out:8.4f} "
              f"{p_in / max(p_out, 1e-9):7.2f} "
              f"{pt[inside].max():8.4f} {pt[~inside].max():8.4f} {top:<20}")

    # CSV
    csv_path = os.path.join(args.out_dir, f"{args.event}_trace.csv")
    with open(csv_path, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["time_s", "inside_gt"]
                      + [f"p_{n.replace('/', '_')}" for n in combos]
                      + [f"top_{n.replace('/', '_')}" for n in combos])
        for i, t in enumerate(times):
            wcsv.writerow([f"{t:.2f}", int(inside[i])]
                          + [f"{P[i, cls]:.5f}" for P in combos.values()]
                          + [CLASS_NAMES[P[i].argmax()] for P in combos.values()])
    print(f"\n  trace -> {csv_path}")

    # Plot: one subplot per input kind, one line per weight set
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(feats.keys())
    fig, axes = plt.subplots(len(kinds), 1, figsize=(12, 3.2 * len(kinds)),
                             sharex=True, squeeze=False)
    for ax, kind in zip(axes[:, 0], kinds):
        for wname in weight_names:
            ax.plot(times, combos[f"{wname}/{kind}"][:, cls],
                    color=WEIGHT_COLORS.get(wname), label=f"{wname} weights")
        ax.axvspan(args.event_start, args.event_end, color="green", alpha=0.2,
                   label="GT window")
        ax.set_ylabel(f"p({args.event})")
        ax.set_title(f"input = {kind}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time in chunk (s)")
    fig.suptitle(os.path.basename(args.video))
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"{args.event}_trace.png")
    fig.savefig(png_path, dpi=130)
    print(f"  plot  -> {png_path}")


if __name__ == "__main__":
    main()
