#!/usr/bin/env python3
"""
localize_multiclass_aug.py
==========================
Train v1 multi-class softmax (multinomial logistic regression with sample_weight
for hard-negative mining) on augmented intuitive features, then localise all 5
milestone events on the OOD folder case via the v1 11-state Viterbi.

This is `localize_multiclass.py` minus the LOCO loop and minus the live encoder
pass on the held-out video:
  - train side: ALL augmented intuitive cases concatenated (multiclass_features_aug/train/)
  - val side:   pre-extracted OOD folder features (multiclass_features_aug/val/)

Decoder (smoothing + Viterbi + per-event extraction) is reused verbatim from
localize_multiclass.

Usage
-----
  python3 localize_multiclass_aug.py \\
      --train_dir multiclass_features_aug/train \\
      --val_npz   multiclass_features_aug/val/SUBJ_1b7d93c2_Y2025_DOY143.npz \\
      --event_csv event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_DIR)

# Reuse the v1 multiclass constants and decoder.
from localize_multiclass import (  # noqa: E402
    NUM_CLASSES, NUM_STATES, STATE_NAMES,
    EVENT_REPORT_ORDER, EVENT_STATE_NAMES,
    VAS_CSV_NAMES,
    pad_proba,
    smooth_log_probs,
    viterbi_decode,
    state_seq_to_event_windows,
    save_plot,
)


# ── Load augmented train features ─────────────────────────────────────────────

def load_train_features(train_dir):
    paths = sorted(glob.glob(os.path.join(train_dir, "case_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no train .npz files in {train_dir}")
    Xs, ys, ws, case_ids = [], [], [], []
    for p in paths:
        d   = np.load(p, allow_pickle=True)
        cid = str(d["case_id"])
        X   = d["features"].astype(np.float32)
        y   = d["labels"].astype(np.int32)
        w   = (d["sample_weights"].astype(np.float32)
               if "sample_weights" in d.files else np.ones(len(y), np.float32))
        k   = int(d["k_aug"]) if "k_aug" in d.files else 1
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        case_ids.append(cid)
        n_hn = int((w > 1.0).sum())
        hist = " ".join(f"{c}:{int((y == c).sum())}" for c in range(NUM_CLASSES)
                        if (y == c).any())
        print(f"  case {cid}: N={len(y)}  hard_neg={n_hn}  k_aug={k}  [{hist}]")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    return X, y, w, case_ids


def fit_classifier(X_train, y_train, w_train, C=1.0):
    from sklearn.linear_model import LogisticRegression
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_s  = (X_train - mean) / std
    clf  = LogisticRegression(
        C=C, max_iter=300, tol=1e-3, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )
    clf.fit(X_s, y_train, sample_weight=w_train)
    n_hn = int((w_train > 1.0).sum())
    print(f"  Trained on {len(y_train)} frames  hard_neg={n_hn}  "
          f"classes_seen={list(clf.classes_)}")
    return clf, mean, std


# ── OOD ground truth ──────────────────────────────────────────────────────────

def parse_ood_csv(path):
    """Returns {event_name: (start_s, end_s)} for the 5 milestones.
    vas_cut_1 and vas_cut_2 are merged into a single 'vas_cut' window
    (union of starts/ends), matching localize_multiclass.parse_all_events."""
    accepted = set(EVENT_REPORT_ORDER) | set(VAS_CSV_NAMES)
    by_event = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["event"].strip()
            if name not in accepted:
                continue
            if name in VAS_CSV_NAMES:
                name = "vas_cut"
            s = float(r["start_sec"])
            e = float(r["end_sec"])
            if name in by_event:
                old_s, old_e = by_event[name]
                by_event[name] = (min(s, old_s), max(e, old_e))
            else:
                by_event[name] = (s, e)
    return by_event


# ── Per-case summary ──────────────────────────────────────────────────────────

WITHIN_3MIN = 180.0


def print_summary(case_id, gt, pred, out_dir):
    lines = []
    lines.append(f"{'='*78}")
    lines.append(f"  Phase localisation (multiclass aug)  case={case_id}")
    lines.append(f"{'='*78}")
    lines.append(f"  {'Event':<16} {'GT (s)':>14} {'Pred (s)':>14} "
                 f"{'start err':>11} {'end err':>11}")
    lines.append(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*11} {'-'*11}")
    within = 0
    n_have = 0
    for name in EVENT_REPORT_ORDER:
        gt_w = gt.get(name)
        pr_w = pred.get(name)
        gt_str   = f"{gt_w[0]:.0f}-{gt_w[1]:.0f}" if gt_w is not None else "MISSING"
        pred_str = f"{pr_w[0]:.0f}-{pr_w[1]:.0f}" if pr_w is not None else "SKIPPED"
        if gt_w is not None and pr_w is not None:
            es = pr_w[0] - gt_w[0]
            ee = pr_w[1] - gt_w[1]
            es_str = f"{es:+.0f}s"
            ee_str = f"{ee:+.0f}s"
            n_have += 1
            if abs(es) <= WITHIN_3MIN:
                within += 1
        else:
            es_str = "--"
            ee_str = "--"
        lines.append(f"  {name:<16} {gt_str:>14} {pred_str:>14} "
                     f"{es_str:>11} {ee_str:>11}")
    lines.append("")
    lines.append(f"  Within 3 min: {within}/{n_have}")
    lines.append(f"{'='*78}")
    text = "\n".join(lines)
    print("\n" + text + "\n")
    path = os.path.join(out_dir, f"{case_id}_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  Summary -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",     default=None,
                   help="Default: <sam3>/multiclass_features_aug/train/")
    p.add_argument("--val_npz",       required=True,
                   help="Path to OOD val .npz from extract_multiclass_features_aug.py --mode val")
    p.add_argument("--event_csv",     required=True,
                   help="OOD event CSV (events_SUBJ_*.csv)")
    p.add_argument("--out_dir",       default=None,
                   help="Default: <sam3>/multiclass_localization_aug/")
    p.add_argument("--smooth_window", type=float, default=20.0,
                   help="Per-class log-prob rolling-mean window in seconds.")
    p.add_argument("--alpha",         type=float, default=0.0,
                   help="Viterbi event-anchor log-prob bonus.")
    p.add_argument("--skip_cost",     type=float, default=0.0,
                   help="Log-prob added to Viterbi skip edges (<=0 only).")
    p.add_argument("--C",             type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.train_dir is None:
        args.train_dir = os.path.join(SAM3_DIR, "multiclass_features_aug", "train")
    if args.out_dir is None:
        args.out_dir   = os.path.join(SAM3_DIR, "multiclass_localization_aug")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Train dir: {args.train_dir}")
    print(f"Val npz:   {args.val_npz}")
    print(f"Event csv: {args.event_csv}\n")

    print("Loading train features ...")
    X_train, y_train, w_train, case_ids = load_train_features(args.train_dir)
    print(f"\n  Total train: {len(y_train)} frames across {len(case_ids)} cases  "
          f"feat_dim={X_train.shape[1]}  hard_neg={int((w_train > 1.0).sum())}")

    print("\nFitting multinomial logistic regression "
          "(class_weight='balanced', sample_weight=hard_neg) ...")
    clf, mean, std = fit_classifier(X_train, y_train, w_train, C=args.C)

    print(f"\nLoading val features: {args.val_npz}")
    d       = np.load(args.val_npz, allow_pickle=True)
    X_val   = d["features"].astype(np.float32)
    times_s = d["times_s"].astype(np.float32)
    case_id = str(d["case_id"])
    print(f"  N={len(times_s)}  duration={times_s[-1]:.1f}s  feat_dim={X_val.shape[1]}")

    order   = np.argsort(times_s)
    X_val   = X_val[order]
    times_s = times_s[order]

    X_val_s   = (X_val - mean) / std
    proba     = pad_proba(clf.predict_proba(X_val_s), clf.classes_)
    log_probs = np.log(np.clip(proba, 1e-12, 1.0))

    # Per-class rolling mean smoothing.
    dt            = float(np.median(np.diff(times_s))) if len(times_s) > 1 else 1.0
    sample_fps    = 1.0 / max(dt, 1e-6)
    window_frames = max(1, int(round(args.smooth_window * sample_fps)))
    log_probs_sm  = smooth_log_probs(log_probs, window_frames)
    print(f"  sample_fps (inferred): {sample_fps:.2f}  smooth_window: "
          f"{args.smooth_window}s = {window_frames} frames")

    state_seq    = viterbi_decode(log_probs_sm, alpha=args.alpha,
                                  skip_cost=args.skip_cost)
    pred_windows = state_seq_to_event_windows(state_seq, times_s)
    gt_windows   = parse_ood_csv(args.event_csv)

    print(f"\n  GT windows:")
    for name in EVENT_REPORT_ORDER:
        gt = gt_windows.get(name)
        if gt:
            print(f"    {name:<14}: {gt[0]:.0f}s-{gt[1]:.0f}s  "
                  f"({gt[0]/60:.1f}-{gt[1]/60:.1f} min)")
        else:
            print(f"    {name:<14}: MISSING")

    print(f"\n  Predicted windows:")
    for name in EVENT_REPORT_ORDER:
        pr = pred_windows.get(name)
        if pr is None:
            print(f"    {name:<14}: SKIPPED by decoder")
            continue
        print(f"    {name:<14}: {pr[0]:.0f}s-{pr[1]:.0f}s  "
              f"({pr[0]/60:.1f}-{pr[1]/60:.1f} min)")

    print_summary(case_id, gt_windows, pred_windows, args.out_dir)

    out_png = os.path.join(args.out_dir, f"{case_id}_multiclass_aug.png")
    save_plot(times_s, log_probs_sm, state_seq, gt_windows, pred_windows,
              case_id, out_png)

    out_npz = os.path.join(args.out_dir, f"{case_id}_multiclass_aug.npz")
    np.savez_compressed(out_npz,
                        times_s          = times_s,
                        log_probs_smooth = log_probs_sm.astype(np.float32),
                        state_seq        = state_seq.astype(np.int32))
    print(f"  Probs/states -> {out_npz}")


if __name__ == "__main__":
    main()
