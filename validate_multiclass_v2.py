#!/usr/bin/env python3
"""
validate_multiclass_v2.py
==========================
OOD validation of a v2 checkpoint on the multi-chunk folder case
(SUBJ_1b7d93c2 by default) — produces predictions.json, plot.png, and a
per-event error table.

Pipeline
--------
1. Load checkpoint from train_multiclass_v2.py. Rebuild AttnPoolBiGRU from
   the saved model_kwargs.
2. Load the val-mode .npz produced by extract_multiclass_features_v2.py
   --mode val on the OOD folder. Features are already stitched into a single
   surgery-absolute timeline (times_s).
3. Forward through model.pool_only() → pooled features (N, 2*embed_dim).
4. Test-time burn-in normalisation: fit μ, σ on pooled features from the
   first --burn_in_seconds of the timeline (default 60 s = first 60 frames
   at 1 fps). Freeze μ, σ; normalise the WHOLE timeline.
   Why pool-only and not after the GRU: the burn-in's purpose is to absorb
   the per-video distribution shift in the encoder/pool output. Doing it
   AFTER the GRU would leave the recurrent state conditioned on
   un-normalised stats.
5. Forward through model.gru_and_head() over the full sequence in one shot
   (one BiGRU forward, no windowing — keeps temporal context intact).
6. log_softmax → per-frame log-emissions.
7. Optional per-class rolling-mean smoothing (lifted from
   localize_multiclass.smooth_log_probs) and Viterbi decoding
   (localize_multiclass.viterbi_decode) — unchanged from v1.
8. Map decoded state sequence to per-event (start_s, end_s) windows
   (localize_multiclass.state_seq_to_event_windows).
9. Score per-event start/end errors against events_SUBJ_*.csv.
10. Save predictions.json, plot.png (reuses v1 save_plot), and
    per_event_errors.txt.

Usage
-----
  python3 validate_multiclass_v2.py \\
      --ckpt    multiclass_checkpoints_v2/20260609_1530/checkpoint_best.pt \\
      --val_features multiclass_features_v2/val/SUBJ_1b7d93c2_Y2025_DOY143.npz \\
      --event_csv    event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_DIR)

from multiclass_model import AttnPoolBiGRU, NUM_CLASSES
from localize_multiclass import (
    CLASS_NAMES,
    EVENT_REPORT_ORDER,
    VAS_CSV_NAMES,
    smooth_log_probs,
    viterbi_decode,
    state_seq_to_event_windows,
    save_plot,
)

DEFAULT_VAL_FEATURES = os.path.join(
    SAM3_DIR, "multiclass_features_v2", "val",
    "SUBJ_1b7d93c2_Y2025_DOY143.npz",
)
DEFAULT_EVENT_CSV = os.path.join(
    SAM3_DIR, "event_annotations",
    "events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv",
)
DEFAULT_OUT_DIR = os.path.join(SAM3_DIR, "multiclass_validation_v2")

WITHIN_3MIN = 180.0


# ── Loading helpers ──────────────────────────────────────────────────────────

def parse_val_events(csv_path):
    """events_SUBJ_*.csv → dict[event_name] = (start_s, end_s).
    vas_cut_1/2 merged into a single 'vas_cut' window (union)."""
    gt = {}
    accepted = set(EVENT_REPORT_ORDER) | set(VAS_CSV_NAMES)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["event"].strip()
            if name not in accepted:
                continue
            if name in VAS_CSV_NAMES:
                name = "vas_cut"
            s = float(r["start_sec"])
            e = float(r["end_sec"])
            if name in gt:
                old_s, old_e = gt[name]
                gt[name] = (min(s, old_s), max(e, old_e))
            else:
                gt[name] = (s, e)
    return gt


def load_val_features(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "case_id":   str(d["case_id"]),
        "fpn1":      np.asarray(d["features_fpn1"]),     # (N, 1, C1, H1, W1) fp16
        "fpn2":      np.asarray(d["features_fpn2"]),     # (N, 1, C2, H2, W2) fp16
        "labels":    np.asarray(d["labels"]),
        "times_s":   np.asarray(d["times_s"]),
        "K":         int(d["K"]),
        "chunks":    list(d["chunks"]) if "chunks" in d.files else [],
        "event_csv": str(d["event_csv"]) if "event_csv" in d.files else "",
    }


def build_model_from_ckpt(ckpt_path, device):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    kwargs = ckpt["model_kwargs"]
    model  = AttnPoolBiGRU(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ── Inference passes ─────────────────────────────────────────────────────────

@torch.no_grad()
def pool_full_sequence(model, fpn1, fpn2, device, batch_size=64):
    """Run model.pool_only over the full sequence in chunks of batch_size
    frames (one (1, T=batch_size, ...) call per chunk). Returns
    (N, 2*embed_dim) as a CPU float32 numpy array.

    The val .npz stores K=1 so we index [:, 0] to get the (N, C, H, W) slice."""
    N    = fpn1.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e  = min(s + batch_size, N)
        f1 = torch.from_numpy(fpn1[s:e, 0].astype(np.float32)).unsqueeze(0).to(device)
        f2 = torch.from_numpy(fpn2[s:e, 0].astype(np.float32)).unsqueeze(0).to(device)
        z  = model.pool_only(f1, f2)             # (1, T, 2*embed_dim)
        outs.append(z.squeeze(0).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def tt_burn_in_normalise(pooled, times_s, burn_in_seconds, skip_initial=5):
    """Fit μ, σ on pooled features in [times_s[skip_initial],
    times_s[skip_initial] + burn_in_seconds). Returns normalised pooled
    features (same shape) plus the (μ, σ, n_burn_samples) stats."""
    skip_initial = min(skip_initial, len(times_s) - 1)
    t0 = float(times_s[skip_initial])
    burn_mask = (times_s >= t0) & (times_s < t0 + burn_in_seconds)
    if burn_mask.sum() < 5:
        burn_mask = np.zeros_like(burn_mask, dtype=bool)
        burn_mask[skip_initial : skip_initial + 5] = True
    mu     = pooled[burn_mask].mean(axis=0).astype(np.float64)
    sigma  = pooled[burn_mask].std(axis=0).astype(np.float64) + 1e-6
    normed = ((pooled.astype(np.float64) - mu) / sigma).astype(np.float32)
    return normed, mu.astype(np.float32), sigma.astype(np.float32), int(burn_mask.sum())


@torch.no_grad()
def forward_gru_head(model, pooled, device):
    """Run model.gru_and_head over the full sequence in one BiGRU pass.
    For typical OOD sequences (~7200 frames at 1 fps, 128-d pooled features)
    the GRU activation footprint is a few MB — no need to chunk."""
    p = torch.from_numpy(pooled).unsqueeze(0).to(device)   # (1, N, 2E)
    logits = model.gru_and_head(p)                          # (1, N, K)
    return logits.squeeze(0).float().cpu().numpy()


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_events(pred_windows, gt):
    """Returns dict[event_name] → {gt, pred, err_start, err_end, status}."""
    out = {}
    for name in EVENT_REPORT_ORDER:
        gt_w = gt.get(name)
        pr_w = pred_windows.get(name)
        if pr_w is None and gt_w is None:
            out[name] = {"gt": None, "pred": None,
                         "err_start_s": None, "err_end_s": None,
                         "status": "both_missing"}
        elif pr_w is None:
            out[name] = {"gt": gt_w, "pred": None,
                         "err_start_s": None, "err_end_s": None,
                         "status": "pred_skipped"}
        elif gt_w is None:
            out[name] = {"gt": None, "pred": pr_w,
                         "err_start_s": None, "err_end_s": None,
                         "status": "no_gt"}
        else:
            out[name] = {
                "gt": gt_w, "pred": pr_w,
                "err_start_s": pr_w[0] - gt_w[0],
                "err_end_s":   pr_w[1] - gt_w[1],
                "status": "ok",
            }
    return out


def format_error_table(case_id, scored, out_path):
    lines = []
    lines.append(f"{'='*82}")
    lines.append(f"  OOD VALIDATION  case={case_id}")
    lines.append(f"{'='*82}")
    lines.append(f"  {'Event':<16}  {'GT (s)':>13}  {'Pred (s)':>13}  "
                 f"{'err_start':>10}  {'err_end':>9}")
    lines.append(f"  {'-'*16}  {'-'*13}  {'-'*13}  {'-'*10}  {'-'*9}")
    for name in EVENT_REPORT_ORDER:
        r = scored[name]
        if r["status"] == "pred_skipped":
            gt_s, gt_e = r["gt"]
            lines.append(f"  {name:<16}  {gt_s:>5.0f}–{gt_e:<6.0f}  "
                         f"{'SKIPPED':>13}  {'':>10}  {'':>9}")
        elif r["status"] in ("both_missing", "no_gt"):
            lines.append(f"  {name:<16}  {'(no GT)':>13}  ...")
        else:
            gt_s, gt_e = r["gt"]
            pr_s, pr_e = r["pred"]
            within = "✓" if abs(r["err_start_s"]) <= WITHIN_3MIN else " "
            lines.append(f"  {name:<16}  {gt_s:>5.0f}–{gt_e:<6.0f}  "
                         f"{pr_s:>5.0f}–{pr_e:<6.0f}  "
                         f"{r['err_start_s']:>+8.0f}s {within}  "
                         f"{r['err_end_s']:>+7.0f}s")
    abs_starts = [abs(r["err_start_s"]) for r in scored.values()
                  if r["err_start_s"] is not None]
    abs_ends   = [abs(r["err_end_s"])   for r in scored.values()
                  if r["err_end_s"]   is not None]
    n_within   = sum(1 for v in abs_starts if v <= WITHIN_3MIN)
    n_total    = len(abs_starts)
    lines.append(f"{'-'*82}")
    if abs_starts:
        lines.append(f"  Aggregate |start err|:  median={np.median(abs_starts):.0f}s  "
                     f"mean={np.mean(abs_starts):.0f}s  max={np.max(abs_starts):.0f}s  "
                     f"within_3min={n_within}/{n_total}")
        lines.append(f"  Aggregate |end   err|:  median={np.median(abs_ends):.0f}s  "
                     f"mean={np.mean(abs_ends):.0f}s  max={np.max(abs_ends):.0f}s")
    lines.append(f"{'='*82}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    return text


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="checkpoint_best.pt from train_multiclass_v2.py")
    p.add_argument("--val_features", default=DEFAULT_VAL_FEATURES,
                   help="Val .npz from extract_multiclass_features_v2.py --mode val")
    p.add_argument("--event_csv",    default=DEFAULT_EVENT_CSV,
                   help="events_SUBJ_*.csv with GT timestamps (surgery-absolute).")
    p.add_argument("--out_dir",      default=DEFAULT_OUT_DIR,
                   help="Output directory; per-case subdir created under here.")
    p.add_argument("--out_tag",      default=None,
                   help="Subdir under --out_dir. Defaults to the val .npz stem.")
    # Decoder knobs (defaults match v1)
    p.add_argument("--burn_in_seconds", type=float, default=60.0,
                   help="Test-time burn-in window for μ/σ fit on pooled features.")
    p.add_argument("--smooth_window",   type=float, default=20.0,
                   help="Per-class log-prob rolling-mean window in seconds.")
    p.add_argument("--alpha",           type=float, default=0.0,
                   help="Event-anchor log-prob bonus added in Viterbi.")
    p.add_argument("--skip_cost",       type=float, default=0.0,
                   help="Log-prob added when taking a skip edge in Viterbi "
                        "(negative discourages skipping; 0 is purely data-driven).")
    p.add_argument("--pool_batch_size", type=int,   default=64,
                   help="Frames per AttnPool forward chunk.")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model + features + GT ─────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    model, ckpt = build_model_from_ckpt(args.ckpt, device)
    pc = model.count_params()
    print(f"  Model: {pc['total']:,} params  "
          f"(fpn1={ckpt['fpn1_channels']}ch  fpn2={ckpt['fpn2_channels']}ch)")
    if "val_acc" in ckpt:
        print(f"  Train-time best val_acc: {ckpt['val_acc']:.4f}  "
              f"(epoch {ckpt.get('epoch','?')})")

    print(f"\nLoading val features: {args.val_features}")
    vf = load_val_features(args.val_features)
    print(f"  case_id={vf['case_id']}  N={vf['fpn1'].shape[0]}  "
          f"K={vf['K']}  duration={vf['times_s'][-1]:.0f}s "
          f"({vf['times_s'][-1]/60:.1f} min)  "
          f"chunks={len(vf['chunks'])}")

    print(f"\nLoading GT events: {args.event_csv}")
    gt = parse_val_events(args.event_csv)
    for name in EVENT_REPORT_ORDER:
        if name in gt:
            s, e = gt[name]
            print(f"  GT {name:<14}: {s:>7.1f}s – {e:<7.1f}s  "
                  f"({s/60:.1f}–{e/60:.1f} min)")
        else:
            print(f"  GT {name:<14}: MISSING")

    # ── Pool all frames ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pooled = pool_full_sequence(model, vf["fpn1"], vf["fpn2"], device,
                                 batch_size=args.pool_batch_size)
    print(f"\nPooled features: {pooled.shape}  dtype={pooled.dtype}  "
          f"({time.perf_counter()-t0:.1f}s)")

    # ── TT burn-in ─────────────────────────────────────────────────────────
    pooled_n, mu, sigma, n_burn = tt_burn_in_normalise(
        pooled, vf["times_s"], args.burn_in_seconds,
    )
    print(f"Burn-in: fit μ/σ on {n_burn} samples "
          f"(first {args.burn_in_seconds:.0f}s of timeline)  "
          f"μ_norm={np.linalg.norm(mu):.3f}  σ_mean={np.mean(sigma):.3f}")

    # ── GRU + head over full sequence ──────────────────────────────────────
    t1 = time.perf_counter()
    logits = forward_gru_head(model, pooled_n, device)   # (N, NUM_CLASSES)
    log_emit = logits - np.max(logits, axis=1, keepdims=True)   # numerical
    log_emit = log_emit - np.log(np.exp(log_emit).sum(axis=1, keepdims=True))
    print(f"GRU + head over {logits.shape[0]} frames: "
          f"{time.perf_counter()-t1:.1f}s")

    # ── Per-class smoothing + Viterbi ──────────────────────────────────────
    # Sample period derived from times_s (median Δt). The val .npz stores
    # variable fps across chunks, but the sampling stride was uniform per
    # chunk (sample_fps=1.0 by default) so times_s[i+1]-times_s[i] ≈ 1.0.
    dt = float(np.median(np.diff(vf["times_s"])))
    window_frames = max(1, int(round(args.smooth_window / max(dt, 1e-6))))
    log_emit_sm = smooth_log_probs(log_emit, window_frames)
    print(f"Smoothing window: {args.smooth_window:.0f}s = {window_frames} frames "
          f"(median Δt={dt:.2f}s)")

    state_seq    = viterbi_decode(log_emit_sm,
                                   alpha=args.alpha, skip_cost=args.skip_cost)
    pred_windows = state_seq_to_event_windows(state_seq, vf["times_s"])

    # ── Score + report ─────────────────────────────────────────────────────
    scored = score_events(pred_windows, gt)

    out_tag = args.out_tag or os.path.splitext(os.path.basename(args.val_features))[0]
    case_out = os.path.join(args.out_dir, out_tag)
    os.makedirs(case_out, exist_ok=True)

    # plot (reuse v1 save_plot — it draws emissions, GT shading, Viterbi states)
    out_png = os.path.join(case_out, "plot.png")
    save_plot(vf["times_s"], log_emit_sm, state_seq, gt, pred_windows,
              vf["case_id"], out_png)

    txt_path = os.path.join(case_out, "per_event_errors.txt")
    format_error_table(vf["case_id"], scored, txt_path)

    # predictions.json
    json_path = os.path.join(case_out, "predictions.json")
    payload = {
        "ckpt":               args.ckpt,
        "val_features":       args.val_features,
        "event_csv":          args.event_csv,
        "case_id":            vf["case_id"],
        "chunks":             vf["chunks"],
        "n_frames":           int(vf["fpn1"].shape[0]),
        "duration_s":         float(vf["times_s"][-1]),
        "burn_in_seconds":    args.burn_in_seconds,
        "burn_in_n_samples":  n_burn,
        "smooth_window_s":    args.smooth_window,
        "viterbi_alpha":      args.alpha,
        "viterbi_skip_cost":  args.skip_cost,
        "events": {
            name: {
                "gt":           ({"start_s": float(r["gt"][0]),
                                  "end_s":   float(r["gt"][1])}
                                 if r["gt"] is not None else None),
                "pred":         ({"start_s": float(r["pred"][0]),
                                  "end_s":   float(r["pred"][1])}
                                 if r["pred"] is not None else None),
                "err_start_s":  (float(r["err_start_s"])
                                 if r["err_start_s"] is not None else None),
                "err_end_s":    (float(r["err_end_s"])
                                 if r["err_end_s"]   is not None else None),
                "status":       r["status"],
            }
            for name, r in scored.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  predictions  → {json_path}")
    print(f"  plot         → {out_png}")
    print(f"  errors txt   → {txt_path}")


if __name__ == "__main__":
    main()
