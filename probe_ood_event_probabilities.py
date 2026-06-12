#!/usr/bin/env python3
"""
probe_ood_event_probabilities.py
================================
Smoking-gun probe for v2 phase emissions on the annotated OOD case.

This is deliberately not a localisation script. It extracts only short
windows around known event timestamps, runs the trained emission model, and
plots/checks whether the target class probability spikes inside the annotated
window.

Example:
  python3 probe_ood_event_probabilities.py \
      --ckpt multiclass_checkpoints_v2/smoke/checkpoint_best.pt \
      --video_folder /path/to/SUBJ_1b7d93c2_Y2025_DOY143 \
      --event_csv event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv \
      --context_seconds 120 \
      --sample_fps 1.0 \
      --batch_size 2 \
      --use_bf16
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_DIR)

from multiclass_model import AttnPoolBiGRU, NUM_CLASSES  # noqa: E402
from extract_multiclass_features_v2 import (  # noqa: E402
    CLASS_NAMES,
    EVENT_TO_CLASS,
    IMG_SIZE,
    VAS_CSV_NAMES,
    VARIANT_TO_CKPT,
    apply_sbs,
    discover_chunks,
    extract_batch_spatial,
    load_sam2,
    preprocess_to_tensor,
)


DEFAULT_OOD_FOLDER = (
    "/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/"
    "SUBJ_1b7d93c2_Y2025_DOY143"
)
DEFAULT_EVENT_CSV = os.path.join(
    SAM3_DIR,
    "event_annotations",
    "events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv",
)
DEFAULT_OUT_DIR = os.path.join(SAM3_DIR, "multiclass_smoking_gun", "probe")


@dataclass
class ChunkInfo:
    path: str
    basename: str
    fps: float
    total_frames: int
    duration_s: float
    offset_s: float


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def parse_events(event_csv):
    events = []
    with open(event_csv, newline="") as f:
        for row in csv.DictReader(f):
            raw_name = row["event"].strip()
            target_name = "vas_cut" if raw_name in VAS_CSV_NAMES else raw_name
            if target_name not in EVENT_TO_CLASS:
                continue
            start_s = float(row["start_sec"])
            end_s = float(row["end_sec"])
            events.append({
                "raw_event": raw_name,
                "target_event": target_name,
                "target_class": int(EVENT_TO_CLASS[target_name]),
                "start_s": start_s,
                "end_s": end_s,
            })
    events.sort(key=lambda e: e["start_s"])
    return events


def build_timeline(folder, pattern):
    paths = discover_chunks(folder, pattern)
    chunks = []
    offset = 0.0
    for path in paths:
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"invalid video metadata: {path}")
        duration = total_frames / fps
        chunks.append(ChunkInfo(
            path=path,
            basename=os.path.basename(path),
            fps=fps,
            total_frames=total_frames,
            duration_s=duration,
            offset_s=offset,
        ))
        offset += duration
    return chunks


def event_sample_points(chunks, start_s, end_s, sample_fps):
    step = 1.0 / sample_fps
    points = []
    for ci, chunk in enumerate(chunks):
        chunk_start = chunk.offset_s
        chunk_end = chunk.offset_s + chunk.duration_s
        ov_start = max(start_s, chunk_start)
        ov_end = min(end_s, chunk_end)
        if ov_end < ov_start:
            continue

        seen_frames = set()
        t = ov_start
        while t <= ov_end + 1e-6:
            local_t = t - chunk.offset_s
            frame_idx = int(round(local_t * chunk.fps))
            frame_idx = min(max(frame_idx, 0), chunk.total_frames - 1)
            if frame_idx not in seen_frames:
                seen_frames.add(frame_idx)
                points.append((ci, frame_idx, float(t)))
            t += step
    points.sort(key=lambda x: x[2])
    return points


def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AttnPoolBiGRU(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_pooled_for_event(sam2_model, clf_model, chunks, points, args, device):
    by_chunk = {}
    for ci, frame_idx, t_abs in points:
        by_chunk.setdefault(ci, []).append((frame_idx, t_abs))

    pooled_parts = []
    times = []
    batch_tensors = []
    batch_times = []

    def flush():
        if not batch_tensors:
            return
        frames = torch.stack(batch_tensors)
        fpn1, fpn2 = extract_batch_spatial(
            sam2_model, frames, device, args.use_bf16,
        )
        f1 = torch.from_numpy(fpn1.astype(np.float32)).unsqueeze(0).to(device)
        f2 = torch.from_numpy(fpn2.astype(np.float32)).unsqueeze(0).to(device)
        z = clf_model.pool_only(f1, f2).squeeze(0).float().cpu().numpy()
        pooled_parts.append(z)
        times.extend(batch_times)
        batch_tensors.clear()
        batch_times.clear()

    for ci in sorted(by_chunk):
        chunk = chunks[ci]
        chunk_points = sorted(by_chunk[ci], key=lambda x: x[0])
        cap = cv2.VideoCapture(chunk.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(chunk_points[0][0]))
        current = int(chunk_points[0][0])

        for frame_idx, t_abs in chunk_points:
            frame_idx = int(frame_idx)
            while current < frame_idx:
                cap.grab()
                current += 1
            ret, frame_bgr = cap.read()
            current += 1
            if not ret or frame_bgr is None:
                frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            frame_bgr = apply_sbs(frame_bgr, args.sbs_eye)
            batch_tensors.append(preprocess_to_tensor(frame_bgr, IMG_SIZE))
            batch_times.append(float(t_abs))
            if len(batch_tensors) >= args.batch_size:
                flush()
        cap.release()

    flush()
    if not pooled_parts:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    pooled = np.concatenate(pooled_parts, axis=0).astype(np.float32)
    times_arr = np.array(times, dtype=np.float32)
    order = np.argsort(times_arr)
    return pooled[order], times_arr[order]


def maybe_normalise_pooled(pooled, times_s, event, mode):
    if mode == "none" or len(times_s) == 0:
        return pooled, {"mode": "none", "n": 0}
    context = (times_s < event["start_s"]) | (times_s > event["end_s"])
    if context.sum() < 5:
        context = np.ones_like(times_s, dtype=bool)
    mu = pooled[context].mean(axis=0, keepdims=True)
    sigma = pooled[context].std(axis=0, keepdims=True) + 1e-6
    return ((pooled - mu) / sigma).astype(np.float32), {
        "mode": mode,
        "n": int(context.sum()),
        "mu_norm": float(np.linalg.norm(mu)),
        "sigma_mean": float(np.mean(sigma)),
    }


@torch.no_grad()
def forward_probs(clf_model, pooled, device):
    if pooled.shape[0] == 0:
        return np.empty((0, NUM_CLASSES), dtype=np.float32)
    z = torch.from_numpy(pooled).unsqueeze(0).to(device)
    logits = clf_model.gru_and_head(z).squeeze(0)
    return torch.softmax(logits, dim=-1).float().cpu().numpy()


def smooth_probs(probs, times_s, smooth_seconds):
    if smooth_seconds <= 0 or len(times_s) < 3:
        return probs
    dt = float(np.median(np.diff(times_s)))
    if dt <= 0:
        return probs
    win = max(1, int(round(smooth_seconds / dt)))
    if win <= 1:
        return probs
    kernel = np.ones(win, dtype=np.float32) / float(win)
    out = np.empty_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = np.convolve(probs[:, c], kernel, mode="same")
    return out


def summarise_event(event, times_s, probs, probs_smooth):
    cls = event["target_class"]
    inside = (times_s >= event["start_s"]) & (times_s <= event["end_s"])
    outside = ~inside
    p = probs_smooth[:, cls] if len(times_s) else np.array([], dtype=np.float32)
    if len(p) == 0:
        return {
            **event,
            "n_samples": 0,
            "status": "empty",
        }

    max_idx = int(np.argmax(p))
    center = 0.5 * (event["start_s"] + event["end_s"])
    center_idx = int(np.argmin(np.abs(times_s - center)))
    return {
        **event,
        "n_samples": int(len(times_s)),
        "status": "ok",
        "target_class_name": CLASS_NAMES[cls],
        "max_target_prob": float(p[max_idx]),
        "max_target_prob_time_s": float(times_s[max_idx]),
        "max_target_prob_rel_to_start_s": float(times_s[max_idx] - event["start_s"]),
        "max_target_prob_inside_gt": bool(inside[max_idx]),
        "max_inside_gt": float(np.max(p[inside])) if inside.any() else None,
        "mean_inside_gt": float(np.mean(p[inside])) if inside.any() else None,
        "max_outside_gt": float(np.max(p[outside])) if outside.any() else None,
        "mean_outside_gt": float(np.mean(p[outside])) if outside.any() else None,
        "top_class_at_gt_center": CLASS_NAMES[int(np.argmax(probs_smooth[center_idx]))],
        "top_prob_at_gt_center": float(np.max(probs_smooth[center_idx])),
        "target_prob_at_gt_center": float(probs_smooth[center_idx, cls]),
    }


def write_event_csv(path, event, times_s, probs, probs_smooth):
    cls = event["target_class"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s",
            "rel_to_event_start_s",
            "inside_gt",
            "target_class",
            "target_prob",
            "target_prob_smooth",
            "top_class_smooth",
            "top_prob_smooth",
        ])
        for i, t in enumerate(times_s):
            top = int(np.argmax(probs_smooth[i]))
            inside = event["start_s"] <= t <= event["end_s"]
            w.writerow([
                f"{float(t):.3f}",
                f"{float(t - event['start_s']):.3f}",
                int(inside),
                CLASS_NAMES[cls],
                f"{float(probs[i, cls]):.6f}",
                f"{float(probs_smooth[i, cls]):.6f}",
                CLASS_NAMES[top],
                f"{float(probs_smooth[i, top]):.6f}",
            ])


def write_summary_text(path, summaries):
    lines = []
    lines.append("OOD event-window probability probe")
    lines.append("=" * 88)
    lines.append(
        f"{'event':<15} {'class':<15} {'n':>5} {'max_in':>8} "
        f"{'max_out':>8} {'mean_in':>8} {'t_max_rel':>10} {'inside?':>8} "
        f"{'top@center':<18} {'p_target@center':>15}"
    )
    lines.append("-" * 88)
    for s in summaries:
        if s.get("status") != "ok":
            lines.append(f"{s['raw_event']:<15} {'EMPTY':<15} {0:>5}")
            continue
        lines.append(
            f"{s['raw_event']:<15} {s['target_class_name']:<15} "
            f"{s['n_samples']:>5} "
            f"{s['max_inside_gt'] if s['max_inside_gt'] is not None else float('nan'):>8.3f} "
            f"{s['max_outside_gt'] if s['max_outside_gt'] is not None else float('nan'):>8.3f} "
            f"{s['mean_inside_gt'] if s['mean_inside_gt'] is not None else float('nan'):>8.3f} "
            f"{s['max_target_prob_rel_to_start_s']:>10.1f} "
            f"{str(s['max_target_prob_inside_gt']):>8} "
            f"{s['top_class_at_gt_center']:<18} "
            f"{s['target_prob_at_gt_center']:>15.3f}"
        )
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")


def write_plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plot: {exc}")
        return

    n = len(results)
    fig_h = max(2.4 * n, 4.0)
    fig, axes = plt.subplots(n, 1, figsize=(12, fig_h), squeeze=False)
    axes = axes[:, 0]
    for ax, result in zip(axes, results):
        event = result["event"]
        times = result["times_s"]
        probs_s = result["probs_smooth"]
        cls = event["target_class"]
        if len(times) == 0:
            ax.set_title(f"{event['raw_event']} (no samples)")
            continue
        target_p = probs_s[:, cls]
        other = probs_s.copy()
        other[:, cls] = -np.inf
        max_other = np.max(other, axis=1)
        ax.plot(times, target_p, label=f"{CLASS_NAMES[cls]} prob", lw=2)
        ax.plot(times, max_other, label="max other class", lw=1, alpha=0.65)
        ax.axvspan(event["start_s"], event["end_s"], color="tab:green",
                   alpha=0.18, label="GT event")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("prob")
        ax.set_title(
            f"{event['raw_event']} -> {CLASS_NAMES[cls]} "
            f"({event['start_s']:.1f}-{event['end_s']:.1f}s)"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("surgery-absolute time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--video_folder", default=DEFAULT_OOD_FOLDER)
    p.add_argument("--event_csv", default=DEFAULT_EVENT_CSV)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--context_seconds", type=float, default=120.0)
    p.add_argument("--sample_fps", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--smooth_seconds", type=float, default=15.0)
    p.add_argument("--tt_norm", choices=["none", "context"], default="none",
                   help="For this probe, 'none' is closest to training. "
                        "'context' z-scores pooled features using non-GT "
                        "frames in each snippet.")
    p.add_argument("--video_pattern", default="*.MP4")
    p.add_argument("--sbs_eye", default="none", choices=["left", "right", "none"])
    p.add_argument("--sam2_variant", default="small",
                   choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--use_bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample_fps must be > 0")
    os.makedirs(args.out_dir, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"OOD folder: {args.video_folder}")
    print(f"Event CSV:  {args.event_csv}")
    print(f"Out dir:    {args.out_dir}")

    events = parse_events(args.event_csv)
    if not events:
        raise SystemExit(f"no supported events in {args.event_csv}")
    print(f"\nEvents ({len(events)}):")
    for e in events:
        print(f"  {e['raw_event']:<14} -> {CLASS_NAMES[e['target_class']]:<14} "
              f"{e['start_s']:.1f}-{e['end_s']:.1f}s")

    chunks = build_timeline(args.video_folder, args.video_pattern)
    print(f"\nChunks ({len(chunks)}):")
    for c in chunks:
        print(f"  {c.basename}  fps={c.fps:.2f} frames={c.total_frames} "
              f"offset={c.offset_s:.1f}s dur={c.duration_s:.1f}s")

    clf_model, ckpt = load_classifier(args.ckpt, device)
    print(f"\nClassifier loaded: epoch={ckpt.get('epoch', '?')} "
          f"model_kwargs={ckpt.get('model_kwargs', {})}")
    sam2_model = load_sam2(args.sam2_variant, device)

    summaries = []
    plot_results = []
    t_all = time.perf_counter()
    for idx, event in enumerate(events, start=1):
        win_start = max(0.0, event["start_s"] - args.context_seconds)
        win_end = event["end_s"] + args.context_seconds
        points = event_sample_points(chunks, win_start, win_end, args.sample_fps)
        print(f"\n[{idx}/{len(events)}] {event['raw_event']} "
              f"window={win_start:.1f}-{win_end:.1f}s samples={len(points)}")

        t0 = time.perf_counter()
        pooled, times_s = extract_pooled_for_event(
            sam2_model, clf_model, chunks, points, args, device,
        )
        pooled, norm_info = maybe_normalise_pooled(
            pooled, times_s, event, args.tt_norm,
        )
        probs = forward_probs(clf_model, pooled, device)
        probs_s = smooth_probs(probs, times_s, args.smooth_seconds)
        summary = summarise_event(event, times_s, probs, probs_s)
        summary["norm"] = norm_info
        summaries.append(summary)
        print(f"  done in {time.perf_counter() - t0:.1f}s; "
              f"max_inside={summary.get('max_inside_gt')} "
              f"max_outside={summary.get('max_outside_gt')}")

        csv_name = f"{idx:02d}_{safe_name(event['raw_event'])}.csv"
        write_event_csv(os.path.join(args.out_dir, csv_name),
                        event, times_s, probs, probs_s)
        summary["csv"] = csv_name
        plot_results.append({
            "event": event,
            "times_s": times_s,
            "probs_smooth": probs_s,
        })

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "video_folder": args.video_folder,
            "event_csv": args.event_csv,
            "context_seconds": args.context_seconds,
            "sample_fps": args.sample_fps,
            "smooth_seconds": args.smooth_seconds,
            "tt_norm": args.tt_norm,
            "summaries": summaries,
        }, f, indent=2)
    write_summary_text(os.path.join(args.out_dir, "summary.txt"), summaries)
    write_plot(os.path.join(args.out_dir, "event_probability_probe.png"),
               plot_results)
    print(f"\nFinished in {time.perf_counter() - t_all:.1f}s")
    print(f"Outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
