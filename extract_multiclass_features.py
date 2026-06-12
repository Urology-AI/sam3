#!/usr/bin/env python3
"""
extract_multiclass_features.py
==============================
Extract SAM2 ViT backbone features for the 11-class surgical phase classifier.

Class layout (11 emission classes, used by the softmax)
-------------------------------------------------------
  0  pre_catheter_pull        (bladder drop / mobilisation)
  1  catheter_pull
  2  post_catheter_pull
  3  posterior_cut
  4  post_posterior_cut
  5  vas_cut                  (shared class for vas_cut_1 AND vas_cut_2 — left
                               vs right is not visually distinguishable from
                               the features, so the softmax learns a single
                               "vas cutting" pattern)
  6  post_vas_cut             (seminal + posterior plane + urethral prep —
                               also the visual signature of the gap BETWEEN
                               the two vas cuts)
  7  apical_cut
  8  post_apical_cut
  9  endobag
 10  post_endobag             (closure)

`seminal_peeling` is intentionally ignored — only a minority of cases have it.

vas_cut_1 and vas_cut_2 CSV rows are both labelled with class 5 (vas_cut) and
kept as separate event windows. The gap between them (if present) gets the
post_vas_cut interphase label (class 6). The 11-state Viterbi in
localize_multiclass.py produces a single vas_cut firing per case.

Only cases that have ALL 5 required events (catheter_pull, posterior_cut,
vas_cut (≥1), apical_cut, endobag) are processed — incomplete cases are
skipped with a warning.

Sample plan, per case
---------------------
For every sampled frame at time t:
  1. If t lies inside an event window  → labelled with that event's class
  2. Else if t lies inside ANY event's safety gap (3s before start, 3s after end)
                                         → DROPPED (annotation boundary ambiguity)
  3. Else                                → labelled with its chronological interphase
     and given sample_weight = hard_neg_weight if it falls in the post-event
     hard-negative window for some event, otherwise 1.0.

Hard-negative window for event ending at e (next event starting at s_next):
    [e + safety_gap,  min(e + safety_gap + hard_neg_duration, s_next - safety_gap)]
The clip against `s_next - safety_gap` is essential — it stops the hard-neg
window from bleeding into the next event (short interphases: apical→endobag
can be 5s in some cases, catheter→posterior ~45s).

Usage
-----
  python3 extract_multiclass_features.py
  python3 extract_multiclass_features.py --sam2_variant small --sample_fps 1.0
  python3 extract_multiclass_features.py --safety_gap 3.0 --hard_neg_duration 10.0
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
ANNOT_CSV = os.path.join(VIDEO_DIR, "annotate_fine.csv")

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

IMG_SIZE = 1024
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# Canonical event order (matches the 11-state machine in localize_multiclass.py)
EVENT_SEQUENCE = [
    "catheter_pull",
    "posterior_cut",
    "vas_cut",          # merged from vas_cut_1 + vas_cut_2 in the CSV
    "apical_cut",
    "endobag",
]
# Cases missing any of these (at least 1 vas_cut) are skipped entirely.
REQUIRED_EVENTS = frozenset(EVENT_SEQUENCE)
# Even indices are interphase states, odd indices are event states.
#   pre_catheter=0, catheter=1, post_catheter=2, posterior=3, post_posterior=4, …
EVENT_TO_CLASS = {name: 2 * i + 1 for i, name in enumerate(EVENT_SEQUENCE)}
NUM_CLASSES    = 2 * len(EVENT_SEQUENCE) + 1  # 11
CLASS_NAMES = [
    "pre_catheter_pull", "catheter_pull",
    "post_catheter_pull", "posterior_cut",
    "post_posterior_cut", "vas_cut",
    "post_vas_cut",      "apical_cut",
    "post_apical_cut",   "endobag",
    "post_endobag",
]
assert len(CLASS_NAMES) == NUM_CLASSES

# Raw CSV event names that get merged into the single `vas_cut` class.
VAS_CSV_NAMES = ("vas_cut_1", "vas_cut_2")


# ── CSV parsing ────────────────────────────────────────────────────────────────

def parse_all_events(path):
    """
    Returns dict: case_id (str) → sorted list of (event_name, start_s, end_s).
    Drops rows whose event is not in EVENT_SEQUENCE or in VAS_CSV_NAMES
    (seminal_peeling is dropped entirely).

    vas_cut_1 and vas_cut_2 CSV rows are each renamed to "vas_cut" but kept
    as SEPARATE entries (not merged) so build_sample_plan emits two distinct
    vas event windows with class 5 and the gap between them gets the
    post_vas_cut interphase label (class 6).
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["event"].strip()
            # Map both vas CSV rows to the shared "vas_cut" class name,
            # but DO NOT merge their windows.
            if name in VAS_CSV_NAMES:
                name = "vas_cut"
            if name not in EVENT_TO_CLASS:
                continue
            fname   = row["filename"].strip()
            case_id = fname.replace("case_", "").replace("_clipped.mp4", "")
            s = float(row["start_sec"])
            e = float(row["end_sec"])
            rows.append((case_id, name, s, e))

    by_case = {}
    for cid, name, s, e in rows:
        by_case.setdefault(cid, []).append((name, s, e))
    for cid in by_case:
        by_case[cid].sort(key=lambda x: x[1])
    return by_case


# ── Sample plan ────────────────────────────────────────────────────────────────

def build_sample_plan(total_frames, fps, sorted_events, video_end_s,
                      sample_fps, safety_gap, hard_neg_duration,
                      hard_neg_weight, max_interphase_per_case=None):
    """
    Returns:
      frame_indices: (N,) int64 video frame indices
      labels:        (N,) int32 class index in [0, NUM_CLASSES)
      weights:       (N,) float32 per-sample weight (1.0 normally, hard_neg_weight in
                     post-event hard-neg windows)
    """
    stride = max(1, int(round(fps / sample_fps)))
    all_indices = np.arange(0, total_frames, stride, dtype=np.int64)

    # Event intervals (class_idx, s, e), already sorted by s.
    event_intervals = [(EVENT_TO_CLASS[name], s, e) for name, s, e in sorted_events]

    # Interphase intervals (class_idx, s, e). pre = [0, event0.s), post_i = (e_i, s_{i+1}),
    # post_endobag = (e_last, video_end].
    interphase = []
    if event_intervals:
        interphase.append((0, 0.0, event_intervals[0][1]))
        for i in range(len(event_intervals) - 1):
            cls_i, _, e_i = event_intervals[i]
            _,     s_n, _ = event_intervals[i + 1]
            interphase.append((cls_i + 1, e_i, s_n))
        cls_last, _, e_last = event_intervals[-1]
        interphase.append((cls_last + 1, e_last, video_end_s + 1e-6))

    # Hard-neg zones, one per event end, clipped against next event's pre-safety gap.
    hard_neg_zones = []
    for i, (_, _, e_i) in enumerate(event_intervals):
        if i + 1 < len(event_intervals):
            s_n = event_intervals[i + 1][1]
            clip_end = s_n - safety_gap
        else:
            clip_end = video_end_s
        zone_s = e_i + safety_gap
        zone_e = min(e_i + safety_gap + hard_neg_duration, clip_end)
        if zone_e > zone_s:
            hard_neg_zones.append((zone_s, zone_e))

    def label_and_weight(t):
        # Events win unconditionally.
        for cls, s, e in event_intervals:
            if s <= t <= e:
                return cls, 1.0
        # Interphase frame — first check whether it's in any safety gap.
        for _, s, e in event_intervals:
            if (s - safety_gap) < t < s or e < t < (e + safety_gap):
                return None, None
        # Hard-neg check.
        w = 1.0
        for zs, ze in hard_neg_zones:
            if zs <= t < ze:
                w = hard_neg_weight
                break
        # Assign interphase class.
        for cls, s, e in interphase:
            if s <= t < e:
                return cls, w
        return None, None

    frame_idx_kept = []
    labels_kept    = []
    weights_kept   = []
    for fi in all_indices:
        t = fi / fps
        lbl, w = label_and_weight(t)
        if lbl is None:
            continue
        frame_idx_kept.append(int(fi))
        labels_kept.append(int(lbl))
        weights_kept.append(float(w))

    frame_indices = np.array(frame_idx_kept, dtype=np.int64)
    labels        = np.array(labels_kept,    dtype=np.int32)
    weights       = np.array(weights_kept,   dtype=np.float32)

    # Optional per-class subsampling — interphase only, hard-negs preserved.
    if max_interphase_per_case is not None and max_interphase_per_case > 0:
        event_cls = set(EVENT_TO_CLASS.values())
        keep = np.ones(len(labels), dtype=bool)
        for cls in np.unique(labels):
            if int(cls) in event_cls:
                continue
            cls_mask = (labels == cls) & (weights == 1.0)  # don't subsample hard-negs
            cls_pos  = np.where(cls_mask)[0]
            if len(cls_pos) > max_interphase_per_case:
                sub = np.linspace(0, len(cls_pos) - 1,
                                  max_interphase_per_case, dtype=int)
                drop = np.setdiff1d(cls_pos, cls_pos[sub])
                keep[drop] = False
        frame_indices = frame_indices[keep]
        labels        = labels[keep]
        weights       = weights[keep]

    return frame_indices, labels, weights


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
    frame_rgb = cv2.resize(frame_rgb, (img_size, img_size),
                           interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


@torch.no_grad()
def extract_batch(model, frames_t, device, use_bf16=False):
    imgs = frames_t.to(device, non_blocking=True)
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(imgs)
    else:
        backbone_out = model.forward_image(imgs)
    fpn = backbone_out["backbone_fpn"]
    pooled = []
    for f in fpn[1:]:                # drop fpn[0] (finest scale, local texture)
        pooled.append(f.mean(dim=[2, 3]))
        pooled.append(f.amax(dim=[2, 3]))
    feat = torch.cat(pooled, dim=1)
    return feat.cpu().float().numpy()


# ── Per-case pipeline ──────────────────────────────────────────────────────────

def process_case(case_id, sorted_events, model, args, device):
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

    # Class summary
    print(f"  case {case_id}: {total_frames} frames @ {fps:.2f} fps  "
          f"({video_end_s/60:.1f} min)")
    print(f"    kept: {len(frame_indices)}  "
          f"hard_neg: {int((weights > 1.0).sum())}")
    for cls in range(NUM_CLASSES):
        n = int((labels == cls).sum())
        n_hn = int(((labels == cls) & (weights > 1.0)).sum())
        if n > 0:
            print(f"      cls {cls:>2} {CLASS_NAMES[cls]:<22}  n={n:<5}  hard_neg={n_hn}")

    if len(frame_indices) == 0:
        print(f"  SKIP: empty sample plan")
        return

    # Sequential grab+read: one seek to the first sample, then grab() (no decode)
    # past intermediate frames and read() only at sample indices. Roughly 5–10×
    # faster than cap.set(POS_FRAMES) per frame on h264/h265 inputs.
    cap = cv2.VideoCapture(video_path)
    frame_tensors = []
    feat_chunks   = []
    pbar = tqdm(total=len(frame_indices), desc=f"  case {case_id}",
                unit="frame", ncols=80)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    current = int(frame_indices[0])
    for fi in frame_indices:
        fi = int(fi)
        while current < fi:
            cap.grab()
            current += 1
        ret, frame_bgr = cap.read()
        current += 1
        pbar.update(1)
        if not ret:
            frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        frame_tensors.append(preprocess_frame(frame_bgr, IMG_SIZE, args.sbs_eye))
        if len(frame_tensors) == args.batch_size:
            feat_chunks.append(
                extract_batch(model, torch.stack(frame_tensors), device,
                              use_bf16=args.use_bf16)
            )
            frame_tensors = []
    pbar.close()
    cap.release()
    if frame_tensors:
        feat_chunks.append(
            extract_batch(model, torch.stack(frame_tensors), device,
                          use_bf16=args.use_bf16)
        )

    features = np.concatenate(feat_chunks, axis=0)
    assert features.shape[0] == len(frame_indices), \
        f"feature count mismatch: {features.shape[0]} vs {len(frame_indices)}"

    out_path = os.path.join(args.out_dir, f"case_{case_id}.npz")
    np.savez_compressed(
        out_path,
        features       = features,
        labels         = labels,
        sample_weights = weights,
        frame_indices  = frame_indices,
        fps            = np.float32(fps),
        case_id        = case_id,
    )
    print(f"  Saved → {out_path}  feat_dim={features.shape[1]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annot_csv",          default=ANNOT_CSV)
    p.add_argument("--out_dir",            default=os.path.join(SAM3_DIR, "multiclass_features"))
    p.add_argument("--sample_fps",         type=float, default=1.0)
    p.add_argument("--safety_gap",         type=float, default=3.0,
                   help="Seconds dropped from training on each side of every event boundary "
                        "(annotation-boundary ambiguity zone).")
    p.add_argument("--hard_neg_duration",  type=float, default=10.0,
                   help="Post-event hard-negative window length, in seconds. Clipped against "
                        "the next event's pre-event safety gap so short interphases never bleed "
                        "into the following event.")
    p.add_argument("--hard_neg_weight",    type=float, default=5.0,
                   help="Sample-weight multiplier applied to hard-negative frames during training.")
    p.add_argument("--max_interphase_per_case", type=int, default=None,
                   help="Optional per-case cap on each interphase class (hard-negs preserved). "
                        "Default None = no cap.")
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--sbs_eye",            default="none", choices=["left", "right", "none"])
    p.add_argument("--sam2_variant",       default="small", choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--cases",              default=None,
                   help="Comma-separated case IDs to process (default: all cases in CSV).")
    p.add_argument("--use_bf16",           action="store_true",
                   help="Wrap forward_image in torch.autocast(bf16) — faster on H100.")
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
    print(f"Sample rate: {args.sample_fps} fps  |  safety_gap: {args.safety_gap}s  "
          f"|  hard_neg: {args.hard_neg_duration}s × {args.hard_neg_weight}  "
          f"|  variant: {args.sam2_variant}  |  out: {args.out_dir}")

    by_case = parse_all_events(args.annot_csv)
    if not by_case:
        print(f"No usable annotations in {args.annot_csv}.")
        return
    if args.cases:
        keep = set(args.cases.split(","))
        by_case = {k: v for k, v in by_case.items() if k in keep}

    # Filter: only process cases that have all 5 required events (at least 1 vas_cut).
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
        print("\n[skip] cases missing required events (excluded from extraction):")
        for cid, miss in skipped:
            print(f"        case {cid}: missing {miss}")
    by_case = complete

    print(f"\nCases to process: {sorted(by_case.keys())}\n")

    model = load_sam2(args.sam2_variant, device)

    for case_id, sorted_events in tqdm(sorted(by_case.items()),
                                       desc="Cases", unit="case", ncols=80):
        print(f"\n[case {case_id}]  events: {[(n, s, e) for n, s, e in sorted_events]}")
        process_case(case_id, sorted_events, model, args, device)

    print(f"\nDone. Features → {args.out_dir}/")


if __name__ == "__main__":
    main()
