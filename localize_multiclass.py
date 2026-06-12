#!/usr/bin/env python3
"""
localize_multiclass.py
======================
LOCO-CV full-video localisation of all 5 surgical milestones using an
11-class softmax + 11-state Viterbi (states map 1:1 to emission classes).

For each held-out case:
  1. Train multinomial logreg on the other cases (with sample_weight from
     extract_multiclass_features.py — hard negatives weighted up).
  2. Run SAM2 backbone over the full held-out video at --sample_fps.
  3. Compute per-frame 11-d log-probabilities, apply rolling mean smoothing
     per class.
  4. Run Viterbi over the 11-state machine with 2 skip edges
     (catheter→posterior, apical→endobag).
  5. Derive (start_s, end_s) for each of the 5 milestone events from the
     decoded state sequence and report errors against the CSV annotations.

State machine (matches CLASS_NAMES below). Allowed transitions:
  self-loop on every state; advance by +1 from every state; plus the 2 skip
  edges 1→3, 7→9.

Only cases that have all 5 required events in the CSV are run.

Usage
-----
  python3 localize_multiclass.py --hold_out 219
  python3 localize_multiclass.py --all --fast
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
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

# ── 11 emission classes (the softmax output dimension) ───────────────────────
NUM_CLASSES = 11
CLASS_NAMES = [
    "pre_catheter_pull", "catheter_pull",
    "post_catheter_pull", "posterior_cut",
    "post_posterior_cut", "vas_cut",
    "post_vas_cut",      "apical_cut",
    "post_apical_cut",   "endobag",
    "post_endobag",
]

# ── 11 Viterbi states (1:1 mapping to emission classes) ──────────────────────
NUM_STATES = 11
STATE_TO_CLASS = list(range(NUM_STATES))
STATE_NAMES = CLASS_NAMES   # identical
assert len(STATE_TO_CLASS) == NUM_STATES == len(STATE_NAMES)

# Event STATES (used for per-event reporting + optional anchor bonus).
# Keys = state indices, values = the GT event name they should match.
EVENT_STATE_NAMES = {
    1: "catheter_pull",
    3: "posterior_cut",
    5: "vas_cut",
    7: "apical_cut",
    9: "endobag",
}
EVENT_STATES = list(EVENT_STATE_NAMES.keys())
EVENT_REPORT_ORDER = ["catheter_pull", "posterior_cut",
                      "vas_cut", "apical_cut", "endobag"]

# Skip edges: stay-or-advance graph plus these shortcuts.
SKIP_EDGES = [
    (1, 3),    # catheter_pull → posterior_cut (skip post_catheter_pull)
    (7, 9),    # apical_cut → endobag (skip post_apical_cut)
]

# Raw CSV event names that map to the shared vas_cut class.
VAS_CSV_NAMES = ("vas_cut_1", "vas_cut_2")


# ── CSV parsing ────────────────────────────────────────────────────────────────

REQUIRED_EVENTS = frozenset(EVENT_REPORT_ORDER)


def parse_all_events(path):
    """Returns dict: case_id → {event_name: (start_s, end_s)} for the 5
    milestone events. vas_cut_1 and vas_cut_2 are merged into a single
    'vas_cut' entry (union of their windows). Cases missing any required
    event are excluded."""
    accepted = set(EVENT_REPORT_ORDER) | set(VAS_CSV_NAMES)
    by_case = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["event"].strip()
            if name not in accepted:
                continue
            if name in VAS_CSV_NAMES:
                name = "vas_cut"
            cid = row["filename"].strip().replace("case_", "").replace("_clipped.mp4", "")
            s = float(row["start_sec"])
            e = float(row["end_sec"])
            if name in by_case.get(cid, {}):
                old_s, old_e = by_case[cid][name]
                by_case[cid][name] = (min(s, old_s), max(e, old_e))
            else:
                by_case.setdefault(cid, {})[name] = (s, e)
    # Drop cases missing any required event.
    complete = {}
    for cid, evs in by_case.items():
        if REQUIRED_EVENTS.issubset(evs):
            complete[cid] = evs
        else:
            missing = sorted(REQUIRED_EVENTS - set(evs))
            print(f"  [skip] case {cid}: missing {missing}")
    return complete


# ── Feature loading ────────────────────────────────────────────────────────────

def load_all_cases(features_dir):
    cases = {}
    for path in sorted(glob.glob(os.path.join(features_dir, "case_*.npz"))):
        d = np.load(path, allow_pickle=True)
        cid = str(d["case_id"])
        cases[cid] = {
            "features":       d["features"].astype(np.float32),
            "labels":         d["labels"].astype(np.int32),
            "sample_weights": (d["sample_weights"].astype(np.float32)
                               if "sample_weights" in d.files
                               else np.ones(len(d["labels"]), np.float32)),
        }
    return cases


# ── Classifier (training side) ─────────────────────────────────────────────────

def train_classifier(cases_dict, hold_out_id, C=1.0):
    from sklearn.linear_model import LogisticRegression
    train_ids = [c for c in cases_dict if c != hold_out_id]
    X_tr = np.concatenate([cases_dict[c]["features"]       for c in train_ids])
    y_tr = np.concatenate([cases_dict[c]["labels"]         for c in train_ids])
    w_tr = np.concatenate([cases_dict[c]["sample_weights"] for c in train_ids])

    m = X_tr.mean(axis=0); s = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - m) / s

    clf = LogisticRegression(
        C=C, max_iter=300, tol=1e-3, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )
    clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
    hn = int((w_tr > 1).sum())
    print(f"  Trained on {len(train_ids)} cases  ({len(y_tr)} frames, "
          f"{hn} hard-negs)")
    return clf, m, s


# ── GPU softmax classifier (for fast inference) ────────────────────────────────

class GPUSoftmax:
    """Multinomial logistic regression on GPU, padded to NUM_CLASSES.

    sklearn returns coef_ with one row per class actually seen in training; we
    pad to 13 rows and set intercept = -inf for any missing class so its
    log-softmax probability is 0."""
    def __init__(self, sklearn_clf, scaler_mean, scaler_std, device):
        K = NUM_CLASSES
        D = sklearn_clf.coef_.shape[1]
        W_full = np.zeros((K, D), dtype=np.float32)
        b_full = np.full(K, -1e9, dtype=np.float32)
        for i, c in enumerate(sklearn_clf.classes_):
            W_full[int(c)] = sklearn_clf.coef_[i].astype(np.float32)
            b_full[int(c)] = float(sklearn_clf.intercept_[i])

        self.W    = torch.from_numpy(W_full).to(device)
        self.b    = torch.from_numpy(b_full).to(device)
        self.mean = torch.from_numpy(scaler_mean.astype(np.float32)).to(device)
        self.std  = torch.from_numpy(scaler_std.astype(np.float32)).to(device)

    @torch.no_grad()
    def log_proba(self, feats):
        z = (feats - self.mean) / self.std
        logits = z @ self.W.T + self.b
        return torch.log_softmax(logits, dim=1)


# ── Video frame dataset ────────────────────────────────────────────────────────

class VideoFrameDataset(IterableDataset):
    """Same sequential grab/read pattern as the binary localizer — splits
    indices into per-worker contiguous chunks, one seek per chunk."""
    def __init__(self, video_path, indices, fps, sbs_eye):
        self.video_path = video_path
        self.indices    = sorted(int(i) for i in indices)
        self.fps        = float(fps)
        self.sbs_eye    = sbs_eye

    def _preprocess(self, frame_bgr):
        if self.sbs_eye == "left":
            frame_bgr = frame_bgr[:, :frame_bgr.shape[1] // 2]
        elif self.sbs_eye == "right":
            frame_bgr = frame_bgr[:, frame_bgr.shape[1] // 2:]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (t - IMG_MEAN) / IMG_STD

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            chunk = self.indices
        else:
            n = len(self.indices)
            sz = (n + info.num_workers - 1) // info.num_workers
            chunk = self.indices[info.id * sz : min((info.id + 1) * sz, n)]
        if not chunk:
            return
        cap = cv2.VideoCapture(self.video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk[0])
            current = chunk[0]
            for fi in chunk:
                while current < fi:
                    cap.grab()
                    current += 1
                ret, frame_bgr = cap.read()
                current += 1
                if not ret:
                    frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                yield self._preprocess(frame_bgr), fi / self.fps
        finally:
            cap.release()


# ── Feature extraction batch ───────────────────────────────────────────────────

@torch.no_grad()
def extract_batch(model, frames_t, device, use_bf16=False, return_gpu=False):
    imgs = frames_t.to(device, non_blocking=True)
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(imgs)
    else:
        backbone_out = model.forward_image(imgs)
    fpn = backbone_out["backbone_fpn"]
    pooled = []
    for f in fpn[1:]:
        pooled.append(f.mean(dim=[2, 3]))
        pooled.append(f.amax(dim=[2, 3]))
    feats = torch.cat(pooled, dim=1).float()
    if return_gpu:
        return feats
    return feats.cpu().numpy()


def load_sam2(variant, device):
    from sam2.build_sam import build_sam2
    ckpt = os.path.join(SAM2_DIR, VARIANT_TO_CKPT[variant])
    cfg  = VARIANT_TO_CFG[variant]
    model = build_sam2(cfg, ckpt, device=device)
    model.eval()
    print(f"  SAM2-{variant} loaded  ({ckpt})")
    return model


def pad_proba(proba, classes_):
    if len(classes_) == NUM_CLASSES and np.array_equal(classes_, np.arange(NUM_CLASSES)):
        return proba
    full = np.zeros((len(proba), NUM_CLASSES), dtype=proba.dtype)
    for i, c in enumerate(classes_):
        full[:, c] = proba[:, i]
    return full


# ── Full-video inference ───────────────────────────────────────────────────────

def infer_full_video(video_path, model, clf, mean, std,
                     sample_fps, batch_size, sbs_eye, device,
                     use_bf16=False, num_workers=0, gpu_classifier=False):
    """Returns (times_s [T], log_probs [T, 11])."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / sample_fps)))
    cap.release()

    gpu_clf = GPUSoftmax(clf, mean, std, device) if gpu_classifier else None

    if num_workers > 0:
        indices = list(range(0, total, stride))
        ds = VideoFrameDataset(video_path, indices, fps, sbs_eye)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, prefetch_factor=2)
        times_s   = []
        logp_chunks = []
        pbar = tqdm(total=len(indices),
                    desc=f"  inference (DL nw={num_workers})",
                    unit="frame", ncols=80)
        for batch_t, batch_times in loader:
            if gpu_clf is not None:
                feats = extract_batch(model, batch_t, device,
                                       use_bf16=use_bf16, return_gpu=True)
                logp_chunks.append(gpu_clf.log_proba(feats))
            else:
                feats = extract_batch(model, batch_t, device, use_bf16=use_bf16)
                feats_s = (feats - mean) / std
                proba = pad_proba(clf.predict_proba(feats_s), clf.classes_)
                logp_chunks.append(np.log(np.clip(proba, 1e-12, 1.0)))
            times_s.extend(batch_times.tolist())
            pbar.update(batch_t.size(0))
        pbar.close()
        if gpu_clf is not None:
            log_probs = torch.cat(logp_chunks).cpu().numpy()
        else:
            log_probs = np.concatenate(logp_chunks, axis=0)
        times_arr = np.array(times_s)
        order = np.argsort(times_arr)
        return times_arr[order], log_probs[order]

    # Inline loop — baseline timing reference.
    n_samples = total // stride
    cap = cv2.VideoCapture(video_path)
    frame_tensors = []
    times_s       = []
    logp_chunks   = []
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
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            t = (t - IMG_MEAN) / IMG_STD
            frame_tensors.append(t)
            times_s.append(t_s)
            pbar.update(1)
            if len(frame_tensors) == batch_size:
                if gpu_clf is not None:
                    feats = extract_batch(model, torch.stack(frame_tensors), device,
                                           use_bf16=use_bf16, return_gpu=True)
                    logp_chunks.append(gpu_clf.log_proba(feats))
                else:
                    feats = extract_batch(model, torch.stack(frame_tensors), device,
                                           use_bf16=use_bf16)
                    feats_s = (feats - mean) / std
                    proba = pad_proba(clf.predict_proba(feats_s), clf.classes_)
                    logp_chunks.append(np.log(np.clip(proba, 1e-12, 1.0)))
                frame_tensors = []
        else:
            if not cap.grab():
                break
        frame_num += 1
    if frame_tensors:
        if gpu_clf is not None:
            feats = extract_batch(model, torch.stack(frame_tensors), device,
                                   use_bf16=use_bf16, return_gpu=True)
            logp_chunks.append(gpu_clf.log_proba(feats))
        else:
            feats = extract_batch(model, torch.stack(frame_tensors), device,
                                   use_bf16=use_bf16)
            feats_s = (feats - mean) / std
            proba = pad_proba(clf.predict_proba(feats_s), clf.classes_)
            logp_chunks.append(np.log(np.clip(proba, 1e-12, 1.0)))
    pbar.close()
    cap.release()
    if gpu_clf is not None:
        log_probs = torch.cat(logp_chunks).cpu().numpy()
    else:
        log_probs = np.concatenate(logp_chunks, axis=0)
    return np.array(times_s), log_probs


# ── Per-class smoothing ────────────────────────────────────────────────────────

def smooth_log_probs(log_probs, window_frames):
    """Rolling mean per class. log_probs is [T, K]. Returns same shape."""
    if window_frames <= 1:
        return log_probs
    pad = window_frames // 2
    kernel = np.ones(window_frames) / window_frames
    T, K = log_probs.shape
    smoothed = np.empty_like(log_probs)
    for k in range(K):
        x_pad = np.pad(log_probs[:, k], pad, mode="edge")
        smoothed[:, k] = np.convolve(x_pad, kernel, mode="valid")[:T]
    return smoothed


# ── Viterbi ────────────────────────────────────────────────────────────────────

def viterbi_decode(log_emit, alpha=0.0, skip_cost=0.0):
    """
    log_emit:   (T, NUM_CLASSES=11) log emission probabilities (already smoothed)
    alpha:      log-prob bonus added to event-state emissions (anchors decoder
                to high-confidence event firings)
    skip_cost:  log-prob added when taking a skip edge (default 0 — purely
                emission-driven; negative values discourage skips)

    Returns:
      state_seq (T,) int32 — path through the NUM_STATES=11 state machine,
      monotone non-decreasing.
    """
    T, K_cls = log_emit.shape
    assert K_cls == NUM_CLASSES

    # Map class log-probs to state log-probs via STATE_TO_CLASS (1:1 here).
    state_emit = log_emit[:, STATE_TO_CLASS].astype(np.float64).copy()  # (T, NUM_STATES)
    if alpha != 0.0:
        for s in EVENT_STATES:
            state_emit[:, s] += alpha

    # Build log-transition matrix over states: -inf everywhere except allowed edges.
    log_trans = np.full((NUM_STATES, NUM_STATES), -np.inf, dtype=np.float64)
    for i in range(NUM_STATES):
        log_trans[i, i] = 0.0                       # stay
        if i + 1 < NUM_STATES:
            log_trans[i, i + 1] = 0.0               # advance by 1
    for fr, to in SKIP_EDGES:
        log_trans[fr, to] = skip_cost               # skip-edge (still allowed, with cost)

    dp = np.full((T, NUM_STATES), -np.inf, dtype=np.float64)
    bp = np.zeros((T, NUM_STATES), dtype=np.int32)

    # Force start at state 0 (every case begins in pre_catheter_pull).
    dp[0, 0] = state_emit[0, 0]

    for t in range(1, T):
        scores    = dp[t - 1, :, None] + log_trans     # (NUM_STATES, NUM_STATES)
        best_prev = np.argmax(scores, axis=0)
        dp[t]     = scores[best_prev, np.arange(NUM_STATES)] + state_emit[t]
        bp[t]     = best_prev

    # No end-state constraint — allow ending anywhere.
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def state_seq_to_event_windows(state_seq, times_s):
    """Returns dict[event_name] → (start_s, end_s) or None."""
    out = {}
    for state_idx, name in EVENT_STATE_NAMES.items():
        mask = (state_seq == state_idx)
        if not mask.any():
            out[name] = None
        else:
            idxs = np.where(mask)[0]
            out[name] = (float(times_s[idxs[0]]), float(times_s[idxs[-1]]))
    return out


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_plot(times_s, log_probs_smooth, state_seq, gt_windows, pred_windows,
              case_id, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    probs = np.exp(log_probs_smooth)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                    gridspec_kw=dict(height_ratios=[3, 1]))

    cmap = plt.get_cmap("tab10")
    plot_event_classes = [(1, "catheter_pull"), (3, "posterior_cut"),
                          (5, "vas_cut"),       (7, "apical_cut"),
                          (9, "endobag")]
    for i, (cls, label) in enumerate(plot_event_classes):
        ax1.plot(times_s, probs[:, cls], color=cmap(i), linewidth=1.2,
                 label=label)

    name_to_color = {
        "catheter_pull": cmap(0), "posterior_cut": cmap(1),
        "vas_cut":       cmap(2),
        "apical_cut":    cmap(3), "endobag":       cmap(4),
    }
    for name, col in name_to_color.items():
        gt = gt_windows.get(name)
        if gt is not None:
            ax1.axvspan(gt[0], gt[1], color=col, alpha=0.20)
        pr = pred_windows.get(name)
        if pr is not None:
            ax1.axvspan(pr[0], pr[1], color=col, alpha=0.45,
                        ymin=0.92, ymax=1.0)

    ax1.set_ylabel("P(event)")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Multi-class phase localisation — case {case_id}")
    ax1.legend(fontsize=8, loc="upper left", ncol=3)
    ax2_top = ax1.secondary_xaxis("top",
                                   functions=(lambda x: x/60, lambda x: x*60))
    ax2_top.set_xlabel("Time (min)")

    # Bottom: Viterbi state assignment over the 11-state machine.
    ax2.plot(times_s, state_seq, drawstyle="steps-post", color="black",
             linewidth=1.0)
    ax2.set_yticks(range(NUM_STATES))
    ax2.set_yticklabels(STATE_NAMES, fontsize=7)
    ax2.set_ylim(-0.5, NUM_STATES - 0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Viterbi state")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlim(times_s[0], times_s[-1])

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  Plot → {out_png}")


# ── Per-case run ───────────────────────────────────────────────────────────────

def run_case(hold_out_id, cases_dict, all_gt, sam2_model, args, device):
    video_path = os.path.join(VIDEO_DIR, f"case_{hold_out_id}_clipped.mp4")
    if not os.path.exists(video_path):
        print(f"  SKIP: video not found → {video_path}")
        return None
    if hold_out_id not in all_gt:
        print(f"  SKIP: no annotations for case {hold_out_id}")
        return None

    gt = all_gt[hold_out_id]
    print(f"\n{'='*70}")
    print(f"  Hold-out: case {hold_out_id}")
    for name in EVENT_REPORT_ORDER:
        if name in gt:
            s, e = gt[name]
            print(f"    GT {name:<14}: {s:.0f}s-{e:.0f}s  ({s/60:.1f}-{e/60:.1f} min)")
        else:
            print(f"    GT {name:<14}: MISSING")
    print(f"{'='*70}")

    clf, mean, std = train_classifier(cases_dict, hold_out_id, C=args.C)

    print(f"\n  Running full-video inference at {args.sample_fps} fps "
          f"(variant={args.sam2_variant}, bf16={args.use_bf16}, "
          f"compile={args.use_compile}, nw={args.num_workers}, "
          f"gpu_clf={args.gpu_classifier}, batch={args.batch_size}) ...")
    t0 = time.perf_counter()
    times_s, log_probs = infer_full_video(
        video_path, sam2_model, clf, mean, std,
        args.sample_fps, args.batch_size, args.sbs_eye, device,
        use_bf16=args.use_bf16, num_workers=args.num_workers,
        gpu_classifier=args.gpu_classifier,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  Inference: {dt:.1f}s  ({len(times_s)} frames, "
          f"{len(times_s) / max(dt, 1e-6):.2f} fps sustained)")

    window_frames = max(1, int(args.smooth_window * args.sample_fps))
    log_probs_sm  = smooth_log_probs(log_probs, window_frames)

    state_seq = viterbi_decode(log_probs_sm,
                               alpha=args.alpha, skip_cost=args.skip_cost)
    pred_windows = state_seq_to_event_windows(state_seq, times_s)

    # Per-event error reporting
    print(f"\n  Predictions:")
    per_event = {}
    for name in EVENT_REPORT_ORDER:
        gt_w = gt.get(name)
        pr_w = pred_windows.get(name)
        if pr_w is None:
            print(f"    {name:<14}: SKIPPED by decoder")
            per_event[name] = {"gt": gt_w, "pred": None,
                               "err_s": None, "err_e": None}
            continue
        ps, pe = pr_w
        if gt_w is not None:
            es = ps - gt_w[0]
            ee = pe - gt_w[1]
            print(f"    {name:<14}: pred {ps:.0f}s-{pe:.0f}s  "
                  f"(start err={es:+.0f}s  end err={ee:+.0f}s)")
            per_event[name] = {"gt": gt_w, "pred": pr_w,
                               "err_s": es, "err_e": ee}
        else:
            print(f"    {name:<14}: pred {ps:.0f}s-{pe:.0f}s  (no GT)")
            per_event[name] = {"gt": None, "pred": pr_w,
                               "err_s": None, "err_e": None}

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"case_{hold_out_id}_multiclass.png")
    save_plot(times_s, log_probs_sm, state_seq, gt, pred_windows,
              hold_out_id, out_png)

    return {"case_id": hold_out_id, "per_event": per_event}


# ── Summary across all held-out cases ──────────────────────────────────────────

WITHIN_3MIN = 180.0

def print_summary(results, out_dir):
    lines = []
    lines.append(f"{'='*78}")
    lines.append(f"  MULTI-CLASS SUMMARY  ({len(results)} held-out cases)")
    lines.append(f"{'='*78}")
    lines.append(f"\n  Per-event start-time errors (seconds):")
    lines.append(f"  {'Event':<16} {'median':>9} {'mean':>9} {'max':>9}  "
                 f"{'<3min':>8}  {'skipped':>8}")
    lines.append(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*9}  {'-'*8}  {'-'*8}")
    for name in EVENT_REPORT_ORDER:
        errs    = [r["per_event"][name]["err_s"] for r in results
                   if r["per_event"][name]["err_s"] is not None]
        skipped = sum(1 for r in results if r["per_event"][name]["pred"] is None)
        if not errs:
            lines.append(f"  {name:<16}  no predictions ({skipped} skipped)")
            continue
        abs_errs = [abs(e) for e in errs]
        median   = sorted(abs_errs)[len(abs_errs) // 2]
        mean     = sum(abs_errs) / len(abs_errs)
        mx       = max(abs_errs)
        within   = sum(1 for e in abs_errs if e <= WITHIN_3MIN)
        lines.append(f"  {name:<16} {median:>9.0f} {mean:>9.0f} {mx:>9.0f}  "
                     f"{within:>4}/{len(errs):<3} {skipped:>8}")
    lines.append(f"\n  Per-case breakdown:")
    short = {"catheter_pull": "cath", "posterior_cut": "post",
             "vas_cut":       "vas",
             "apical_cut":    "apic", "endobag":       "endo"}
    for r in results:
        parts = []
        for name in EVENT_REPORT_ORDER:
            e = r["per_event"][name]["err_s"]
            if e is None:
                parts.append(f"{short[name]}=skip")
            else:
                parts.append(f"{short[name]}={e:+.0f}s")
        lines.append(f"    case {r['case_id']:<6}  " + "  ".join(parts))
    lines.append(f"{'='*78}")
    text = "\n".join(lines)
    print("\n" + text + "\n")
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  Summary → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--hold_out", help="Case ID to hold out (e.g. 219)")
    group.add_argument("--all",      action="store_true",
                       help="Run LOCO over every case with features.")
    p.add_argument("--annot_csv",    default=ANNOT_CSV)
    p.add_argument("--features_dir", default=os.path.join(SAM3_DIR, "multiclass_features"))
    p.add_argument("--out_dir",      default=os.path.join(SAM3_DIR, "multiclass_localization"))
    p.add_argument("--sample_fps",   type=float, default=0.5)
    p.add_argument("--smooth_window",type=float, default=20.0,
                   help="Per-class log-prob rolling-mean window in seconds.")
    p.add_argument("--alpha",        type=float, default=0.0,
                   help="Event-anchor log-prob bonus (added to event-class emissions in Viterbi).")
    p.add_argument("--skip_cost",    type=float, default=0.0,
                   help="Log-prob added to skip edges (negative discourages skipping; default 0 = data-driven).")
    p.add_argument("--C",            type=float, default=1.0)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--sbs_eye",      default="none", choices=["left", "right", "none"])
    p.add_argument("--sam2_variant", default="small", choices=list(VARIANT_TO_CKPT.keys()))
    p.add_argument("--use_bf16",     action="store_true")
    p.add_argument("--use_compile",  action="store_true")
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--gpu_classifier", action="store_true")
    p.add_argument("--fast",         action="store_true",
                   help="Shortcut: enables --use_bf16, --use_compile, --gpu_classifier, num_workers=4.")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.fast:
        args.use_bf16       = True
        args.use_compile    = True
        args.gpu_classifier = True
        if args.num_workers == 0:
            args.num_workers = 4

    print(f"\nLoading features from {args.features_dir}/")
    cases_dict = load_all_cases(args.features_dir)
    if not cases_dict:
        print("No features found. Run extract_multiclass_features.py first.")
        return
    all_gt = parse_all_events(args.annot_csv)

    print(f"Loading SAM2 backbone (variant={args.sam2_variant})...")
    sam2_model = load_sam2(args.sam2_variant, device)

    if args.use_compile:
        print("  Wrapping forward_image with torch.compile(mode='default')...")
        sam2_model.forward_image = torch.compile(sam2_model.forward_image, mode="default")
        print("  Warmup (~30s)...")
        warmup = torch.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
        for _ in range(5):
            with torch.no_grad():
                if args.use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = sam2_model.forward_image(warmup)
                else:
                    _ = sam2_model.forward_image(warmup)
        torch.cuda.synchronize()
        del warmup
        torch.cuda.empty_cache()
        print("  Compile ready.")

    os.makedirs(args.out_dir, exist_ok=True)
    hold_outs = list(cases_dict.keys()) if args.all else [args.hold_out]

    results = []
    for cid in hold_outs:
        if cid not in cases_dict:
            print(f"  SKIP: no features for case {cid}")
            continue
        r = run_case(cid, cases_dict, all_gt, sam2_model, args, device)
        if r is not None:
            results.append(r)

    if len(results) >= 1:
        print_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
