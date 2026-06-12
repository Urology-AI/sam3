#!/usr/bin/env python3
"""
train_multiclass_v2.py
=======================
Train AttnPoolBiGRU on v2 features (output of extract_multiclass_features_v2.py
in train mode).

Pipeline
--------
1. Glob multiclass_features_v2/train/case_*.npz.
2. Random 80/20 split of the case list → train cases / val cases (seed-fixed).
3. Build windowed dataset:
   - Train: random 64-frame windows, random augmented variant per frame.
     Each case contributes WINDOWS_PER_CASE windows per epoch.
   - Val:   deterministic non-overlapping windows starting at index 0,
     always variant 0 (clean) for stable per-epoch model-selection signal.
4. Cross-entropy loss with two multipliers per frame:
   - Class-balance weight w_c = N / (K · count_c)  (sklearn "balanced" formula,
     computed once from the training-set labels).
   - Per-frame sample_weight from the .npz (1.0 normal, 5.0 hard-neg,
     0 frames are masked out of the loss).
5. AdamW + cosine LR schedule (no warmup), gradient clip 1.0, 30 epochs default.
6. Save best checkpoint by val accuracy on weight>0 frames; also save final.
7. Per-epoch log to log.txt + structured per_epoch_val.json.

Output (default --out_dir = multiclass_checkpoints_v2/<run_id>):
  checkpoint_best.pt   — best model by val accuracy
  checkpoint_final.pt  — last epoch
  log.txt              — human-readable per-epoch log
  per_epoch_val.json   — per-epoch train/val loss + accuracy + LR

Notes on what is NOT in this script (intentionally)
---------------------------------------------------
- Full 15-fold LOCO. Per the v2 plan, the headline metric is OOD validation
  on SUBJ_1b7d93c2 (handled by validate_multiclass_v2.py). The 80/20 split
  here is the per-epoch model-selection signal, not a generalisation
  estimate. If you want LOCO numbers, run this script in a loop with one
  case held out at a time (orchestrator job, not this script's concern).
- Test-time burn-in normalisation. That belongs to validation, not training.

Usage
-----
  python3 train_multiclass_v2.py
  python3 train_multiclass_v2.py --epochs 50 --batch_size 4 --lr 5e-4
"""

import argparse
import datetime
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_DIR)
from multiclass_model import AttnPoolBiGRU, NUM_CLASSES

FEATURES_DIR_DEFAULT = os.path.join(SAM3_DIR, "multiclass_features_v2", "train")
CHECKPOINT_ROOT      = os.path.join(SAM3_DIR, "multiclass_checkpoints_v2")
CLASS_NAMES = [
    "pre_catheter_pull", "catheter_pull",
    "post_catheter_pull", "posterior_cut",
    "post_posterior_cut", "vas_cut",
    "post_vas_cut",      "apical_cut",
    "post_apical_cut",   "endobag",
    "post_endobag",
]


# ── Per-case feature loader ──────────────────────────────────────────────────

class CaseFeatures:
    """Loads one case's .npz into RAM. Total memory across 15 cases at K=4
    is ~13 GB — manageable for single-process training on H100 nodes."""
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.case_id  = str(d["case_id"])
        self.fpn1     = np.asarray(d["features_fpn1"])     # (N, K, C1, H1, W1) fp16
        self.fpn2     = np.asarray(d["features_fpn2"])     # (N, K, C2, H2, W2) fp16
        self.labels   = np.asarray(d["labels"]).astype(np.int64)
        self.weights  = np.asarray(d["sample_weights"]).astype(np.float32)
        self.N        = int(self.labels.shape[0])
        self.K        = int(d["K"])
        self.C1       = int(self.fpn1.shape[2])
        self.C2       = int(self.fpn2.shape[2])
        self.path     = path


# ── Windowed dataset ─────────────────────────────────────────────────────────

class WindowedCaseDataset(Dataset):
    """Returns 64-frame windows.

    Train mode: random start offset, random augmented-variant index per frame.
                Each case contributes `windows_per_case` windows per epoch.
    Val   mode: deterministic non-overlapping windows starting at index 0,
                always variant 0 (clean) for stable model-selection signal.

    Output dict per item:
      fpn1    : float32 (T, C1, H1, W1)
      fpn2    : float32 (T, C2, H2, W2)
      labels  : int64   (T,)
      weights : float32 (T,)   — 0 marks padding at the end of short val windows
    """
    def __init__(self, cases, window_size=64, windows_per_case=None,
                 mode="train", event_windows_per_case=0):
        self.cases       = cases
        self.window_size = window_size
        self.mode        = mode
        if mode == "train":
            assert windows_per_case is not None
            self.windows_per_case = windows_per_case
            self.event_windows_per_case = event_windows_per_case
            self.event_positions = {}
            self.index = [("random", ci, -1) for ci in range(len(cases))
                          for _ in range(windows_per_case)]
            if event_windows_per_case > 0:
                event_classes = [1, 3, 5, 7, 9]
                for ci, c in enumerate(cases):
                    for cls in event_classes:
                        pos = np.where(c.labels == cls)[0]
                        if len(pos) == 0:
                            continue
                        self.event_positions[(ci, cls)] = pos
                        self.index.extend(
                            ("event", ci, cls)
                            for _ in range(event_windows_per_case)
                        )
        else:
            self.index = []
            for ci, c in enumerate(cases):
                for start in range(0, c.N, window_size):
                    self.index.append((ci, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        T = self.window_size
        if self.mode == "train":
            kind, ci, cls = self.index[idx]
            c = self.cases[ci]
            if kind == "event":
                pos = self.event_positions[(ci, cls)]
                center = int(pos[np.random.randint(0, len(pos))])
                jitter = np.random.randint(0, T)
                start = center - jitter
                start = max(0, min(start, max(0, c.N - T)))
            else:
                start = np.random.randint(0, max(1, c.N - T + 1))
            variants = np.random.randint(0, c.K, size=T)
        else:
            ci, start = self.index[idx]
            c = self.cases[ci]
            variants = np.zeros(T, dtype=np.int64)

        end = min(start + T, c.N)
        t_have = end - start

        # Fancy-indexed gather; np handles fp16 → fp32 cast at .astype below.
        rows = np.arange(start, end)
        fpn1 = c.fpn1[rows, variants[:t_have]]  # (t_have, C1, H1, W1)
        fpn2 = c.fpn2[rows, variants[:t_have]]

        labels  = c.labels[start:end]
        weights = c.weights[start:end]

        if t_have < T:
            # Right-pad with zeros; weight=0 masks these out of the loss/accuracy.
            pad = T - t_have
            fpn1 = np.concatenate([fpn1, np.zeros((pad,) + fpn1.shape[1:],
                                                   dtype=fpn1.dtype)])
            fpn2 = np.concatenate([fpn2, np.zeros((pad,) + fpn2.shape[1:],
                                                   dtype=fpn2.dtype)])
            labels  = np.concatenate([labels,  np.zeros(pad, dtype=np.int64)])
            weights = np.concatenate([weights, np.zeros(pad, dtype=np.float32)])

        return {
            "fpn1":    torch.from_numpy(fpn1.astype(np.float32)),
            "fpn2":    torch.from_numpy(fpn2.astype(np.float32)),
            "labels":  torch.from_numpy(labels),
            "weights": torch.from_numpy(weights),
        }


# ── Class weights ────────────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes):
    """Same formula as sklearn class_weight='balanced':
        w_c = n_samples / (num_classes * count_c)
    Classes absent from the training set get weight 0 so the model never
    backprops a non-existent class."""
    n = len(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, n / (num_classes * counts.clip(min=1)), 0.0)
    return weights.astype(np.float32)


# ── Weighted CE ───────────────────────────────────────────────────────────────

def weighted_ce_loss(logits, labels, weights, class_weights_t):
    """
    logits: (B, T, K)
    labels: (B, T) int64
    weights: (B, T) float — per-frame sample_weight (0 masks)
    class_weights_t: (K,) float — class-balance multiplier
    Returns a scalar mean loss over weight>0 frames.
    """
    B, T, K = logits.shape
    flat_logits  = logits.reshape(-1, K)
    flat_labels  = labels.reshape(-1)
    flat_weights = weights.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")  # (B*T,)
    per_frame = ce * class_weights_t[flat_labels] * flat_weights
    mask = flat_weights > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return per_frame[mask].sum() / mask.sum().clamp(min=1).float()


# ── Train / val epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, sched, class_weights_t, device):
    model.train()
    loss_sum, n_samples = 0.0, 0
    for batch in loader:
        f1 = batch["fpn1"].to(device, non_blocking=True)
        f2 = batch["fpn2"].to(device, non_blocking=True)
        y  = batch["labels"].to(device, non_blocking=True)
        w  = batch["weights"].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        logits = model(f1, f2)
        loss   = weighted_ce_loss(logits, y, w, class_weights_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        sched.step()

        loss_sum  += float(loss.item()) * f1.size(0)
        n_samples += f1.size(0)
    return loss_sum / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, class_weights_t, device):
    model.eval()
    loss_sum, n_samples = 0.0, 0
    correct, total = 0, 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total   = np.zeros(NUM_CLASSES, dtype=np.int64)
    for batch in loader:
        f1 = batch["fpn1"].to(device, non_blocking=True)
        f2 = batch["fpn2"].to(device, non_blocking=True)
        y  = batch["labels"].to(device, non_blocking=True)
        w  = batch["weights"].to(device, non_blocking=True)
        logits = model(f1, f2)
        loss   = weighted_ce_loss(logits, y, w, class_weights_t)
        loss_sum  += float(loss.item()) * f1.size(0)
        n_samples += f1.size(0)
        preds = logits.argmax(dim=-1)
        mask  = w > 0
        correct += ((preds == y) & mask).sum().item()
        total   += mask.sum().item()
        for c in range(NUM_CLASSES):
            cls_mask = (y == c) & mask
            per_class_total[c]   += cls_mask.sum().item()
            per_class_correct[c] += ((preds == c) & cls_mask).sum().item()
    val_loss = loss_sum / max(n_samples, 1)
    val_acc  = correct / max(total, 1)
    per_class_acc = (per_class_correct / np.maximum(per_class_total, 1)).tolist()
    return val_loss, val_acc, per_class_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=FEATURES_DIR_DEFAULT)
    p.add_argument("--out_dir",      default=None,
                   help="Defaults to multiclass_checkpoints_v2/<run_id>")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--window_size",  type=int,   default=64)
    p.add_argument("--windows_per_case", type=int, default=32,
                   help="Random windows per train case per epoch.")
    p.add_argument("--event_windows_per_case", type=int, default=0,
                   help="Additional event-centered windows per train case, "
                        "per event class, per epoch. Useful for smoke tests "
                        "because event labels are extremely sparse.")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--embed_dim",    type=int,   default=64)
    p.add_argument("--gru_hidden",   type=int,   default=64)
    p.add_argument("--num_heads",    type=int,   default=2)
    p.add_argument("--val_frac",     type=float, default=0.20,
                   help="Fraction of cases held out for per-epoch val "
                        "(model selection only — NOT a generalisation estimate). "
                        "Set 0 to train on all cases and select by train loss.")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=0,
                   help="DataLoader workers. 0 = single-process (cases pre-loaded "
                        "in RAM; workers would duplicate ~15 GB).")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Discover features ─────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(args.features_dir, "case_*.npz")))
    if not paths:
        raise SystemExit(f"no case_*.npz in {args.features_dir}; "
                         f"run extract_multiclass_features_v2.py --mode train first")
    print(f"Found {len(paths)} cases in {args.features_dir}/")

    # ── 80/20 split ───────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if args.val_frac <= 0:
        n_val = 0
    else:
        n_val = max(1, int(round(len(shuffled) * args.val_frac)))
    val_paths   = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    if not train_paths:
        raise SystemExit("empty training split; reduce --val_frac")

    print(f"Train cases ({len(train_paths)}): "
          f"{[os.path.basename(p).replace('case_', '').replace('.npz', '') for p in train_paths]}")
    print(f"Val   cases ({len(val_paths)}):   "
          f"{[os.path.basename(p).replace('case_', '').replace('.npz', '') for p in val_paths]}")
    if not val_paths:
        print("Val split disabled; checkpoint selection uses lowest train loss.")

    # ── Load all cases into RAM ───────────────────────────────────────────
    print(f"\nLoading train cases ...")
    train_cases = [CaseFeatures(p) for p in train_paths]
    print(f"Loading val cases ...")
    val_cases   = [CaseFeatures(p) for p in val_paths]

    # Schema consistency check
    C1, C2 = train_cases[0].C1, train_cases[0].C2
    for c in train_cases + val_cases:
        assert (c.C1, c.C2) == (C1, C2), \
            f"channel mismatch on case {c.case_id}: ({c.C1}, {c.C2}) vs ({C1}, {C2})"
    H1, W1 = train_cases[0].fpn1.shape[3:]
    H2, W2 = train_cases[0].fpn2.shape[3:]
    print(f"Feature schema: fpn1=({C1}, {H1}, {W1})  fpn2=({C2}, {H2}, {W2})")

    # ── Class weights from train set only ─────────────────────────────────
    all_train_labels  = np.concatenate([c.labels  for c in train_cases])
    all_train_weights = np.concatenate([c.weights for c in train_cases])
    class_weights = compute_class_weights(all_train_labels, NUM_CLASSES)
    class_weights_t = torch.from_numpy(class_weights).to(device)
    print(f"\nClass histogram (train): "
          f"{np.bincount(all_train_labels, minlength=NUM_CLASSES).tolist()}")
    print(f"Class weights:           "
          f"{[f'{w:.2f}' for w in class_weights]}")
    print(f"Hard-neg train frames:   {int((all_train_weights > 1.0).sum())}")

    # ── Datasets / loaders ────────────────────────────────────────────────
    train_ds = WindowedCaseDataset(train_cases,
                                    window_size=args.window_size,
                                    windows_per_case=args.windows_per_case,
                                    mode="train",
                                    event_windows_per_case=args.event_windows_per_case)
    val_ds = None
    if val_cases:
        val_ds = WindowedCaseDataset(val_cases,
                                     window_size=args.window_size,
                                     mode="val")
    val_len = len(val_ds) if val_ds is not None else 0
    print(f"\nDataset sizes: train={len(train_ds)} windows, val={val_len} windows")
    if args.event_windows_per_case > 0:
        print(f"Event-window oversampling: {args.event_windows_per_case} "
              f"windows/case/event-class/epoch")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = AttnPoolBiGRU(
        fpn1_channels=C1, fpn2_channels=C2,
        embed_dim=args.embed_dim, gru_hidden=args.gru_hidden,
        num_heads=args.num_heads, dropout=args.dropout,
        num_classes=NUM_CLASSES,
    ).to(device)
    pc = model.count_params()
    print(f"\nModel params: total={pc['total']:,}  "
          f"pool1={pc['pool1']:,}  pool2={pc['pool2']:,}  "
          f"gru={pc['gru']:,}  head={pc['head']:,}")

    optimiser = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                       T_max=max(1, total_steps))

    # ── Output dir + log files ────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if args.out_dir is None:
        args.out_dir = os.path.join(CHECKPOINT_ROOT, run_id)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path  = os.path.join(args.out_dir, "log.txt")
    json_path = os.path.join(args.out_dir, "per_epoch_val.json")
    print(f"\nOutput: {args.out_dir}/")
    with open(log_path, "w") as f:
        f.write(f"# train_multiclass_v2 run_id={run_id}\n")
        f.write(f"# args: {json.dumps(vars(args), indent=2, default=str)}\n")
        f.write(f"# train_cases: {[c.case_id for c in train_cases]}\n")
        f.write(f"# val_cases:   {[c.case_id for c in val_cases]}\n")
        f.write(f"# class_weights: {class_weights.tolist()}\n")

    # ── Train loop ────────────────────────────────────────────────────────
    epoch_records = []
    best_val_acc  = float("-inf")
    best_path     = os.path.join(args.out_dir, "checkpoint_best.pt")

    print(f"\n{'='*70}\n  Training {args.epochs} epochs\n{'='*70}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimiser, sched, class_weights_t, device,
        )
        train_dt = time.perf_counter() - t0

        if val_loader is not None:
            t1 = time.perf_counter()
            val_loss, val_acc, per_class_acc = validate(
                model, val_loader, class_weights_t, device,
            )
            val_dt = time.perf_counter() - t1
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            per_class_acc = [float("nan")] * NUM_CLASSES
            val_dt = 0.0
        lr = sched.get_last_lr()[0]

        # Macro-style mean over event classes only — the metric we care about.
        event_classes = [1, 3, 5, 7, 9]
        event_accs = [per_class_acc[c] for c in event_classes
                      if not np.isnan(per_class_acc[c])]
        event_mean_acc = float(np.mean(event_accs)) if event_accs else float("nan")

        line = (f"epoch={epoch:>3d}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  event_acc={event_mean_acc:.4f}  "
                f"lr={lr:.2e}  t_train={train_dt:.1f}s  t_val={val_dt:.1f}s")
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        epoch_records.append({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "val_acc":        val_acc,
            "event_mean_acc": event_mean_acc,
            "per_class_acc":  per_class_acc,
            "lr":             lr,
            "t_train_s":      train_dt,
            "t_val_s":        val_dt,
        })
        with open(json_path, "w") as f:
            json.dump(epoch_records, f, indent=2)

        # Save best by val_acc (consider event_mean_acc as a tiebreaker).
        # With val disabled, select by lowest training loss for smoke tests.
        score = -train_loss if val_loader is None else val_acc + 0.1 * event_mean_acc
        if score > best_val_acc:
            best_val_acc = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "event_mean_acc":   event_mean_acc,
                "per_class_acc":    per_class_acc,
                "class_weights":    class_weights.tolist(),
                "fpn1_channels":    C1,
                "fpn2_channels":    C2,
                "fpn1_spatial":     (H1, W1),
                "fpn2_spatial":     (H2, W2),
                "model_kwargs": {
                    "fpn1_channels": C1,
                    "fpn2_channels": C2,
                    "embed_dim":     args.embed_dim,
                    "gru_hidden":    args.gru_hidden,
                    "num_heads":     args.num_heads,
                    "dropout":       args.dropout,
                    "num_classes":   NUM_CLASSES,
                },
                "args":             vars(args),
                "train_cases":      [c.case_id for c in train_cases],
                "val_cases":        [c.case_id for c in val_cases],
            }, best_path)
            print(f"  ↳ new best (score={score:.4f}, val_acc={val_acc:.4f}) → "
                  f"checkpoint_best.pt")
            with open(log_path, "a") as f:
                f.write(f"  ↳ new best (score={score:.4f}) saved\n")

    # ── Save final ────────────────────────────────────────────────────────
    final_path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            args.epochs,
        "class_weights":    class_weights.tolist(),
        "fpn1_channels":    C1,
        "fpn2_channels":    C2,
        "model_kwargs": {
            "fpn1_channels": C1,
            "fpn2_channels": C2,
            "embed_dim":     args.embed_dim,
            "gru_hidden":    args.gru_hidden,
            "num_heads":     args.num_heads,
            "dropout":       args.dropout,
            "num_classes":   NUM_CLASSES,
        },
        "args":             vars(args),
        "train_cases":      [c.case_id for c in train_cases],
        "val_cases":        [c.case_id for c in val_cases],
    }, final_path)
    print(f"\nFinal checkpoint → {final_path}")
    print(f"Best checkpoint  → {best_path}  (score={best_val_acc:.4f})")
    print(f"Log              → {log_path}")
    print(f"Per-epoch JSON   → {json_path}")


if __name__ == "__main__":
    main()
