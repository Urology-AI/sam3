#!/usr/bin/env python3
"""
train_multiclass_classifier.py
==============================
LOCO-CV evaluation of the 13-class softmax phase classifier on SAM2 features.

This is a sanity check before bothering with full-video localisation: do the
backbone features actually carry phase-level signal across all 13 classes?

Inputs:  .npz files produced by extract_multiclass_features.py.
Each contains: features [N, 640], labels [N] int 0..12, sample_weights [N] float.

Output: per-class AUC-ROC (one-vs-rest), top-1 accuracy, macro F1, and a 13×13
confusion matrix saved as PNG.

Usage
-----
  python3 train_multiclass_classifier.py
  python3 train_multiclass_classifier.py --features_dir multiclass_features
"""

import argparse
import glob
import os

import numpy as np

SAM3_DIR             = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR_DEFAULT = os.path.join(SAM3_DIR, "multiclass_features")

NUM_CLASSES = 11
CLASS_NAMES = [
    "pre_catheter_pull", "catheter_pull",
    "post_catheter_pull", "posterior_cut",
    "post_posterior_cut", "vas_cut",
    "post_vas_cut",      "apical_cut",
    "post_apical_cut",   "endobag",
    "post_endobag",
]
EVENT_CLASSES = [1, 3, 5, 7, 9]   # the 5 milestone events (vas_cut_1 + vas_cut_2 merged)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_all_cases(features_dir):
    cases = []
    for path in sorted(glob.glob(os.path.join(features_dir, "case_*.npz"))):
        d = np.load(path, allow_pickle=True)
        case_id = str(d["case_id"])
        features = d["features"].astype(np.float32)
        labels   = d["labels"].astype(np.int32)
        weights  = d["sample_weights"].astype(np.float32) \
                   if "sample_weights" in d.files else np.ones(len(labels), np.float32)
        cases.append((case_id, features, labels, weights))
        # per-case class histogram
        hist = " ".join(f"{c}:{int((labels == c).sum())}" for c in range(NUM_CLASSES)
                        if (labels == c).any())
        print(f"  case {case_id}: N={len(labels)}  hard_neg={(weights > 1).sum()}  [{hist}]")
    return cases


# ── Scaling ────────────────────────────────────────────────────────────────────

def fit_scaler(X):
    return X.mean(axis=0), X.std(axis=0) + 1e-8

def apply_scaler(X, m, s):
    return (X - m) / s


# ── Padded probability helper ──────────────────────────────────────────────────

def pad_proba(proba, classes_):
    """sklearn predict_proba returns (N, K) over only the classes seen in training.
    Re-expand to (N, NUM_CLASSES) with zeros for missing classes."""
    if len(classes_) == NUM_CLASSES and np.array_equal(classes_, np.arange(NUM_CLASSES)):
        return proba
    full = np.zeros((len(proba), NUM_CLASSES), dtype=proba.dtype)
    for i, c in enumerate(classes_):
        full[:, c] = proba[:, i]
    return full


# ── LOCO-CV ────────────────────────────────────────────────────────────────────

def loco_cv(cases, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    fold_results = []

    for i, (test_id, X_te, y_te, _w_te) in enumerate(cases):
        train = [(X, y, w) for j, (_, X, y, w) in enumerate(cases) if j != i]
        X_tr = np.concatenate([X for X, _, _ in train], axis=0)
        y_tr = np.concatenate([y for _, y, _ in train], axis=0)
        w_tr = np.concatenate([w for _, _, w in train], axis=0)

        m, s = fit_scaler(X_tr)
        X_tr_s = apply_scaler(X_tr, m, s)
        X_te_s = apply_scaler(X_te, m, s)

        # multi_class="multinomial" with lbfgs is sklearn's default softmax logreg
        # for >2 classes; class_weight="balanced" multiplies with sample_weight.
        clf = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
        proba = pad_proba(clf.predict_proba(X_te_s), clf.classes_)
        pred  = proba.argmax(axis=1)

        # Per-class one-vs-rest AUC
        per_class_auc = {}
        for c in range(NUM_CLASSES):
            y_bin = (y_te == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                per_class_auc[c] = float("nan")
            else:
                per_class_auc[c] = roc_auc_score(y_bin, proba[:, c])
        macro_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
        acc      = accuracy_score(y_te, pred)

        # Aggregate metric for the 6 event classes only (more decision-relevant)
        event_aucs = [per_class_auc[c] for c in EVENT_CLASSES
                      if not np.isnan(per_class_auc[c])]
        event_mean_auc = float(np.mean(event_aucs)) if event_aucs else float("nan")

        fold_results.append({
            "case_id":         test_id,
            "y_true":          y_te,
            "y_pred":          pred,
            "proba":           proba,
            "per_class_auc":   per_class_auc,
            "event_mean_auc":  event_mean_auc,
            "macro_f1":        macro_f1,
            "acc":             acc,
        })

        auc_str = "  ".join(f"{c}={per_class_auc[c]:.2f}" if not np.isnan(per_class_auc[c])
                            else f"{c}=NA" for c in EVENT_CLASSES)
        print(f"  fold {i+1}/{len(cases)}  test=case_{test_id:<4}  "
              f"event_AUC[{auc_str}]  macroF1={macro_f1:.3f}  acc={acc:.3f}")

    return fold_results


# ── Reporting ──────────────────────────────────────────────────────────────────

def report(fold_results):
    print(f"\n{'='*70}")
    print(f"  Multi-class LOCO-CV summary ({len(fold_results)} cases)")
    print(f"{'='*70}")
    # Per-class mean AUC across folds
    print(f"\n  Per-class one-vs-rest AUC:")
    for c in range(NUM_CLASSES):
        aucs = [r["per_class_auc"][c] for r in fold_results
                if not np.isnan(r["per_class_auc"][c])]
        if aucs:
            tag = " (event)" if c in EVENT_CLASSES else ""
            print(f"    cls {c:>2}  {CLASS_NAMES[c]:<22}{tag:<9}  "
                  f"AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
                  f"(min={min(aucs):.3f}  max={max(aucs):.3f}  n={len(aucs)})")
        else:
            print(f"    cls {c:>2}  {CLASS_NAMES[c]:<22}          AUC = NA")

    event_aucs = [r["event_mean_auc"] for r in fold_results
                  if not np.isnan(r["event_mean_auc"])]
    macro_f1s  = [r["macro_f1"] for r in fold_results]
    accs       = [r["acc"]      for r in fold_results]
    print(f"\n  Mean event-class AUC : {np.mean(event_aucs):.3f} ± {np.std(event_aucs):.3f}")
    print(f"  Macro F1             : {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
    print(f"  Top-1 accuracy       : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"{'='*70}\n")


def save_confusion(fold_results, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("  matplotlib/sklearn missing — skipping plot")
        return

    y_true = np.concatenate([r["y_true"] for r in fold_results])
    y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("13-class LOCO confusion matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Confusion → {out_png}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=FEATURES_DIR_DEFAULT)
    p.add_argument("--out_dir",      default=None,
                   help="Defaults to <features_dir>.")
    p.add_argument("--C",            type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.features_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading features from {args.features_dir}/")
    cases = load_all_cases(args.features_dir)
    if not cases:
        print("No .npz files found. Run extract_multiclass_features.py first.")
        return

    print(f"\nLOCO-CV with multinomial logreg (C={args.C})\n")
    fold_results = loco_cv(cases, C=args.C)
    report(fold_results)
    save_confusion(fold_results, os.path.join(out_dir, "confusion_multiclass.png"))


if __name__ == "__main__":
    main()
