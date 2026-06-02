#!/usr/bin/env python3
"""
train_endobag_classifier.py
===========================
Train and evaluate a binary endobagging classifier on top of SAM2 ViT features.

Input: .npz files produced by extract_endobag_features.py (one per case).
Protocol: leave-one-case-out (LOCO) cross-validation.

Models tried:
  1. Logistic regression
  2. MLP (1 hidden layer)

Metrics: AUC-ROC (primary), F1, accuracy, confusion matrix.

Usage
-----
  python3 train_endobag_classifier.py
  python3 train_endobag_classifier.py --features_dir custom_features/ --model mlp
  python3 train_endobag_classifier.py --model both
"""

import argparse
import os
import glob

import numpy as np

SAM3_DIR             = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR_DEFAULT = os.path.join(SAM3_DIR, "endobag_features")

# Saved feature vector layout (1024-d):
#   [0:256]    fpn[1] avg-pool   (mid-level)
#   [256:512]  fpn[1] max-pool   (mid-level)
#   [512:768]  fpn[2] avg-pool   (scene-level)
#   [768:1024] fpn[2] max-pool   (scene-level)
SLICE_RANGES = {
    "all":  (None, None),
    "fpn1": (0,    512),
    "fpn2": (512,  None),
}


def apply_feature_slice(features, slice_name):
    lo, hi = SLICE_RANGES[slice_name]
    return features[:, lo:hi]


def load_all_cases(features_dir, feature_slice="all"):
    cases = []
    for path in sorted(glob.glob(os.path.join(features_dir, "case_*.npz"))):
        d = np.load(path, allow_pickle=True)
        case_id  = str(d["case_id"])
        features = d["features"].astype(np.float32)
        features = apply_feature_slice(features, feature_slice)
        labels   = d["labels"].astype(np.int32)
        cases.append((case_id, features, labels))
        n_pos = labels.sum()
        print(f"  case {case_id}: {len(labels)} frames  pos={n_pos}  neg={len(labels)-n_pos}"
              f"  feat_dim={features.shape[1]}")
    return cases


def fit_scaler(X_train):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return mean, std

def apply_scaler(X, mean, std):
    return (X - mean) / std


def loco_cv(cases, model_type, C=1.0, hidden=256):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    results = []

    for i, (test_id, X_test, y_test) in enumerate(cases):
        train_cases = [(cid, X, y) for j, (cid, X, y) in enumerate(cases) if j != i]
        X_train = np.concatenate([X for _, X, _ in train_cases], axis=0)
        y_train = np.concatenate([y for _, _, y in train_cases], axis=0)

        mean, std = fit_scaler(X_train)
        X_train_s = apply_scaler(X_train, mean, std)
        X_test_s  = apply_scaler(X_test,  mean, std)

        if model_type == "logreg":
            clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                     solver="lbfgs", random_state=42)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=500,
                                early_stopping=True, validation_fraction=0.1,
                                random_state=42, learning_rate_init=1e-3)

        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        f1  = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        results.append({"case_id": test_id, "auc": auc, "f1": f1, "acc": acc,
                         "y_true": y_test, "y_prob": y_prob})
        print(f"  fold {i+1}/{len(cases)}  test=case_{test_id:>3}  "
              f"AUC={auc:.3f}  F1={f1:.3f}  acc={acc:.3f}  "
              f"(train: {len(y_train)} frames, test: {len(y_test)} frames)")

    return results


def report(results, model_type):
    aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]
    f1s  = [r["f1"]  for r in results]
    accs = [r["acc"] for r in results]

    print(f"\n{'='*60}")
    print(f"  Model: {model_type}")
    print(f"  LOCO-CV over {len(results)} cases")
    print(f"  AUC-ROC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
          f"(min={np.min(aucs):.3f}  max={np.max(aucs):.3f})")
    print(f"  F1      : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"{'='*60}\n")

    return np.mean(aucs)


def confusion_matrix_plot(results, model_type, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        print("  matplotlib/sklearn not available — skipping plot")
        return

    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_pred_all = np.concatenate([(r["y_prob"] >= 0.5).astype(int) for r in results])
    cm = confusion_matrix(y_true_all, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Other", "Endobagging"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"LOCO-CV confusion matrix ({model_type})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {out_png}")


def roc_plot(results, model_type, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        if np.isnan(r["auc"]):
            continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, alpha=0.4, linewidth=1,
                label=f"case_{r['case_id']} (AUC={r['auc']:.2f})")

    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_prob_all = np.concatenate([r["y_prob"] for r in results])
    from sklearn.metrics import roc_auc_score, roc_curve as rc
    fpr, tpr, _ = rc(y_true_all, y_prob_all)
    auc_pooled = roc_auc_score(y_true_all, y_prob_all)
    ax.plot(fpr, tpr, color="black", linewidth=2,
            label=f"Pooled (AUC={auc_pooled:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — Endobagging classifier ({model_type}, LOCO-CV)")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  ROC plot → {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=FEATURES_DIR_DEFAULT)
    p.add_argument("--model",        default="logreg",
                   choices=["logreg", "mlp", "both"])
    p.add_argument("--C",            type=float, default=1.0)
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--feature_slice", default="all",
                   choices=list(SLICE_RANGES.keys()),
                   help="all=1024-d, fpn1=mid-level 512-d, fpn2=scene-level 512-d")
    p.add_argument("--out_dir",      default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.features_dir

    print(f"Loading features from {args.features_dir}/  (slice={args.feature_slice})")
    cases = load_all_cases(args.features_dir, feature_slice=args.feature_slice)
    if not cases:
        print("No .npz files found. Run extract_endobag_features.py first.")
        return

    n_pos_total = sum(int(y.sum()) for _, _, y in cases)
    n_neg_total = sum(int((1-y).sum()) for _, _, y in cases)
    print(f"\nTotal: {len(cases)} cases  "
          f"{n_pos_total} positive frames  {n_neg_total} negative frames\n")

    models = ["logreg", "mlp"] if args.model == "both" else [args.model]

    tag = f"{args.feature_slice}"
    for model_type in models:
        print(f"\n{'─'*60}")
        print(f"  Running LOCO-CV with {model_type}  (slice={tag})")
        print(f"{'─'*60}")
        results = loco_cv(cases, model_type, C=args.C, hidden=args.hidden)
        report(results, model_type)
        os.makedirs(out_dir, exist_ok=True)
        confusion_matrix_plot(results, model_type,
                              os.path.join(out_dir, f"confusion_{model_type}_{tag}.png"))
        roc_plot(results, model_type,
                 os.path.join(out_dir, f"roc_{model_type}_{tag}.png"))


if __name__ == "__main__":
    main()
