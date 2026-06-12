#!/usr/bin/env bash
# run_v2_pipeline.sh
# -----------------------------------------------------------------------------
# Orchestrator for the v2 multi-class phase-localisation pipeline.
#
# Stages (skip-aware; each stage looks at on-disk state and skips if its
# output already exists):
#   0. v1 baseline on the OOD case (so we have apples-to-apples numbers
#      to compare v2 against on the same recording).
#   1. Extract v2 train features from the 15 intuitive cases (K augmented
#      variants per frame).
#   2. Extract v2 val features from the OOD folder (no augmentation, K=1,
#      stitched timeline).
#   3. Train the AttnPoolBiGRU classifier.
#   4. Validate the best checkpoint on the OOD case.
#   5. Print a side-by-side v1-vs-v2 per-event error table.
#
# Run inside the singularity container:
#   bash run_v2_pipeline.sh
#
# Env overrides (all optional):
#   K_AUG=4                    variants per frame at extraction
#   EPOCHS=30                  training epochs
#   LR=1e-3                    AdamW learning rate
#   BATCH_SIZE=8               training/extraction batch size
#   WINDOWS_PER_CASE=32        random windows per train case per epoch
#   CASES="213,214"            comma-separated case IDs to extract/train on
#                              (smoke test only; default = all 15 cases).
#                              Affects stages 1 and 3.
#   OOD_FOLDER=<path>          OOD video folder (default: SUBJ_1b7d93c2)
#   EVENT_CSV=<path>           events_SUBJ_*.csv with GT
#   V1_CLASSIFIER=<path>       v1 .joblib classifier (default: multiclass_clf_small.joblib)
#   CKPT=<path>                v2 checkpoint to validate. If unset, uses
#                              the most recent checkpoint_best.pt after train.
#   SKIP_BASELINE=1            skip stage 0
#   SKIP_EXTRACT_TRAIN=1       skip stage 1
#   SKIP_EXTRACT_VAL=1         skip stage 2
#   SKIP_TRAIN=1               skip stage 3 (CKPT or latest is used)
#   SKIP_VALIDATE=1            skip stage 4
# -----------------------------------------------------------------------------
set -euo pipefail

SAM3_DIR=/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
cd "${SAM3_DIR}"

# ── Defaults ─────────────────────────────────────────────────────────────────
K_AUG="${K_AUG:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WINDOWS_PER_CASE="${WINDOWS_PER_CASE:-32}"
CASES="${CASES:-}"
OOD_FOLDER="${OOD_FOLDER:-/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_1b7d93c2_Y2025_DOY143}"
EVENT_CSV="${EVENT_CSV:-${SAM3_DIR}/event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv}"
V1_CLASSIFIER="${V1_CLASSIFIER:-${SAM3_DIR}/multiclass_clf_small.joblib}"

OOD_NAME=$(basename "${OOD_FOLDER%/}")

TRAIN_FEAT_DIR="${SAM3_DIR}/multiclass_features_v2/train"
VAL_FEAT_DIR="${SAM3_DIR}/multiclass_features_v2/val"
VAL_FEAT_FILE="${VAL_FEAT_DIR}/${OOD_NAME}.npz"

BASELINE_OUT="${SAM3_DIR}/multiclass_inference/${OOD_NAME}"
BASELINE_JSON="${BASELINE_OUT}/predictions.json"

CHECKPOINT_ROOT="${SAM3_DIR}/multiclass_checkpoints_v2"
VALIDATION_ROOT="${SAM3_DIR}/multiclass_validation_v2"

ts() { date '+%H:%M:%S'; }

echo "============================================================"
echo "  v2 pipeline   $(date)"
echo "  OOD_FOLDER : ${OOD_FOLDER}"
echo "  EVENT_CSV  : ${EVENT_CSV}"
echo "  K_AUG=${K_AUG}  EPOCHS=${EPOCHS}  LR=${LR}  BATCH=${BATCH_SIZE}"
if [[ -n "${CASES}" ]]; then
    echo "  CASES=${CASES}   (smoke run — limited case set)"
fi
echo "============================================================"

# ── Stage 0: v1 baseline on the OOD case ─────────────────────────────────────
if [[ -n "${SKIP_BASELINE:-}" ]]; then
    echo "[$(ts)] [0] SKIP_BASELINE set — skipping v1 baseline"
elif [[ -f "${BASELINE_JSON}" ]]; then
    echo "[$(ts)] [0] v1 baseline already at ${BASELINE_JSON} — skipping"
else
    if [[ ! -f "${V1_CLASSIFIER}" ]]; then
        echo "ERROR: v1 classifier not found: ${V1_CLASSIFIER}"
        echo "  Train it first: python3 infer_multiclass.py --video <one-case> --save_classifier ${V1_CLASSIFIER} --fast"
        exit 1
    fi
    echo "[$(ts)] [0] Running v1 baseline on ${OOD_NAME} ..."
    python3 infer_multiclass.py \
        --video "${OOD_FOLDER}" \
        --classifier_ckpt "${V1_CLASSIFIER}" \
        --fast
    if [[ ! -f "${BASELINE_JSON}" ]]; then
        echo "ERROR: v1 baseline did not produce ${BASELINE_JSON}"
        exit 1
    fi
fi

# ── Stage 1: extract v2 train features ───────────────────────────────────────
# Skip-policy: when CASES is empty, skip if ≥15 case .npz exist. When CASES
# is set (smoke run), skip only when every requested case already has a file.
needs_extract_train=1
if [[ -n "${SKIP_EXTRACT_TRAIN:-}" ]]; then
    echo "[$(ts)] [1] SKIP_EXTRACT_TRAIN set — skipping train extraction"
    needs_extract_train=0
elif [[ -z "${CASES}" ]]; then
    N_TRAIN_NPZ=$(ls -1 "${TRAIN_FEAT_DIR}"/case_*.npz 2>/dev/null | wc -l || true)
    if [[ "${N_TRAIN_NPZ}" -ge 15 ]]; then
        echo "[$(ts)] [1] ${N_TRAIN_NPZ} train .npz already in ${TRAIN_FEAT_DIR}/ — skipping"
        needs_extract_train=0
    fi
else
    all_present=1
    IFS=',' read -ra _CASE_ARR <<<"${CASES}"
    for c in "${_CASE_ARR[@]}"; do
        if [[ ! -f "${TRAIN_FEAT_DIR}/case_${c}.npz" ]]; then
            all_present=0
            break
        fi
    done
    if [[ "${all_present}" -eq 1 ]]; then
        echo "[$(ts)] [1] All requested CASES=${CASES} already extracted — skipping"
        needs_extract_train=0
    fi
fi

if [[ "${needs_extract_train}" -eq 1 ]]; then
    extra_args=()
    if [[ -n "${CASES}" ]]; then
        extra_args+=(--cases "${CASES}")
    fi
    echo "[$(ts)] [1] Extracting v2 train features (K=${K_AUG}, augmentation ON, " \
         "cases=${CASES:-all}) ..."
    python3 extract_multiclass_features_v2.py \
        --mode train \
        --k_aug "${K_AUG}" \
        --use_bf16 \
        --batch_size "${BATCH_SIZE}" \
        "${extra_args[@]}"
fi

# ── Stage 2: extract v2 val features for the OOD folder ──────────────────────
if [[ -n "${SKIP_EXTRACT_VAL:-}" ]]; then
    echo "[$(ts)] [2] SKIP_EXTRACT_VAL set — skipping val extraction"
elif [[ -f "${VAL_FEAT_FILE}" ]]; then
    echo "[$(ts)] [2] Val features already at ${VAL_FEAT_FILE} — skipping"
else
    echo "[$(ts)] [2] Extracting v2 val features for ${OOD_NAME} ..."
    python3 extract_multiclass_features_v2.py \
        --mode val \
        --video_folder "${OOD_FOLDER}" \
        --event_csv    "${EVENT_CSV}" \
        --use_bf16 \
        --batch_size "${BATCH_SIZE}"
    if [[ ! -f "${VAL_FEAT_FILE}" ]]; then
        echo "ERROR: val extraction did not produce ${VAL_FEAT_FILE}"
        exit 1
    fi
fi

# ── Stage 3: train AttnPoolBiGRU ─────────────────────────────────────────────
if [[ -n "${SKIP_TRAIN:-}" ]]; then
    echo "[$(ts)] [3] SKIP_TRAIN set — skipping training"
else
    echo "[$(ts)] [3] Training AttnPoolBiGRU (epochs=${EPOCHS}, lr=${LR}) ..."
    python3 train_multiclass_v2.py \
        --epochs           "${EPOCHS}" \
        --lr               "${LR}" \
        --batch_size       "${BATCH_SIZE}" \
        --windows_per_case "${WINDOWS_PER_CASE}"
fi

# Resolve checkpoint to validate. Explicit CKPT wins; otherwise pick newest.
if [[ -z "${CKPT:-}" ]]; then
    CKPT=$(ls -1t "${CHECKPOINT_ROOT}"/*/checkpoint_best.pt 2>/dev/null | head -n 1 || true)
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "ERROR: no v2 checkpoint found under ${CHECKPOINT_ROOT}/*/checkpoint_best.pt"
    echo "       Train first or pass CKPT=<path> to use a specific checkpoint."
    exit 1
fi
echo "[$(ts)] Using checkpoint: ${CKPT}"

# ── Stage 4: validate ────────────────────────────────────────────────────────
V2_OUT="${VALIDATION_ROOT}/${OOD_NAME}"
V2_JSON="${V2_OUT}/predictions.json"
if [[ -n "${SKIP_VALIDATE:-}" && -f "${V2_JSON}" ]]; then
    echo "[$(ts)] [4] SKIP_VALIDATE set and ${V2_JSON} exists — reusing"
else
    echo "[$(ts)] [4] Validating on ${OOD_NAME} ..."
    python3 validate_multiclass_v2.py \
        --ckpt          "${CKPT}" \
        --val_features  "${VAL_FEAT_FILE}" \
        --event_csv     "${EVENT_CSV}" \
        --out_tag       "${OOD_NAME}"
fi

# ── Stage 5: v1 vs v2 side-by-side ───────────────────────────────────────────
echo ""
echo "[$(ts)] [5] v1 vs v2 per-event error comparison"

BASELINE_JSON_PY="${BASELINE_JSON}" V2_JSON_PY="${V2_JSON}" EVENT_CSV_PY="${EVENT_CSV}" \
OOD_NAME_PY="${OOD_NAME}" CKPT_PY="${CKPT}" python3 <<'PY'
import csv
import json
import os
import sys

baseline_path = os.environ["BASELINE_JSON_PY"]
v2_path       = os.environ["V2_JSON_PY"]
event_csv     = os.environ["EVENT_CSV_PY"]
ood_name      = os.environ["OOD_NAME_PY"]
ckpt_path     = os.environ["CKPT_PY"]

EVENT_REPORT_ORDER = ["catheter_pull", "posterior_cut", "vas_cut",
                       "apical_cut", "endobag"]
VAS = ("vas_cut_1", "vas_cut_2")


def parse_gt(path):
    gt = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["event"].strip()
            if name in VAS:
                name = "vas_cut"
            if name not in EVENT_REPORT_ORDER:
                continue
            s = float(r["start_sec"]); e = float(r["end_sec"])
            if name in gt:
                old_s, old_e = gt[name]
                gt[name] = (min(s, old_s), max(e, old_e))
            else:
                gt[name] = (s, e)
    return gt


def v1_pred(js, name):
    evt = js.get("events", {}).get(name)
    if evt is None:
        return None
    return (evt["start_s"], evt["end_s"])


def v2_pred(js, name):
    evt = js.get("events", {}).get(name)
    if evt is None or evt.get("pred") is None:
        return None
    p = evt["pred"]
    return (p["start_s"], p["end_s"])


def err_str(pred, gt_window, which):
    if pred is None or gt_window is None:
        return "skip"
    err = pred[0] - gt_window[0] if which == "start" else pred[1] - gt_window[1]
    return f"{err:+.0f}s"


def err_val(pred, gt_window, which):
    if pred is None or gt_window is None:
        return None
    return pred[0] - gt_window[0] if which == "start" else pred[1] - gt_window[1]


with open(baseline_path) as f:
    v1 = json.load(f)
with open(v2_path) as f:
    v2 = json.load(f)
gt = parse_gt(event_csv)

print("")
print("=" * 96)
print(f"  v1 vs v2  case={ood_name}")
print(f"  v1 ckpt: multiclass_clf_small.joblib  (sklearn logreg)")
print(f"  v2 ckpt: {ckpt_path}")
print("=" * 96)
print(f"  {'Event':<16}  {'GT start':>8}  {'GT end':>8}  "
      f"{'v1 err_s':>10}  {'v2 err_s':>10}  {'v1 err_e':>10}  {'v2 err_e':>10}")
print(f"  {'-'*16}  {'-'*8}  {'-'*8}  "
      f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

abs_v1_s, abs_v2_s = [], []
abs_v1_e, abs_v2_e = [], []
for name in EVENT_REPORT_ORDER:
    g = gt.get(name)
    if g is None:
        print(f"  {name:<16}  (no GT)")
        continue
    p1 = v1_pred(v1, name)
    p2 = v2_pred(v2, name)
    print(f"  {name:<16}  {g[0]:>8.0f}  {g[1]:>8.0f}  "
          f"{err_str(p1, g, 'start'):>10}  {err_str(p2, g, 'start'):>10}  "
          f"{err_str(p1, g, 'end'):>10}  {err_str(p2, g, 'end'):>10}")
    e1s = err_val(p1, g, "start");  e2s = err_val(p2, g, "start")
    e1e = err_val(p1, g, "end");    e2e = err_val(p2, g, "end")
    if e1s is not None: abs_v1_s.append(abs(e1s))
    if e2s is not None: abs_v2_s.append(abs(e2s))
    if e1e is not None: abs_v1_e.append(abs(e1e))
    if e2e is not None: abs_v2_e.append(abs(e2e))

print("-" * 96)


def agg(name, vals):
    if not vals:
        print(f"  {name:<24}  no predictions")
        return
    import statistics as st
    median = sorted(vals)[len(vals) // 2]
    mean   = sum(vals) / len(vals)
    mx     = max(vals)
    print(f"  {name:<24}  median={median:>5.0f}s  mean={mean:>5.0f}s  max={mx:>5.0f}s  n={len(vals)}/5")


print("  Aggregate |start error|:")
agg("  v1", abs_v1_s)
agg("  v2", abs_v2_s)
print("  Aggregate |end error|:")
agg("  v1", abs_v1_e)
agg("  v2", abs_v2_e)
print("=" * 96)
PY

echo ""
echo "[$(ts)] DONE."
echo "  v1 baseline    → ${BASELINE_JSON}"
echo "  v2 predictions → ${V2_JSON}"
echo "  v2 plot        → ${V2_OUT}/plot.png"
echo "  v2 errors      → ${V2_OUT}/per_event_errors.txt"
