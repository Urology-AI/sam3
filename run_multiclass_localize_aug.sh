#!/usr/bin/env bash
# run_multiclass_localize_aug.sh
# ==============================
# v1 multi-class softmax + 11-state Viterbi pipeline (run_multiclass_localization.sh)
# + v2 augmentation recipe + OOD folder inference.
#
#   1. Extract augmented multi-class features on the 15 intuitive cases.
#      Sample plan = v1 multiclass: events + interphases + safety-gap drop
#      + hard-negative window (10s post-event x 5x weight).
#   2. Extract OOD folder features (multi-chunk stitched into surgery-absolute
#      time, no augmentation).
#   3. Train multinomial logistic regression on ALL augmented train features
#      with sample_weight (the hard-neg multiplier from step 1) and
#      class_weight="balanced" -- identical to localize_multiclass.train_classifier.
#   4. Predict log-probs on the OOD features, smooth per class, run the
#      v1 11-state Viterbi (skip edges 1->3 and 7->9), extract per-event
#      windows, score against the OOD CSV.
#
# Env overrides:
#   K_AUG              default: 4
#   SAM2_VARIANT       default: small
#   SAMPLE_FPS_EXTRACT default: 1.0
#   SAMPLE_FPS_VAL     default: 0.5
#   BATCH_SIZE_TRAIN   default: 16
#   BATCH_SIZE_VAL     default: 32
#   SAFETY_GAP         default: 3.0
#   HARD_NEG_DURATION  default: 10.0
#   HARD_NEG_WEIGHT    default: 5.0
#   SMOOTH_WINDOW      default: 20.0   (per-class log-prob rolling mean)
#   ALPHA              default: 0.0    (Viterbi event-anchor bonus)
#   SKIP_COST          default: 0.0    (Viterbi skip-edge cost, <=0)
#   OOD_FOLDER         default: SUBJ_1b7d93c2_Y2025_DOY143
#   OOD_EVENT_CSV      default: events_SUBJ_1b7d93c2_Y2025_DOY143_*.csv
#   SKIP_TRAIN_EXTRACT default: 0
#   SKIP_VAL_EXTRACT   default: 0
#
# Usage:
#   bash run_multiclass_localize_aug.sh
#   K_AUG=6 SAM2_VARIANT=large bash run_multiclass_localize_aug.sh
#   SKIP_TRAIN_EXTRACT=1 bash run_multiclass_localize_aug.sh

set -euo pipefail

SAM3_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SAM3_DIR}"

K_AUG="${K_AUG:-4}"
SAM2_VARIANT="${SAM2_VARIANT:-small}"
SAMPLE_FPS_EXTRACT="${SAMPLE_FPS_EXTRACT:-1.0}"
SAMPLE_FPS_VAL="${SAMPLE_FPS_VAL:-0.5}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-16}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-32}"
SAFETY_GAP="${SAFETY_GAP:-3.0}"
HARD_NEG_DURATION="${HARD_NEG_DURATION:-10.0}"
HARD_NEG_WEIGHT="${HARD_NEG_WEIGHT:-5.0}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-20.0}"
ALPHA="${ALPHA:-0.0}"
SKIP_COST="${SKIP_COST:-0.0}"

OOD_FOLDER="${OOD_FOLDER:-/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_1b7d93c2_Y2025_DOY143}"
OOD_EVENT_CSV="${OOD_EVENT_CSV:-${SAM3_DIR}/event_annotations/events_SUBJ_1b7d93c2_Y2025_DOY143_1780943931906.csv}"

SKIP_TRAIN_EXTRACT="${SKIP_TRAIN_EXTRACT:-0}"
SKIP_VAL_EXTRACT="${SKIP_VAL_EXTRACT:-0}"

FEAT_ROOT="${SAM3_DIR}/multiclass_features_aug"
TRAIN_DIR="${FEAT_ROOT}/train"
VAL_DIR="${FEAT_ROOT}/val"
VAL_NAME="$(basename "${OOD_FOLDER%/}")"
VAL_NPZ="${VAL_DIR}/${VAL_NAME}.npz"
OUT_DIR="${SAM3_DIR}/multiclass_localization_aug"

echo "============================================================"
echo "  Multiclass phase localisation: v1 + v2 aug, OOD folder inference"
echo "  K_AUG              : ${K_AUG}"
echo "  SAM2_VARIANT       : ${SAM2_VARIANT}"
echo "  safety_gap         : ${SAFETY_GAP}s"
echo "  hard_neg           : ${HARD_NEG_DURATION}s x ${HARD_NEG_WEIGHT}"
echo "  smooth_window      : ${SMOOTH_WINDOW}s"
echo "  viterbi alpha      : ${ALPHA}"
echo "  viterbi skip_cost  : ${SKIP_COST}"
echo "  OOD folder         : ${OOD_FOLDER}"
echo "  OOD event csv      : ${OOD_EVENT_CSV}"
echo "  Train features dir : ${TRAIN_DIR}"
echo "  Val features dir   : ${VAL_DIR}"
echo "  Localisation out   : ${OUT_DIR}"
echo "============================================================"

# ── Step 1: extract augmented train features ─────────────────────────────────
if [[ "${SKIP_TRAIN_EXTRACT}" != "1" ]]; then
    echo ""
    echo "[step 1] Extracting augmented multi-class train features ..."
    python3 extract_multiclass_features_aug.py \
        --mode              train \
        --sample_fps        "${SAMPLE_FPS_EXTRACT}" \
        --safety_gap        "${SAFETY_GAP}" \
        --hard_neg_duration "${HARD_NEG_DURATION}" \
        --hard_neg_weight   "${HARD_NEG_WEIGHT}" \
        --k_aug             "${K_AUG}" \
        --sam2_variant      "${SAM2_VARIANT}" \
        --batch_size        "${BATCH_SIZE_TRAIN}" \
        --use_bf16 \
        --out_dir           "${TRAIN_DIR}"
else
    echo ""
    echo "[step 1] skip (SKIP_TRAIN_EXTRACT=1)"
fi

# ── Step 2: extract OOD val features ─────────────────────────────────────────
if [[ "${SKIP_VAL_EXTRACT}" != "1" || ! -f "${VAL_NPZ}" ]]; then
    echo ""
    echo "[step 2] Extracting OOD folder features (no aug) ..."
    python3 extract_multiclass_features_aug.py \
        --mode         val \
        --video_folder "${OOD_FOLDER}" \
        --sample_fps   "${SAMPLE_FPS_VAL}" \
        --sam2_variant "${SAM2_VARIANT}" \
        --batch_size   "${BATCH_SIZE_VAL}" \
        --use_bf16 \
        --out_dir      "${VAL_DIR}"
else
    echo ""
    echo "[step 2] skip (${VAL_NPZ} exists and SKIP_VAL_EXTRACT=1)"
fi

if ! compgen -G "${TRAIN_DIR}/case_*.npz" > /dev/null; then
    echo "ERROR: no train .npz files in ${TRAIN_DIR}"
    exit 1
fi
if [[ ! -f "${VAL_NPZ}" ]]; then
    echo "ERROR: val .npz missing -> ${VAL_NPZ}"
    exit 1
fi

# ── Step 3: train + localise ─────────────────────────────────────────────────
echo ""
echo "[step 3] Training multinomial LR + Viterbi decoding on OOD ..."
python3 localize_multiclass_aug.py \
    --train_dir     "${TRAIN_DIR}" \
    --val_npz       "${VAL_NPZ}" \
    --event_csv     "${OOD_EVENT_CSV}" \
    --out_dir       "${OUT_DIR}" \
    --smooth_window "${SMOOTH_WINDOW}" \
    --alpha         "${ALPHA}" \
    --skip_cost     "${SKIP_COST}"

echo ""
echo "============================================================"
echo "  Done. Results in: ${OUT_DIR}/"
echo "============================================================"
