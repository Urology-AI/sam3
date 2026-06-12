#!/usr/bin/env bash
# run_multiclass_localization.sh
# ==============================
# End-to-end driver for the 11-class softmax + Viterbi pipeline:
#   1. extract_multiclass_features.py  (per-case SAM2 features + sample weights)
#   2. train_multiclass_classifier.py   (LOCO sanity-check metrics)
#   3. localize_multiclass.py           (LOCO full-video sweep + per-event errors)
#
# Env overrides:
#   SAM2_VARIANT         (default: small)
#   SAMPLE_FPS_EXTRACT   (default: 1.0)
#   SAMPLE_FPS_LOCALIZE  (default: 0.5)
#   SAFETY_GAP           (default: 3.0)   seconds dropped around every event boundary
#   HARD_NEG_DURATION    (default: 10.0)  hard-negative window length post-event
#   HARD_NEG_WEIGHT      (default: 5.0)   sample-weight multiplier in hard-neg window
#   ALPHA                (default: 0.0)   event-anchor log-prob bonus in Viterbi
#   SKIP_COST            (default: 0.0)   log-prob added to skip edges (≤0 only)
#   SKIP_EXTRACT         (default: 0)
#   SKIP_TRAIN_EVAL      (default: 0)     skip the LOCO sanity-check step
#   FAST                 (default: 1)
#
# Extra args after the script name are forwarded to localize_multiclass.py.
#
# Examples:
#   bash run_multiclass_localization.sh
#   SAFETY_GAP=4 HARD_NEG_DURATION=15 bash run_multiclass_localization.sh
#   ALPHA=2.0 bash run_multiclass_localization.sh
#   SKIP_EXTRACT=1 bash run_multiclass_localization.sh --hold_out 219
#   SAM2_VARIANT=large FAST=0 bash run_multiclass_localization.sh

set -euo pipefail

SAM3_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SAM3_DIR"

SAM2_VARIANT="${SAM2_VARIANT:-small}"
SAMPLE_FPS_EXTRACT="${SAMPLE_FPS_EXTRACT:-1.0}"
SAMPLE_FPS_LOCALIZE="${SAMPLE_FPS_LOCALIZE:-0.5}"
SAFETY_GAP="${SAFETY_GAP:-3.0}"
HARD_NEG_DURATION="${HARD_NEG_DURATION:-10.0}"
HARD_NEG_WEIGHT="${HARD_NEG_WEIGHT:-5.0}"
ALPHA="${ALPHA:-0.0}"
SKIP_COST="${SKIP_COST:-0.0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_TRAIN_EVAL="${SKIP_TRAIN_EVAL:-0}"
FAST="${FAST:-1}"

FEATURES_DIR="${SAM3_DIR}/multiclass_features"
OUT_DIR="${SAM3_DIR}/multiclass_localization"

echo "============================================================"
echo "  Multi-class phase localisation"
echo "  backbone           : ${SAM2_VARIANT}"
echo "  sample_fps         : extract=${SAMPLE_FPS_EXTRACT}  localize=${SAMPLE_FPS_LOCALIZE}"
echo "  safety_gap         : ${SAFETY_GAP}s  (drop around every event boundary)"
echo "  hard_neg           : ${HARD_NEG_DURATION}s × ${HARD_NEG_WEIGHT}  (post-event, clipped against next event)"
echo "  viterbi alpha      : ${ALPHA}  (event-anchor bonus)"
echo "  viterbi skip_cost  : ${SKIP_COST}"
echo "  features → ${FEATURES_DIR}"
echo "  results  → ${OUT_DIR}"
echo "  fast               : ${FAST}"
echo "============================================================"

# ── Step 1: extract ──────────────────────────────────────────────────────────
if [ "$SKIP_EXTRACT" = "1" ] && compgen -G "${FEATURES_DIR}/case_*.npz" > /dev/null; then
    echo ""
    echo "[skip extract] ${FEATURES_DIR}/ already populated and SKIP_EXTRACT=1"
else
    echo ""
    echo "[step 1] Extracting multi-class features..."
    EXTRACT_FLAGS=( --sam2_variant "$SAM2_VARIANT"
                    --sample_fps "$SAMPLE_FPS_EXTRACT"
                    --safety_gap "$SAFETY_GAP"
                    --hard_neg_duration "$HARD_NEG_DURATION"
                    --hard_neg_weight "$HARD_NEG_WEIGHT" )
    if [ "$FAST" = "1" ]; then
        EXTRACT_FLAGS+=( --use_bf16 )
    fi
    python3 extract_multiclass_features.py "${EXTRACT_FLAGS[@]}"
fi

# ── Step 2: LOCO-CV sanity check ────────────────────────────────────────────
if [ "$SKIP_TRAIN_EVAL" != "1" ]; then
    echo ""
    echo "[step 2] LOCO-CV classifier sanity check..."
    python3 train_multiclass_classifier.py --features_dir "$FEATURES_DIR" \
                                            --out_dir "$OUT_DIR"
fi

# ── Step 3: localise ────────────────────────────────────────────────────────
echo ""
echo "[step 3] Full-video LOCO localisation..."
LOCALIZE_FLAGS=( --sam2_variant "$SAM2_VARIANT"
                 --sample_fps "$SAMPLE_FPS_LOCALIZE"
                 --features_dir "$FEATURES_DIR"
                 --out_dir "$OUT_DIR"
                 --alpha "$ALPHA"
                 --skip_cost "$SKIP_COST" )
if [ "$FAST" = "1" ]; then
    LOCALIZE_FLAGS+=( --fast )
fi
# If the caller forwarded extra args (e.g. --hold_out 219), respect them; otherwise --all.
if [[ "$*" != *--hold_out* && "$*" != *--all* ]]; then
    LOCALIZE_FLAGS+=( --all )
fi

python3 localize_multiclass.py "${LOCALIZE_FLAGS[@]}" "$@"

echo ""
echo "============================================================"
echo "  Done. Results: ${OUT_DIR}/"
echo "============================================================"
