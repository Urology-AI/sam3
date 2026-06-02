#!/usr/bin/env bash
# run_phase_localization.sh
# =========================
# Extract SAM2 features for an annotated surgical phase, then run LOCO-CV
# full-video localisation for that phase.
#
# Usage:
#   bash run_phase_localization.sh <event>
#   bash run_phase_localization.sh <event> --sam2_variant small --fast
#
# Where <event> is one of the values in the `event` column of annotate_fine.csv:
#   endobag, vas_cut_1, vas_cut_2, catheter_pull,
#   apical_cut, posterior_cut, seminal_peeling
#
# Env overrides:
#   SAM2_VARIANT  (default: small)
#   SAMPLE_FPS    (default: 1.0 for extract, 0.5 for localise)
#   SKIP_EXTRACT  (default: 0)  set 1 to skip extraction if features exist
#   FAST          (default: 1)  set 0 to run localisation with no optimisations
#
# Examples:
#   bash run_phase_localization.sh vas_cut_1
#   SAM2_VARIANT=large FAST=0 bash run_phase_localization.sh apical_cut
#   SKIP_EXTRACT=1 bash run_phase_localization.sh endobag

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <event> [extra args forwarded to localize_endobag.py]"
    echo ""
    echo "Known events in annotate_fine.csv:"
    echo "  endobag  vas_cut_1  vas_cut_2  catheter_pull"
    echo "  apical_cut  posterior_cut  seminal_peeling"
    exit 1
fi

EVENT="$1"; shift

SAM3_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SAM3_DIR"

SAM2_VARIANT="${SAM2_VARIANT:-small}"
SAMPLE_FPS_EXTRACT="${SAMPLE_FPS_EXTRACT:-1.0}"
SAMPLE_FPS_LOCALIZE="${SAMPLE_FPS_LOCALIZE:-0.5}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FAST="${FAST:-1}"

FEATURES_DIR="${SAM3_DIR}/${EVENT}_features"
OUT_DIR="${SAM3_DIR}/${EVENT}_localization"

echo "============================================================"
echo "  Phase localisation: ${EVENT}"
echo "  backbone:     ${SAM2_VARIANT}"
echo "  features →    ${FEATURES_DIR}"
echo "  localisation →${OUT_DIR}"
echo "  fast mode:    ${FAST}"
echo "============================================================"

# ── Step 1: extract features ──────────────────────────────────────────────────
if [ "$SKIP_EXTRACT" = "1" ] && compgen -G "${FEATURES_DIR}/case_*.npz" > /dev/null; then
    echo ""
    echo "[skip extract] ${FEATURES_DIR}/ already populated and SKIP_EXTRACT=1"
else
    echo ""
    echo "[step 1] Extracting ${EVENT} features..."
    python3 extract_endobag_features.py \
        --event "$EVENT" \
        --sam2_variant "$SAM2_VARIANT" \
        --sample_fps "$SAMPLE_FPS_EXTRACT"
fi

# ── Step 2: localise across all cases (LOCO-CV) ──────────────────────────────
LOCALIZE_FLAGS=( --event "$EVENT"
                 --sam2_variant "$SAM2_VARIANT"
                 --sample_fps "$SAMPLE_FPS_LOCALIZE"
                 --all )
if [ "$FAST" = "1" ]; then
    LOCALIZE_FLAGS+=( --fast )
fi

echo ""
echo "[step 2] Localising ${EVENT} (LOCO-CV across all cases with features)..."
python3 localize_endobag.py "${LOCALIZE_FLAGS[@]}" "$@"

echo ""
echo "============================================================"
echo "  Done. Results: ${OUT_DIR}/"
echo "============================================================"
