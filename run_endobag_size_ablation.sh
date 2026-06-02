#!/usr/bin/env bash
# run_endobag_size_ablation.sh
# ============================
# Backbone-size + FPN-scale ablation for the endobagging classifier.
#
# Step 1: extract Hiera-S features  (skipped if endobag_features_small/ already populated)
# Step 2: LOCO-CV classifier ablation
#           variants: large, small
#           slices:   all (1024-d), fpn1 (mid-level 512-d), fpn2 (scene-level 512-d)
# Step 3 (optional, gated by RUN_LOCALIZATION=1):
#           full-video localisation with Hiera-S, for head-to-head against the
#           existing large baseline in endobag_localization/.
#
# Usage:
#   bash run_endobag_size_ablation.sh
#   RUN_LOCALIZATION=1 bash run_endobag_size_ablation.sh

set -euo pipefail

SAM3_DIR="/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3"
cd "$SAM3_DIR"

LARGE_FEATS="endobag_features"
SMALL_FEATS="endobag_features_small"

RUN_LOCALIZATION="${RUN_LOCALIZATION:-0}"

# ── Step 1: extract Hiera-S features ──────────────────────────────────────────
if compgen -G "$SMALL_FEATS/case_*.npz" > /dev/null; then
    echo "[skip] small features already present in $SMALL_FEATS/"
else
    echo "[step 1] Extracting Hiera-S features → $SMALL_FEATS/"
    python3 extract_endobag_features.py \
        --sam2_variant small \
        --out_dir "$SMALL_FEATS"
fi

# ── Step 2: classifier-level LOCO-CV ablation ─────────────────────────────────
for variant in large small; do
    case $variant in
        large) feats=$LARGE_FEATS ;;
        small) feats=$SMALL_FEATS ;;
    esac
    for slice in all fpn1 fpn2; do
        echo ""
        echo "================================================================"
        echo "  LOCO-CV   variant=$variant   slice=$slice"
        echo "================================================================"
        python3 train_endobag_classifier.py \
            --features_dir "$feats" \
            --feature_slice "$slice"
    done
done

# ── Step 3 (optional): full-video localisation with Hiera-S ───────────────────
if [[ "$RUN_LOCALIZATION" == "1" ]]; then
    echo ""
    echo "================================================================"
    echo "  Full-video localisation  variant=small  slice=all"
    echo "================================================================"
    python3 localize_endobag.py \
        --all \
        --sam2_variant small \
        --feature_slice all \
        --features_dir "$SMALL_FEATS" \
        --out_dir endobag_localization_small
fi

echo ""
echo "Done."
