#!/usr/bin/env bash
# infer_vas.sh
# Run VAS segmentation inference on a clip from a source video.
# Run from an interactive GPU node (bsub -Is -q gpu ...).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SOURCE_VIDEO="/sc/arion/projects/video_rarp/neel_projects/intuitive_videos/case_214_clipped.mp4"
OUT_DIR="${SCRIPT_DIR}/vas_inference/case_214_900-930"

DET_CKPT="${SCRIPT_DIR}/detector_training/checkpoints/20260522_1403/checkpoint_best.pth"
DECODER_CKPT="${SCRIPT_DIR}/sam2_decoder_training/checkpoints/20260522_1448/checkpoint_best.pth"

START_SEC=900   # 15:00
END_SEC=930     # 15:30

mkdir -p "$OUT_DIR"

python3 "${SCRIPT_DIR}/infer_prostate_bidir.py" \
    --video          "${SOURCE_VIDEO}"  \
    --detector_ckpt  "${DET_CKPT}"      \
    --decoder_ckpt   "${DECODER_CKPT}"  \
    --output_dir     "${OUT_DIR}"       \
    --start_sec      "${START_SEC}"     \
    --end_sec        "${END_SEC}"

echo "Done. Output → ${OUT_DIR}/overlay.mp4"
