#!/usr/bin/env bash
# infer_vas_batch.sh
# Run VAS segmentation inference on all clips listed in untitled.txt.
# Run from an interactive GPU node (bsub -Is -q gpu ...).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_DIR="/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
CLIP_LIST="${VIDEO_DIR}/untitled.txt"

DET_CKPT="${SCRIPT_DIR}/detector_training/checkpoints/20260522_1403/checkpoint_best.pth"
DECODER_CKPT="${SCRIPT_DIR}/sam2_decoder_training/checkpoints/20260522_1448/checkpoint_best.pth"

# Convert MM:SS or HH:MM:SS to total seconds (10# forces decimal, avoids octal for 08/09)
to_seconds() {
    local t="$1"
    local IFS=':'
    read -ra parts <<< "$t"
    local secs=0
    for p in "${parts[@]}"; do
        secs=$(( secs * 60 + 10#$p ))
    done
    echo "$secs"
}

total=0
failed=0

while IFS= read -r line <&3 || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue

    # Expected format: "case 219 12:20-12:50"
    case_num=$(echo "$line" | awk '{print $2}')
    timerange=$(echo "$line" | awk '{print $3}')
    start_str="${timerange%-*}"
    end_str="${timerange#*-}"

    start_sec=$(to_seconds "$start_str")
    end_sec=$(to_seconds "$end_str")

    video="${VIDEO_DIR}/case_${case_num}_clipped.mp4"
    out_dir="${SCRIPT_DIR}/vas_inference/case_${case_num}_${start_sec}-${end_sec}"

    echo "================================================================"
    echo "  case ${case_num}  ${start_str}–${end_str}  (${start_sec}s–${end_sec}s)"
    echo "  video  → ${video}"
    echo "  output → ${out_dir}"
    echo "================================================================"

    if [[ ! -f "$video" ]]; then
        echo "WARNING: video not found, skipping: ${video}"
        (( failed++ )) || true
        continue
    fi

    mkdir -p "$out_dir"

    python3 "${SCRIPT_DIR}/infer_prostate_bidir.py" \
        --video         "$video"         \
        --detector_ckpt "$DET_CKPT"     \
        --decoder_ckpt  "$DECODER_CKPT" \
        --output_dir    "$out_dir"       \
        --start_sec     "$start_sec"     \
        --end_sec       "$end_sec"

    echo "Done → ${out_dir}/overlay.mp4"
    echo ""
    (( total++ )) || true

done 3< "$CLIP_LIST"

echo "================================================================"
echo "Finished: ${total} clip(s) processed, ${failed} skipped."
echo "Overlays saved under ${SCRIPT_DIR}/vas_inference/"
echo "================================================================"
