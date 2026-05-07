#!/usr/bin/env bash
# run_fast_inference.sh
# Clip 16:40–18:30 from chahat_edited.mp4, then run the full fast inference pipeline.
#
# Usage:
#   bash run_fast_inference.sh
#
# Run from an interactive GPU node (bsub -Is -q gpu ...).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SOURCE_VIDEO="$SCRIPT_DIR/aua_videos/chahat_edited.mp4"
CLIP_START="00:22:22"
CLIP_END="00:25:19"
CLIP_PATH="$SCRIPT_DIR/aua_videos/chahat_edited_2222_2519.mp4"
OUT_DIR="$SCRIPT_DIR/aua_videos/inferred_videos/chahat_edited_2222_2519"

# ── 1. Clip the source video (fast stream copy, runs on host) ─────────────────
mkdir -p "$OUT_DIR"

if [ ! -f "$CLIP_PATH" ]; then
    echo "[1] Clipping ${CLIP_START} → ${CLIP_END} ..."
    ffmpeg -y -ss "$CLIP_START" -to "$CLIP_END" -i "$SOURCE_VIDEO" -c copy "$CLIP_PATH"
    echo "    Saved: $CLIP_PATH"
else
    echo "[1] Clip already exists, skipping ffmpeg: $CLIP_PATH"
fi

# ── 2. Write a one-shot override that swaps in the clip path + output dir ─────
OVERRIDE="$SCRIPT_DIR/_run_this_clip.py"
cat > "$OVERRIDE" << PYEOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")
import detect_segment_fast as m
m.VIDEO_CLIP  = "$CLIP_PATH"
m.OUTPUT_DIR  = "$OUT_DIR"
m.MAX_CHUNKS  = None   # process the full clip (not just the benchmark 10-chunk cap)
m.main()
PYEOF

# ── 3. Run ───────────────────────────────────────────────────────────────────
echo "[2] Launching fast inference pipeline..."
python3 "$OVERRIDE"

# ── 4. Clean up temp override ─────────────────────────────────────────────────
rm -f "$OVERRIDE"
echo "Done.  Output → $OUT_DIR/overlay.mp4"
