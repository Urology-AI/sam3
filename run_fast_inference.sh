#!/usr/bin/env bash
# run_fast_inference.sh
# Run detect_segment_fast.py (batch pre-encoding + frame-skip + torch.compile)
# inside the SAM3 Singularity container on an interactive GPU node.
#
# Usage:
#   bash run_fast_inference.sh
#
# To request an interactive GPU session first:
#   bsub -Is -q gpu -n 4 -R "rusage[mem=32000,ngpus_physical=1]" -gpu "mode=shared:j_exclusive=no" bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER="/sc/arion/projects/video_rarp/neel_projects/sam3_dev"

singularity exec --nv --writable \
    "$CONTAINER" \
    python3 "$SCRIPT_DIR/detect_segment_fast.py" \
    "$@"
