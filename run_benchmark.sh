#!/bin/bash
# Run benchmark_sam2large.py inside the SAM3 Singularity container.
# Replace <container_path> with your SAM3 container, then run:
#   bash run_benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/benchmark_sam2large.py"
