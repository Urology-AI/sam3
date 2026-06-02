#!/usr/bin/env bash
# Benchmark SAM2 forward_image across {large, small} × {fp32, bf16, bf16+compile}.
# Pure GPU forward — no video I/O, no preprocessing. Validates the bf16/compile
# speedup claims and the Hiera-S vs Hiera-L ratio.
#
# Usage:
#   bash run_encoder_benchmark.sh                # default batch 8
#   BATCH_SIZE=32 bash run_encoder_benchmark.sh  # check throughput at larger batch
set -euo pipefail

cd "$(dirname "$0")"

BATCH_SIZE="${BATCH_SIZE:-8}"
N_WARMUP="${N_WARMUP:-15}"
N_ITERS="${N_ITERS:-30}"

python3 benchmark_encoder.py \
    --batch_size "$BATCH_SIZE" \
    --n_warmup   "$N_WARMUP" \
    --n_iters    "$N_ITERS"
