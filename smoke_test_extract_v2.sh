#!/usr/bin/env bash
# smoke_test_extract_v2.sh
# -----------------------------------------------------------------------------
# Validates extract_multiclass_features_v2.py end-to-end on a single training
# case (K=2 variants, augmentation ON), then inspects the resulting .npz and
# dumps sample augmented frames for visual inspection.
#
# Run inside the singularity container. From the segmentation_prostate/sam3
# directory:
#   bash smoke_test_extract_v2.sh
#
# Optional env vars:
#   CASE_ID    — case to extract (default 213)
#   OUT_DIR    — smoke-test output dir (default multiclass_features_v2_smoke)
#
# What this script does, in order:
#   1. Run extraction on one case (K=2, --use_bf16).
#   2. Open the .npz, verify schema/shape/dtype, print sanity stats, and check
#      that variant 0 (clean) and variant 1 (augmented) actually differ.
#   3. Dump 4 raw frames × 4 augmented variants as JPEGs to inspect HUD-paste
#      placement and visual plausibility of the augmentation stack.
# -----------------------------------------------------------------------------
set -euo pipefail

SAM3_DIR=/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
cd "${SAM3_DIR}"

CASE_ID="${CASE_ID:-213}"
OUT_DIR="${OUT_DIR:-${SAM3_DIR}/multiclass_features_v2_smoke}"

echo "============================================================"
echo "  Smoke test: extract_multiclass_features_v2.py"
echo "  Case:     ${CASE_ID}"
echo "  Out dir:  ${OUT_DIR}"
echo "============================================================"

echo ""
echo "[1/3] Running extraction (K=2, bf16) ..."
python3 extract_multiclass_features_v2.py \
    --mode train \
    --cases "${CASE_ID}" \
    --k_aug 2 \
    --use_bf16 \
    --batch_size 16 \
    --out_dir "${OUT_DIR}"

echo ""
echo "[2/3] Inspecting output .npz ..."
CASE_ID="${CASE_ID}" OUT_DIR="${OUT_DIR}" python3 <<'PY'
import os
import numpy as np

out_dir = os.environ["OUT_DIR"]
case_id = os.environ["CASE_ID"]
path    = os.path.join(out_dir, f"case_{case_id}.npz")

print(f"  File: {path}")
print(f"  Size: {os.path.getsize(path)/1024/1024:.1f} MB")

d = np.load(path, allow_pickle=True)
print(f"  Keys: {list(d.files)}")
for k in d.files:
    a = d[k]
    if isinstance(a, np.ndarray) and a.dtype != object and a.ndim > 0:
        print(f"    {k:<16} shape={str(a.shape):<35} dtype={a.dtype}")
    else:
        print(f"    {k:<16} value={a}")

fpn1    = d["features_fpn1"]
fpn2    = d["features_fpn2"]
labels  = d["labels"]
weights = d["sample_weights"]

# Schema check (option B: 16x16 / 8x8); channel counts detected at runtime
N, K, C1 = fpn1.shape[:3]
_, _, C2 = fpn2.shape[:3]
assert fpn1.shape == (N, K, C1, 16, 16), f"fpn1 shape wrong: {fpn1.shape}"
assert fpn2.shape == (N, K, C2,  8,  8), f"fpn2 shape wrong: {fpn2.shape}"
assert labels.shape  == (N,)
assert weights.shape == (N,)
assert fpn1.dtype == np.float16
assert fpn2.dtype == np.float16
print(f"\n  Schema OK  (N={N}, K={K}, C_fpn1={C1}, C_fpn2={C2})")

# Sanity stats
print(f"\n  fpn1 stats: min={float(fpn1.min()):.3f}  max={float(fpn1.max()):.3f}  "
      f"mean={float(fpn1.mean()):.3f}  any_nan={bool(np.isnan(fpn1).any())}")
print(f"  fpn2 stats: min={float(fpn2.min()):.3f}  max={float(fpn2.max()):.3f}  "
      f"mean={float(fpn2.mean()):.3f}  any_nan={bool(np.isnan(fpn2).any())}")
print(f"  label hist (11 classes): {np.bincount(labels, minlength=11).tolist()}")
print(f"  hard-neg frames        : {int((weights > 1).sum())}")

# Augmentation divergence: variant 0 (clean) vs variant 1 (aug) should differ
diff = np.abs(fpn1[:, 0].astype(np.float32) - fpn1[:, 1].astype(np.float32))
print(f"\n  |variant0 - variant1| on fpn1: mean={diff.mean():.4f}  max={diff.max():.4f}")
if diff.mean() < 1e-4:
    print("  WARNING: augmented and clean variants look nearly identical "
          "— augmentation may be silently no-op")
else:
    print("  Augmented variants differ from clean (augmentation is active)")

# Storage projection at full scale (K=4, 15 cases @ this N), using
# the actual detected channel counts.
bytes_per_variant_per_frame = (C1*16*16 + C2*8*8) * 2  # fp16
full_total_gb = bytes_per_variant_per_frame * 4 * N * 15 / 1024**3
print(f"\n  Storage projection (K=4, 15 cases @ this N): "
      f"{bytes_per_variant_per_frame/1024:.1f} KB/variant/frame  →  "
      f"{full_total_gb:.1f} GB total")
PY

echo ""
echo "[3/3] Dumping sample augmented frames for visual inspection ..."
CASE_ID="${CASE_ID}" OUT_DIR="${OUT_DIR}" python3 <<'PY'
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, "/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3")
from extract_multiclass_features_v2 import (
    load_panel_library, augment_frame, PANEL_DIR,
)

VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
case_id   = os.environ["CASE_ID"]
out_root  = os.environ["OUT_DIR"]
sample_dir = os.path.join(out_root, "aug_samples")
os.makedirs(sample_dir, exist_ok=True)

video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
if not os.path.exists(video_path):
    print(f"  video not found: {video_path}")
    sys.exit(1)

panels = load_panel_library(PANEL_DIR)
rng = random.Random(42)

cap   = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 4 evenly-spaced frames across the video
sample_idxs = np.linspace(total * 0.1, total * 0.9, 4).astype(int)

for i, fi in enumerate(sample_idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    # Down-scale for inspection (1080p frames are big)
    scale = 640.0 / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    cv2.imwrite(os.path.join(sample_dir, f"sample_{i}_clean.jpg"),
                cv2.resize(frame, new_size))
    for k in range(4):
        aug = augment_frame(frame, panels, rng)
        cv2.imwrite(os.path.join(sample_dir, f"sample_{i}_aug{k}.jpg"),
                    cv2.resize(aug, new_size))
cap.release()
print(f"  Sample frames → {sample_dir}/")
print(f"  Inspect: 4 clean + 16 augmented JPEGs.")
print(f"  Visual checks to look for:")
print(f"    • HUD panel pasted top-right ~75% of the time (other corners ~25%)")
print(f"    • Panel covers 18-40% of frame width, aspect preserved")
print(f"    • Occasional letterbox/pillarbox bars")
print(f"    • Some frames visibly blurred, some sharper (unsharp), some")
print(f"      colour-shifted, some pixelated (downscale-upscale or low JPEG q)")
PY

echo ""
echo "============================================================"
echo "  Smoke test complete."
echo "    .npz       → ${OUT_DIR}/case_${CASE_ID}.npz"
echo "    samples    → ${OUT_DIR}/aug_samples/"
echo "============================================================"
