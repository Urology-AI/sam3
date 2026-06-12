#!/usr/bin/env bash
# test_augmentations.sh
# -----------------------------------------------------------------------------
# Visual-only test of the augmentation pipeline in extract_multiclass_features_v2.py.
# Skips SAM2 entirely — samples a few frames, applies augmentation, writes JPEGs.
# Useful for iterating on aug_paste_panel / _crop_to_content / placement priors
# without paying the SAM2 forward-pass cost.
#
# Run inside the singularity container:
#   bash test_augmentations.sh
#
# Optional env vars:
#   CASE_ID   — intuitive case to sample from (default 213)
#   N_FRAMES  — frames to sample (default 6)
#   N_AUG     — augmented variants per frame (default 6)
#   OUT_DIR   — where to dump JPEGs (default aug_test_samples)
#   SEED      — RNG seed for reproducibility (default 42)
# -----------------------------------------------------------------------------
set -euo pipefail

SAM3_DIR=/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
cd "${SAM3_DIR}"

CASE_ID="${CASE_ID:-213}"
N_FRAMES="${N_FRAMES:-6}"
N_AUG="${N_AUG:-6}"
OUT_DIR="${OUT_DIR:-${SAM3_DIR}/aug_test_samples}"
SEED="${SEED:-42}"

CASE_ID="${CASE_ID}" N_FRAMES="${N_FRAMES}" N_AUG="${N_AUG}" OUT_DIR="${OUT_DIR}" SEED="${SEED}" python3 <<'PY'
import os
import random
import shutil
import sys

import cv2
import numpy as np

SAM3_DIR = "/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3"
sys.path.insert(0, SAM3_DIR)
from extract_multiclass_features_v2 import (
    load_panel_library, augment_frame, PANEL_DIR,
)

VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
case_id   = os.environ["CASE_ID"]
n_frames  = int(os.environ["N_FRAMES"])
n_aug     = int(os.environ["N_AUG"])
out_dir   = os.environ["OUT_DIR"]
seed      = int(os.environ["SEED"])

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

video_path = os.path.join(VIDEO_DIR, f"case_{case_id}_clipped.mp4")
if not os.path.exists(video_path):
    raise SystemExit(f"video not found: {video_path}")

panels = load_panel_library(PANEL_DIR)
if not panels:
    raise SystemExit("no panels loaded")
rng = random.Random(seed)

# Dump a contact-sheet of the cropped panels themselves so you can see what
# the pipeline considers "panel content" after _crop_to_content().
panel_dir_out = os.path.join(out_dir, "_cropped_panels")
os.makedirs(panel_dir_out, exist_ok=True)
for i, p in enumerate(panels):
    cv2.imwrite(os.path.join(panel_dir_out, f"panel_{i:02d}_cropped.jpg"), p)
print(f"  Cropped panels → {panel_dir_out}/")

cap   = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_idxs = np.linspace(total * 0.10, total * 0.90, n_frames).astype(int)

# Use a downscaled view for inspection (1080p frames are large)
INSPECT_W = 720

for i, fi in enumerate(sample_idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    scale = INSPECT_W / max(w, h)
    inspect_size = (int(w * scale), int(h * scale))

    cv2.imwrite(os.path.join(out_dir, f"frame{i}_clean.jpg"),
                cv2.resize(frame, inspect_size))
    for k in range(n_aug):
        aug = augment_frame(frame, panels, rng)
        cv2.imwrite(os.path.join(out_dir, f"frame{i}_aug{k}.jpg"),
                    cv2.resize(aug, inspect_size))
cap.release()

print(f"  Augmented samples → {out_dir}/")
print(f"    {n_frames} frames × ({1} clean + {n_aug} augmented) = "
      f"{n_frames * (1 + n_aug)} JPEGs")
print(f"  Look for:")
print(f"    • Panel content (widgets only, no big black canvas) pasted top-right ~75%")
print(f"    • Aspect-natural placement (tall thin strips look right-side, short wide look top-side)")
print(f"    • Other corners occasionally (25%)")
print(f"    • Mix of blur/sharpness/colour/JPEG-quality changes")
PY
