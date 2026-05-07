#!/usr/bin/env python3
"""
Render SAM3 mask overlays to video.

Reads binary .npy masks produced by sam3_bbox_segment.py and composites
them onto the original video frames, writing the result to OUTPUT_DIR.
"""

import os
import numpy as np
import cv2

VIDEO_PATH = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2/test_videos/case_213_clipped_27:30.mp4"
MASKS_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2/test_videos/case_213_clipped_27:30_sam3_output/masks"
OUTPUT_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2/prostate_inference_overlay"

MASK_COLOR = (0, 255, 0)
MASK_ALPHA = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w_vid}x{h_vid} @ {fps:.1f} fps")

out_path = os.path.join(OUTPUT_DIR, "case_213_clipped_27:30_sam3_overlay.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid))

fidx = 0
written = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask_path = os.path.join(MASKS_DIR, f"{fidx:06d}.npy")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        if mask.shape[:2] != (h_vid, w_vid):
            mask = cv2.resize(mask, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
        if mask.any():
            overlay = frame.copy()
            overlay[mask.astype(bool)] = MASK_COLOR
            frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

    writer.write(frame)
    fidx += 1
    written += 1
    if fidx % 100 == 0:
        print(f"  {fidx} frames written...")

cap.release()
writer.release()
print(f"Done — {written} frames written to {out_path}")
