#!/usr/bin/env python3
"""
rerender_overlay.py
Re-render an overlay video from pre-saved .npy masks with a new alpha.

Usage:
    python3 rerender_overlay.py \
        --video   aua_videos/chahat_edited_1640_1830.mp4 \
        --masks   aua_videos/inferred_videos/chahat_edited_1640_1830/masks \
        --output  aua_videos/inferred_videos/chahat_edited_1640_1830/overlay_a15.mp4 \
        --alpha   0.15
"""

import argparse, bisect, os, subprocess
import cv2
import numpy as np


MASK_COLOR_BGR = (0, 255, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--masks",  required=True, help="directory of .npy mask files")
    ap.add_argument("--output", required=True, help="path for output .mp4")
    ap.add_argument("--alpha",  type=float, default=0.35)
    args = ap.parse_args()

    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Load keyframe masks
    keyframe_masks = {}
    for fname in sorted(os.listdir(args.masks)):
        if fname.endswith(".npy"):
            fidx = int(os.path.splitext(fname)[0])
            m = np.load(os.path.join(args.masks, fname)).astype(np.float32)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            keyframe_masks[fidx] = m
    kf_idxs = sorted(keyframe_masks.keys())
    print(f"{len(kf_idxs)} keyframe masks  |  alpha={args.alpha}  |  {n} frames @ {fps:.1f} fps")

    def get_mask_at(fidx):
        if not kf_idxs:
            return None
        pos = bisect.bisect_left(kf_idxs, fidx)
        if pos == len(kf_idxs):
            return keyframe_masks[kf_idxs[-1]]
        if pos == 0 or kf_idxs[pos] == fidx:
            return keyframe_masks[kf_idxs[pos]]
        prev_idx, next_idx = kf_idxs[pos - 1], kf_idxs[pos]
        a = (fidx - prev_idx) / (next_idx - prev_idx)
        return (1.0 - a) * keyframe_masks[prev_idx] + a * keyframe_masks[next_idx]

    out_raw = args.output.replace(".mp4", "_raw.mp4")
    writer  = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap     = cv2.VideoCapture(args.video)
    color_layer = np.full((h, w, 3), MASK_COLOR_BGR, dtype=np.float32)

    for fidx in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        mask_f = get_mask_at(fidx)
        if mask_f is not None:
            alpha_map = (mask_f * args.alpha)[:, :, None]
            frame = (frame * (1.0 - alpha_map) + color_layer * alpha_map).astype(np.uint8)
        writer.write(frame)
        if fidx % 600 == 0:
            print(f"  frame {fidx}/{n}", flush=True)

    cap.release()
    writer.release()

    print("Re-muxing to H.264...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[:400])
    os.remove(out_raw)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
