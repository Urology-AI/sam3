#!/usr/bin/env python3
"""
benchmark_sam2_frames.py
========================

Finds how many actual video frames can be pre-encoded into SAM2's
cached_features dict on a single GPU before OOM.

Tests FRAME_COUNTS in sequence, stopping at the first OOM.
Reports peak allocated GPU memory after each successful run.

Usage:
    python3 aua_tracking/benchmark_sam2_frames.py \
        --video aua_videos/M_05272026090741_U013419052709341_1_001_0005-01.MP4 \
        --start_s 324.5

The start_s should be a timestamp inside a visible-anatomy segment so
frames aren't blank.  Any segment from annotated_external_iliac.json works.
"""

import argparse
import subprocess
import sys
import os
import time
import gc
from typing import List, Optional

import cv2
import numpy as np
import torch

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

ENCODE_BATCH_SIZE = 8

# Frame counts to test (actual video frames, ascending)
FRAME_COUNTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]


# ── ffmpeg loader ──────────────────────────────────────────────────────────────

class FFmpegFrameLoader:
    """Load exactly n_frames actual video frames starting at start_s via ffmpeg."""
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path: str, start_s: float, n_frames: int,
                 image_size: int, video_w: int, video_h: int,
                 sbs_eye_x: int = 0, sbs_eye_w: Optional[int] = None):
        eye_w = sbs_eye_w if sbs_eye_w is not None else video_w
        vf_args = (["-vf", f"crop={sbs_eye_w}:{video_h}:{sbs_eye_x}:0"]
                   if sbs_eye_w is not None else [])
        cmd = (["ffmpeg", "-v", "quiet",
                "-ss", f"{start_s:.6f}",
                "-i", video_path,
                "-frames:v", str(n_frames)]
               + vf_args
               + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        frame_bytes = eye_w * video_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frames = []
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(video_h, eye_w, 3).copy()
            f = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames.append(f)
        proc.stdout.close()
        proc.wait()
        self._frames = frames
        self.n_frames = len(frames)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, i: int) -> torch.Tensor:
        f = self._frames[i]
        t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)
        t -= self._MEAN
        t /= self._STD
        return t


def pre_encode(predictor, state: dict, loader: FFmpegFrameLoader,
               device: torch.device) -> float:
    """Pre-encode all frames. Returns elapsed seconds."""
    n = len(loader)
    t0 = time.perf_counter()
    for b0 in range(0, n, ENCODE_BATCH_SIZE):
        b1 = min(b0 + ENCODE_BATCH_SIZE, n)
        batch = torch.stack([loader[i].to(device).float() for i in range(b0, b1)])
        bb = predictor.forward_image(batch)
        for j, fidx in enumerate(range(b0, b1)):
            state["cached_features"][fidx] = (
                batch[j:j+1].clone(),
                {"backbone_fpn":  [f[j:j+1].clone() for f in bb["backbone_fpn"]],
                 "vision_pos_enc": [p[j:j+1].clone() for p in bb["vision_pos_enc"]]},
            )
        del batch, bb
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def gpu_allocated_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def gpu_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 1024**3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_s", type=float, default=324.5,
                    help="Timestamp in video to start loading frames from")
    ap.add_argument("--sbs_eye", default="left", choices=["left", "right", "none"])
    ap.add_argument("--no_compile", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video: {video_w}x{video_h}  fps={fps:.3f}")

    sbs = args.sbs_eye != "none"
    eye_w = video_w // 2 if sbs else video_w
    eye_x = 0 if (args.sbs_eye == "left" or not sbs) else eye_w

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    from sam2.build_sam import build_sam2_video_predictor
    import tempfile, shutil
    from pathlib import Path

    print("\nLoading SAM2 model...")
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)

    if not args.no_compile:
        print("Compiling...")
        predictor.image_encoder   = torch.compile(predictor.image_encoder,   mode="default")
        predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
        predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")

    model_gb = gpu_allocated_gb()
    print(f"Model loaded: {model_gb:.2f} GB allocated\n")

    # Need a dummy state for cached_features
    scratch = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam2_mask_tracks"
    os.makedirs(scratch, exist_ok=True)

    # Warm up compile with a single frame
    print("Warming up compile with 1 frame...")
    warm_loader = FFmpegFrameLoader(args.video, args.start_s, 1, image_size,
                                    video_w, video_h, eye_x, eye_w if sbs else None)
    tmp = tempfile.mkdtemp(prefix="sam2_bench_", dir=scratch)
    # write a dummy jpg for init_state
    frame0_bgr = cv2.cvtColor(
        (warm_loader[0].permute(1,2,0).numpy() * np.array([0.229,0.224,0.225]) +
         np.array([0.485,0.456,0.406])).clip(0,1).astype(np.float32) * 255,
        cv2.COLOR_RGB2BGR
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), frame0_bgr)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        warm_state = predictor.init_state(video_path=tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        warm_state["images"]      = warm_loader
        warm_state["num_frames"]  = 1
        warm_state["video_height"] = video_h
        warm_state["video_width"]  = eye_w
        pre_encode(predictor, warm_state, warm_loader, device)
        predictor.reset_state(warm_state)
    del warm_loader, warm_state
    torch.cuda.empty_cache()
    gc.collect()

    after_warmup_gb = gpu_allocated_gb()
    print(f"After warmup: {after_warmup_gb:.2f} GB allocated\n")

    print(f"{'Frames':>8}  {'Seconds':>8}  {'Encode(s)':>10}  {'Alloc(GB)':>10}  {'Reserv(GB)':>11}  Result")
    print("-" * 70)

    last_good = 0
    for n_frames in FRAME_COUNTS:
        torch.cuda.empty_cache()
        gc.collect()

        loader = FFmpegFrameLoader(args.video, args.start_s, n_frames, image_size,
                                   video_w, video_h, eye_x, eye_w if sbs else None)
        actual_loaded = len(loader)
        dur_s = actual_loaded / fps

        tmp = tempfile.mkdtemp(prefix="sam2_bench_", dir=scratch)
        frame0_bgr_arr = cv2.VideoCapture(args.video)
        frame0_bgr_arr.set(cv2.CAP_PROP_POS_MSEC, args.start_s * 1000)
        ret, fr = frame0_bgr_arr.read()
        frame0_bgr_arr.release()
        if sbs:
            fr = fr[:, eye_x:eye_x+eye_w]
        cv2.imwrite(os.path.join(tmp, "000000.jpg"), fr if ret else np.zeros((video_h, eye_w, 3), np.uint8))

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=tmp)
                shutil.rmtree(tmp, ignore_errors=True)
                state["images"]       = loader
                state["num_frames"]   = actual_loaded
                state["video_height"] = video_h
                state["video_width"]  = eye_w

                t_enc = pre_encode(predictor, state, loader, device)
                alloc = gpu_allocated_gb()
                resv  = gpu_reserved_gb()
                predictor.reset_state(state)

            print(f"{actual_loaded:>8}  {dur_s:>8.1f}  {t_enc:>10.2f}  {alloc:>10.2f}  {resv:>11.2f}  OK")
            last_good = actual_loaded

        except torch.OutOfMemoryError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{actual_loaded:>8}  {dur_s:>8.1f}  {'':>10}  {'':>10}  {'':>11}  OOM")
            print(f"\n  OOM at {actual_loaded} frames ({dur_s:.1f}s)")
            print(f"  Last successful: {last_good} frames ({last_good/fps:.1f}s)")
            print(f"\n  Recommended MAX_ACTUAL_FRAMES = {int(last_good * 0.85)}"
                  f"  (85% of last good, safety margin)")
            break

        del loader, state
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"\nAll frame counts passed. Max tested: {FRAME_COUNTS[-1]} ({FRAME_COUNTS[-1]/fps:.1f}s)")
        print(f"Recommended MAX_ACTUAL_FRAMES >= {FRAME_COUNTS[-1]}")


if __name__ == "__main__":
    main()
