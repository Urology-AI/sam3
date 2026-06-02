#!/usr/bin/env python3
"""
benchmark.py
============
Throughput benchmarks for the endobag localiser. Two sections:

  Section 1 — Encoder forward (synthetic input, no I/O)
      {hiera_l, hiera_s} × {fp32, bf16, bf16 + compile}
      Validates raw GPU encoder speed. Output: fps + peak VRAM + speedup over fp32.

  Section 2 — End-to-end pipeline (decode + preprocess + encoder, real video)
      Hiera-S + bf16 + compile fixed as the encoder.
      I/O methods compared (random-access sample pattern):
        - cv2 single-threaded            (current production path)
        - cv2 + DataLoader(num_workers)  (CPU decode parallel to GPU compute)
        - TorchCodec CPU                 (libav-based, ML-oriented decoder)
        - TorchCodec NVDEC               (frames decoded directly on the H100)

      All four exercise the same frame index list and the same encoder model
      — only the decode + preprocess + H2D path differs.

Usage
-----
  python3 benchmark.py
  python3 benchmark.py --skip_encoder              # only Section 2
  python3 benchmark.py --skip_pipeline             # only Section 1
  python3 benchmark.py --video_path /path/to.mp4   # override default test video
  python3 benchmark.py --num_frames 500 --stride 120

TorchCodec is optional: if `pip install torchcodec` hasn't been run, Section 2
skips those rows and prints a hint.
"""

import argparse
import gc
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
VIDEO_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

VARIANT_TO_CKPT = {
    "large": "checkpoints/sam2.1_hiera_large.pt",
    "small": "checkpoints/sam2.1_hiera_small.pt",
}
VARIANT_TO_CFG = {
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
}

IMG_SIZE = 1024
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

try:
    from torchcodec.decoders import VideoDecoder as TCVideoDecoder  # noqa: F401
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


# ─── Model loading + forward ──────────────────────────────────────────────────

def load_sam2(variant, device):
    from sam2.build_sam import build_sam2
    ckpt = os.path.join(SAM2_DIR, VARIANT_TO_CKPT[variant])
    cfg  = VARIANT_TO_CFG[variant]
    model = build_sam2(cfg, ckpt, device=device)
    model.eval()
    return model


@torch.no_grad()
def encoder_forward(model, imgs, use_bf16=True):
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_image(imgs)
    return model.forward_image(imgs)


# ─── Preprocessing (shared) ───────────────────────────────────────────────────

def preprocess_bgr(frame_bgr):
    """cv2 BGR uint8 frame → normalized [3, H, W] float tensor (on CPU)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Encoder forward
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_encoder_cell(variant, use_bf16, use_compile,
                            batch_size, n_warmup, n_iters, device):
    model = load_sam2(variant, device)
    if use_compile:
        model.forward_image = torch.compile(model.forward_image, mode="default")

    imgs = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)

    for _ in range(n_warmup):
        _ = encoder_forward(model, imgs, use_bf16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = encoder_forward(model, imgs, use_bf16)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()

    total_time   = t1 - t0
    total_frames = n_iters * batch_size
    fps          = total_frames / total_time
    ms_per_frame = total_time / total_frames * 1000

    del model, imgs
    gc.collect()
    torch.cuda.empty_cache()

    return fps, ms_per_frame, peak_mb


def run_encoder_section(args, device):
    print("=" * 72)
    print("Section 1 — Encoder forward  (synthetic input, no I/O)")
    print("=" * 72)
    print(f"Batch: {args.batch_size}  |  warmup: {args.n_warmup}  |  measured iters: {args.n_iters}")
    print(f"Input: [{args.batch_size}, 3, {IMG_SIZE}, {IMG_SIZE}]\n")

    header = f"  {'variant':<6}  {'config':<20}  {'fps':>8}  {'ms/frame':>9}  {'peak GB':>8}  speedup"
    print(header)
    print("  " + "-" * (len(header) - 2))

    configs = [
        ("fp32 baseline",    False, False),
        ("+ bf16 autocast",  True,  False),
        ("+ bf16 + compile", True,  True),
    ]

    for variant in ["large", "small"]:
        baseline_fps = None
        for label, use_bf16, use_compile in configs:
            fps, ms, peak_mb = benchmark_encoder_cell(
                variant, use_bf16, use_compile,
                args.batch_size, args.n_warmup, args.n_iters, device,
            )
            spdup = "" if baseline_fps is None else f"  ({fps/baseline_fps:.2f}× vs fp32)"
            if baseline_fps is None:
                baseline_fps = fps
            print(f"  {variant:<6}  {label:<20}  {fps:>8.2f}  {ms:>9.2f}  {peak_mb/1024:>8.2f}{spdup}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


# --- Method 1: cv2 single-threaded (matches extract_endobag_features.py) ----

def bench_cv2_serial(video_path, indices, model, batch_size, device):
    cap = cv2.VideoCapture(video_path)
    batch    = []
    n_frames = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        batch.append(preprocess_bgr(frame_bgr))
        if len(batch) == batch_size:
            imgs = torch.stack(batch).to(device, non_blocking=True)
            _ = encoder_forward(model, imgs, use_bf16=True)
            n_frames += batch_size
            batch = []

    if batch:
        imgs = torch.stack(batch).to(device, non_blocking=True)
        _ = encoder_forward(model, imgs, use_bf16=True)
        n_frames += len(batch)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cap.release()
    return n_frames / (t1 - t0)


# --- Method 2: cv2 + DataLoader(num_workers) ---------------------------------

class CV2RandomDataset(Dataset):
    def __init__(self, video_path, indices):
        self.video_path = video_path
        self.indices    = indices
        self._cap       = None  # lazily opened per-worker

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.indices[i]))
        ret, frame_bgr = self._cap.read()
        if not ret:
            frame_bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return preprocess_bgr(frame_bgr)


def bench_cv2_dataloader(video_path, indices, model, batch_size, num_workers, device):
    ds = CV2RandomDataset(video_path, indices)
    loader = DataLoader(
        ds,
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = True,
        prefetch_factor = 2 if num_workers > 0 else None,
        shuffle         = False,
    )
    n_frames = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        _ = encoder_forward(model, batch, use_bf16=True)
        n_frames += batch.size(0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_frames / (t1 - t0)


# --- Method 3 & 4: TorchCodec CPU / NVDEC ------------------------------------

def bench_torchcodec(video_path, indices, model, batch_size, decode_device, device):
    """
    decode_device: "cpu" → libav CPU decode, frames on CPU then copied to GPU.
                   "cuda:0" → NVDEC, frames decoded directly to GPU memory.
    """
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(source=video_path, device=decode_device)

    n_frames     = 0
    indices_list = [int(i) for i in indices]
    mean_dev     = IMG_MEAN.to(device).view(1, 3, 1, 1)
    std_dev      = IMG_STD.to(device).view(1, 3, 1, 1)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, len(indices_list), batch_size):
        chunk = indices_list[start:start + batch_size]
        fb     = decoder.get_frames_at(indices=chunk)
        frames = fb.data  # [N, C, H, W] uint8 on decode_device
        if frames.device != device:
            frames = frames.to(device, non_blocking=True)
        frames = frames.float() / 255.0
        frames = torch.nn.functional.interpolate(
            frames, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        )
        frames = (frames - mean_dev) / std_dev
        _ = encoder_forward(model, frames, use_bf16=True)
        n_frames += len(chunk)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n_frames / (t1 - t0)


# --- Pipeline orchestration --------------------------------------------------

def run_pipeline_section(args, device):
    print("=" * 72)
    print("Section 2 — End-to-end pipeline  (decode + preprocess + encoder)")
    print("=" * 72)

    if not os.path.exists(args.video_path):
        print(f"Video not found: {args.video_path}")
        print("Pass --video_path /path/to/video.mp4 to override.\n")
        return

    fps_src, total = get_video_info(args.video_path)
    n_target = min(args.num_frames, max(1, total // args.stride))
    indices  = np.array([i * args.stride for i in range(n_target)], dtype=np.int64)

    print(f"Video : {args.video_path}")
    print(f"        fps={fps_src:.1f}  total_frames={total}")
    print(f"Sample: {n_target} frames at stride {args.stride} "
          f"(range 0–{indices[-1]} = {indices[-1]/fps_src:.0f}s)")
    print(f"Encoder: Hiera-S + bf16 + compile  |  batch={args.batch_size}\n")

    # ── Single shared encoder (Hiera-S + bf16 + compile) ──
    print("Loading + warming encoder (compile graph capture, ~30s)...")
    model = load_sam2("small", device)
    model.forward_image = torch.compile(model.forward_image, mode="default")
    warmup_imgs = torch.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(args.n_warmup):
        _ = encoder_forward(model, warmup_imgs, use_bf16=True)
    torch.cuda.synchronize()
    del warmup_imgs
    torch.cuda.empty_cache()
    print("  encoder ready.\n")

    print(f"  {'method':<32}  {'fps':>8}  speedup")
    print("  " + "-" * 56)

    # Baseline: cv2 single-threaded random seek
    fps_serial = bench_cv2_serial(args.video_path, indices, model, args.batch_size, device)
    print(f"  {'cv2 serial (current)':<32}  {fps_serial:>8.2f}  1.00×")

    # DataLoader sweep
    for nw in args.num_workers_sweep:
        fps_dl = bench_cv2_dataloader(
            args.video_path, indices, model, args.batch_size, nw, device,
        )
        print(f"  {f'cv2 + DataLoader(nw={nw})':<32}  {fps_dl:>8.2f}  {fps_dl/fps_serial:.2f}×")

    # TorchCodec CPU + NVDEC
    if HAS_TORCHCODEC:
        try:
            fps_tc_cpu = bench_torchcodec(
                args.video_path, indices, model, args.batch_size, "cpu", device,
            )
            print(f"  {'TorchCodec (CPU decode)':<32}  {fps_tc_cpu:>8.2f}  {fps_tc_cpu/fps_serial:.2f}×")
        except Exception as e:
            print(f"  TorchCodec CPU FAILED: {e}")

        try:
            fps_tc_gpu = bench_torchcodec(
                args.video_path, indices, model, args.batch_size, "cuda:0", device,
            )
            print(f"  {'TorchCodec (NVDEC, GPU)':<32}  {fps_tc_gpu:>8.2f}  {fps_tc_gpu/fps_serial:.2f}×")
        except Exception as e:
            print(f"  TorchCodec NVDEC FAILED: {e}")
    else:
        print(f"  {'TorchCodec':<32}  (not installed — pip install torchcodec)")

    print()


# ──────────────────────────────────────────────────────────────────────────────

def parse_workers_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    # Section 1
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_warmup",   type=int, default=15)
    p.add_argument("--n_iters",    type=int, default=30)
    # Section 2
    p.add_argument("--video_path", default=os.path.join(VIDEO_DIR, "case_213_clipped.mp4"))
    p.add_argument("--num_frames", type=int, default=200,
                   help="Number of frames to sample for pipeline benchmark")
    p.add_argument("--stride",     type=int, default=120,
                   help="Frame stride (60 fps video × 0.5 fps sample = stride 120)")
    p.add_argument("--num_workers_sweep", type=parse_workers_list, default=[0, 2, 4, 8],
                   help="Comma-sep list of num_workers values to test (0 = same process)")
    # Section toggles
    p.add_argument("--skip_encoder",  action="store_true")
    p.add_argument("--skip_pipeline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TorchCodec available: {HAS_TORCHCODEC}\n")

    if not args.skip_encoder:
        run_encoder_section(args, device)

    if not args.skip_pipeline:
        run_pipeline_section(args, device)


if __name__ == "__main__":
    main()
