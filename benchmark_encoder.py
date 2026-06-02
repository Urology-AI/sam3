#!/usr/bin/env python3
"""
benchmark_encoder.py
====================
Measure SAM2 forward_image throughput across {hiera_l, hiera_s} × {fp32, bf16, bf16+compile}.

Pure GPU benchmark — uses a synthetic [B, 3, 1024, 1024] tensor pre-allocated on the
device, no video decode, no preprocessing. Numbers reflect the encoder forward pass
itself, which is what bf16 + torch.compile affect; the end-to-end extraction speedup
depends additionally on I/O and will diverge from these numbers.
"""

import argparse
import gc
import os
import sys
import time

import torch

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
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


def load_sam2(variant, device):
    from sam2.build_sam import build_sam2
    ckpt = os.path.join(SAM2_DIR, VARIANT_TO_CKPT[variant])
    cfg  = VARIANT_TO_CFG[variant]
    model = build_sam2(cfg, ckpt, device=device)
    model.eval()
    return model


@torch.no_grad()
def forward(model, imgs, use_bf16):
    if use_bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.forward_image(imgs)
    return model.forward_image(imgs)


def benchmark_cell(variant, use_bf16, use_compile, batch_size, n_warmup, n_iters, device):
    model = load_sam2(variant, device)
    if use_compile:
        model.forward_image = torch.compile(model.forward_image, mode="default")

    imgs = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)

    for _ in range(n_warmup):
        _ = forward(model, imgs, use_bf16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = forward(model, imgs, use_bf16)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_warmup",   type=int, default=15)
    p.add_argument("--n_iters",    type=int, default=30)
    p.add_argument("--variants",   default="large,small")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}  |  warmup: {args.n_warmup}  |  measured iters: {args.n_iters}")
    print(f"Input shape: [{args.batch_size}, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print()

    configs = [
        ("fp32 baseline",      False, False),
        ("+ bf16 autocast",    True,  False),
        ("+ bf16 + compile",   True,  True),
    ]

    header = f"  {'variant':<6}  {'config':<20}  {'fps':>8}  {'ms/frame':>9}  {'peak GB':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for variant in args.variants.split(","):
        baseline_fps = None
        for label, use_bf16, use_compile in configs:
            fps, ms, peak_mb = benchmark_cell(
                variant, use_bf16, use_compile,
                args.batch_size, args.n_warmup, args.n_iters, device,
            )
            speedup = "" if baseline_fps is None else f"  ({fps/baseline_fps:.2f}× vs fp32)"
            if baseline_fps is None:
                baseline_fps = fps
            print(f"  {variant:<6}  {label:<20}  {fps:>8.2f}  {ms:>9.2f}  {peak_mb/1024:>8.2f}{speedup}")
            results.append((variant, label, fps, ms, peak_mb))
        print()

    # Cross-variant comparison (best-config fps for each variant)
    by_variant = {}
    for variant, label, fps, _, _ in results:
        if label.startswith("+ bf16 + compile"):
            by_variant[variant] = fps
    if "large" in by_variant and "small" in by_variant:
        ratio = by_variant["small"] / by_variant["large"]
        print(f"Best-config Hiera-S vs Hiera-L throughput: {ratio:.2f}×")


if __name__ == "__main__":
    main()
