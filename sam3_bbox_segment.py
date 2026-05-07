#!/usr/bin/env python3
"""
SAM3 Video Segmentation — Box Prompt via SAM2-style Tracker API
================================================================
Uses Sam3TrackerPredictor (the SAM2-inherited tracker) which has
reliable box prompt support via add_new_points_or_box().

This is the same API shown in:
  examples/sam3_for_sam2_video_task_example.ipynb

The Sam3VideoPredictor's handle_request(bounding_boxes=...) is buggy
for box-only prompts (GitHub issues #193, #204). This script bypasses
that entirely by using the tracker directly.

Memory management:
  VideoFileStreamingLoader + init_state_streaming keep RAM O(1) for
  any video length. offload_state_to_cpu=True keeps GPU usage flat.
  Masks written to disk per-frame; no large dict held in RAM.

Coordinate handling:
  SAM3's add_new_points_or_box with rel_coordinates=True (default)
  expects box in [0,1] normalised range and multiplies by image_size
  (1024) internally. We divide the video-pixel box by [W, H, W, H]
  before passing it — matching how SAM2 handles this internally.
  Using rel_coordinates=False with raw video-pixel coords feeds them
  straight to the prompt encoder in the wrong space → garbage masks.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np
import torch
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2/test_videos/case_213_clipped_27:30.mp4"
OUTPUT_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2/test_videos/case_213_clipped_27:30_sam3_output"

SAM3_CKPT = "/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"

# Scratch space for the 1-frame temp dir used by init_state_streaming.
# Never use /tmp on Minerva/Arion — that resolves to the login node.
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"

# Bounding box in ABSOLUTE pixel coordinates (xyxy format, video resolution)
BOX_XYXY = np.array([509, 217, 863, 711], dtype=np.float32)

MASK_COLOR = (0, 255, 0)
MASK_ALPHA = 0.20

# Memory guards
GPU_MEM_LIMIT_FRACTION = 0.85
CPU_MEM_MIN_GB         = 8.0
CPU_CHECK_EVERY        = 50

# ── Setup ─────────────────────────────────────────────────────────────────────
assert os.path.isfile(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
assert os.path.isfile(SAM3_CKPT),  f"SAM3 checkpoint not found: {SAM3_CKPT}"
assert torch.cuda.is_available(), "CUDA not available"

print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_TMP, exist_ok=True)

# ── Memory monitoring helpers ─────────────────────────────────────────────────

def _cpu_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


def cpu_check(tag: str = "") -> None:
    avail = _cpu_mem_available_gb()
    label = f" [{tag}]" if tag else ""
    print(f"  CPU RAM{label}: {avail:.1f} GB available")
    if avail < CPU_MEM_MIN_GB:
        print(f"\n[FATAL] CPU RAM guard triggered{label}\n"
              f"  Only {avail:.1f} GB available, threshold {CPU_MEM_MIN_GB:.1f} GB.\n"
              f"  Exiting cleanly.", flush=True)
        sys.exit(1)


def gpu_check(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    limit = total * GPU_MEM_LIMIT_FRACTION
    label = f" [{tag}]" if tag else ""
    print(f"  GPU mem{label}: {alloc:.2f} / {total:.1f} GB  "
          f"({alloc/total*100:.1f}%  limit {GPU_MEM_LIMIT_FRACTION*100:.0f}%)")
    if alloc > limit:
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(0) / 1e9
        if alloc > limit:
            print(f"\n[FATAL] GPU guard triggered{label}\n"
                  f"  {alloc:.2f} GB > {limit:.2f} GB limit. Exiting cleanly.", flush=True)
            sys.exit(1)


# ── Streaming frame loader ────────────────────────────────────────────────────

class VideoFileStreamingLoader:
    """O(1)-RAM frame loader — reads one frame per __getitem__, discards it."""

    _MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    @staticmethod
    def _count_frames(cap):
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported <= 0:
            return 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, reported - 1)
        ret, _ = cap.read()
        if ret:
            return reported
        lo, hi = 0, reported - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, _ = cap.read()
            lo = mid if ret else lo
            hi = mid - 1 if not ret else hi
        return lo + 1

    def __init__(self, video_path: str, image_size: int, step: int = 1):
        self.video_path = video_path
        self.image_size = image_size
        self.step       = step
        probe = cv2.VideoCapture(video_path)
        if not probe.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.src_fps      = probe.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_width  = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_raw             = self._count_frames(probe)
        probe.release()
        reported = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        if n_raw != reported:
            print(f"  [VideoLoader] CAP_PROP_FRAME_COUNT reported {reported} "
                  f"but only {n_raw} readable — using {n_raw}.")
        self.n_total  = n_raw
        self.n_frames = (n_raw + step - 1) // step
        self._cap      = cv2.VideoCapture(video_path)
        self._next_raw = 0

    def __len__(self):
        return self.n_frames

    def __getitem__(self, logical_idx: int) -> torch.Tensor:
        raw_idx = logical_idx * self.step
        if raw_idx < self._next_raw:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
            self._next_raw = raw_idx
        elif raw_idx > self._next_raw:
            for _ in range(raw_idx - self._next_raw):
                self._cap.grab()
            self._next_raw = raw_idx
        ret, frame = self._cap.read()
        self._next_raw += 1
        if not ret:
            raise IndexError(f"Frame {raw_idx} not readable from {self.video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        img -= self._MEAN
        img /= self._STD
        return img

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


# ── Streaming init_state ──────────────────────────────────────────────────────

def init_state_streaming(predictor, video_path: str):
    """
    Bootstrap init_state on a 1-frame temp dir, then swap state["images"]
    with a VideoFileStreamingLoader to avoid the N-frame pre-allocation.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame 0 from {video_path}")

    init_dir = tempfile.mkdtemp(prefix="sam3_init1_", dir=SCRATCH_TMP)
    try:
        cv2.imwrite(os.path.join(init_dir, "000000.jpg"), frame0)
        state = predictor.init_state(
            video_path=init_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
    finally:
        shutil.rmtree(init_dir, ignore_errors=True)

    image_size = getattr(predictor, "image_size", 1024)
    loader = VideoFileStreamingLoader(video_path, image_size)
    state["images"]     = loader
    state["num_frames"] = loader.n_frames
    return state, loader.n_frames


# ── GPU + CPU guard around propagation ───────────────────────────────────────

def propagate_with_guard(predictor, state, start_frame_idx=0,
                         max_frame_num_to_track=None, reverse=False):
    if max_frame_num_to_track is None:
        max_frame_num_to_track = state["num_frames"]
    gpu_available = torch.cuda.is_available()
    total_mem     = (torch.cuda.get_device_properties(0).total_memory
                     if gpu_available else 0)
    gpu_limit     = total_mem * GPU_MEM_LIMIT_FRACTION

    for frame_num, result in enumerate(
        predictor.propagate_in_video(
            state, start_frame_idx, max_frame_num_to_track, reverse,
            propagate_preflight=True,
        ), 1
    ):
        if gpu_available:
            alloc = torch.cuda.memory_allocated(0)
            if alloc > gpu_limit:
                torch.cuda.empty_cache()
                if torch.cuda.memory_allocated(0) > gpu_limit:
                    print(f"\n[FATAL] GPU guard at frame {frame_num}. Exiting.", flush=True)
                    sys.exit(1)
        if frame_num % CPU_CHECK_EVERY == 0:
            if _cpu_mem_available_gb() < CPU_MEM_MIN_GB:
                print(f"\n[FATAL] CPU RAM guard at frame {frame_num}. Exiting.", flush=True)
                sys.exit(1)
        yield result


# ── Read video metadata ───────────────────────────────────────────────────────
print(f"\n[1/6] Reading video metadata...")
t0 = time.time()
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Normalise box to [0,1] — SAM3's rel_coordinates=True (default) multiplies
# by image_size (1024) to reach 1024x1024 internal space. Raw video-pixel
# coords fed with rel_coordinates=False go into the prompt encoder as-is
# at the wrong scale and produce garbage masks.
BOX_NORM = BOX_XYXY / np.array([w_vid, h_vid, w_vid, h_vid], dtype=np.float32)

print(f"  {frame_count} frames, {w_vid}x{h_vid} @ {fps:.1f} fps")
print(f"  Box (abs xyxy):  {BOX_XYXY.tolist()}")
print(f"  Box (norm 0-1):  {BOX_NORM.tolist()}")

# ── Build SAM3 tracker predictor ────────────────────────────────────────────
print(f"\n[2/6] Building SAM3 tracker predictor...")
t0 = time.time()

from sam3.model_builder import build_tracker

predictor = build_tracker(apply_temporal_disambiguation=True, with_backbone=True)

print(f"  Loading weights from checkpoint...")
ckpt = torch.load(SAM3_CKPT, map_location="cpu", weights_only=True)
ckpt = ckpt.get("model", ckpt)

# tracker.* → strip prefix for the mask decoder / prompt encoder / transformer
tracker_state = {k[len("tracker."):]: v for k, v in ckpt.items() if k.startswith("tracker.")}

# detector.backbone.* → backbone.* for the image encoder (backbone was stored
# under the detector namespace; the predictor expects it under "backbone.")
backbone_state = {k[len("detector."):]: v for k, v in ckpt.items()
                  if k.startswith("detector.backbone.")}

combined_state = {**tracker_state, **backbone_state}
missing, unexpected = predictor.load_state_dict(combined_state, strict=False)
if missing:
    print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
print(f"  Loaded {len(tracker_state)} tracker + {len(backbone_state)} backbone keys")
del ckpt, tracker_state, backbone_state, combined_state

predictor = predictor.to("cuda", dtype=torch.bfloat16)
predictor.eval()  # sets self.training=False, skips all training-only branches
predictor.teacher_force_obj_scores_for_mem = False
print(f"  Done in {time.time()-t0:.1f}s")

gpu_check("after model load")
cpu_check("after model load")

# ── Initialize inference state (streaming, O(1) RAM) ─────────────────────────
print(f"\n[3/6] Initializing inference state (streaming)...")
t0 = time.time()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state, n_frames = init_state_streaming(predictor, VIDEO_PATH)

print(f"  State initialized in {time.time()-t0:.1f}s  ({n_frames} frames)")
gpu_check("after init_state")
cpu_check("after init_state")

# ── Add box prompt on frame 0 ────────────────────────────────────────────────
print(f"\n[4/6] Adding box prompt on frame 0...")
t0 = time.time()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # BOX_NORM is [0,1] — rel_coordinates=True (default) scales to 1024x1024 internally
    frame_idx, object_ids, _low_res, video_res_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=BOX_NORM,
    )

print(f"  Box prompt added in {time.time()-t0:.1f}s")
print(f"  frame_idx={frame_idx}, object_ids={object_ids}")
print(f"  video_res_masks shape={video_res_masks.shape}, dtype={video_res_masks.dtype}")

mask_frame0 = (video_res_masks[0] > 0.0).cpu().numpy().squeeze()
print(f"  Frame 0 mask: shape={mask_frame0.shape}, pixels={mask_frame0.sum()}, "
      f"coverage={mask_frame0.mean()*100:.1f}%")

# ── Propagate — write masks to disk frame-by-frame ───────────────────────────
print(f"\n[5/6] Propagating segmentation through video...")
t0 = time.time()

masks_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(masks_dir, exist_ok=True)
mask_count = 0

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # SAM3 yields 5 values: frame_idx, obj_ids, low_res, video_res, obj_scores
    for out_frame_idx, _obj_ids, _low_res, out_video_masks, _scores in propagate_with_guard(
        predictor, inference_state
    ):
        mask = (out_video_masks[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
        if mask.any():
            np.save(os.path.join(masks_dir, f"{out_frame_idx:06d}.npy"), mask)
            mask_count += 1
        if out_frame_idx < 3 or out_frame_idx % 50 == 0:
            print(f"    Frame {out_frame_idx}: pixels={int(mask.sum())}, "
                  f"coverage={mask.mean()*100:.1f}%")

print(f"  Propagated — saved {mask_count} masks in {time.time()-t0:.1f}s")
gpu_check("after propagation")
cpu_check("after propagation")

# ── Render overlay video ───────────────────────────────────────────────────────
print(f"\n[6/6] Rendering output video...")
t0 = time.time()

out_video_path = os.path.join(OUTPUT_DIR, "overlay.mp4")
writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid))

cap = cv2.VideoCapture(VIDEO_PATH)
fidx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask_path = os.path.join(masks_dir, f"{fidx:06d}.npy")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        if mask.shape[:2] != (h_vid, w_vid):
            mask = cv2.resize(mask, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
        if mask.any():
            overlay = frame.copy()
            overlay[mask.astype(bool)] = MASK_COLOR
            frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    if fidx == 0:
        x1, y1, x2, y2 = BOX_XYXY.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "bbox prompt", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    writer.write(frame)
    fidx += 1

cap.release()
writer.release()

print(f"  Overlay: {out_video_path}")
print(f"  Masks:   {masks_dir}/ ({mask_count} files)")
print(f"  Render done in {time.time()-t0:.1f}s")

# ── Cleanup ───────────────────────────────────────────────────────────────────
try:
    predictor.reset_state(inference_state)
except Exception as e:
    print(f"  reset_state: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  Input:        {VIDEO_PATH}")
print(f"  Output:       {OUTPUT_DIR}")
print(f"  Masks saved:  {mask_count} / {frame_count} frames")
print(f"  Box (xyxy):   {BOX_XYXY.tolist()}")
print(f"  GPU mem:      {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("=" * 60)
