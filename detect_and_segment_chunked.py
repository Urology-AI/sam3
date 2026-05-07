#!/usr/bin/env python3
"""
detect_and_segment_chunked.py
==============================
Phase 1 chunked reinit: re-runs the SAM3 detector at the start of every
K frames, resets the tracker memory, then propagates through the chunk.
Handles long-video drift by periodically anchoring the track to a fresh
text-conditioned detection.

Usage
-----
  python3 detect_and_segment_chunked.py \\
      --video /path/to/video.mp4 \\
      --detector_ckpt detector_training/checkpoints/<ts>/checkpoint_best.pth \\
      --reinit_interval 150          # frames between reinits (~5 s at 30 fps)
      --clip_start 4888 --clip_end 4958
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAM3_DIR)

SAM3_CKPT = ("/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
             "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")
BPE_PATH  = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
SCRATCH_TMP       = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Memory helpers ─────────────────────────────────────────────────────────────

def _cpu_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


def cpu_check(tag=""):
    avail = _cpu_mem_available_gb()
    label = f" [{tag}]" if tag else ""
    print(f"  CPU RAM{label}: {avail:.1f} GB available")
    if avail < 8.0:
        print(f"[FATAL] CPU RAM guard{label}: {avail:.1f} GB < 8 GB. Exiting.")
        sys.exit(1)


def gpu_check(tag=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    label = f" [{tag}]" if tag else ""
    print(f"  GPU mem{label}: {alloc:.2f} / {total:.1f} GB ({alloc/total*100:.1f}%)")
    if alloc > total * 0.85:
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(0) / 1e9 > total * 0.85:
            print(f"[FATAL] GPU guard{label}. Exiting.")
            sys.exit(1)


# ── Streaming video loader ─────────────────────────────────────────────────────

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


def init_state_streaming(predictor, video_path: str):
    """
    Init tracker state using a 1-frame temp dir, then replace the image
    store with the full-video streaming loader. Returns (state, loader).
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
    return state, loader


def reset_tracker_state(inference_state):
    """Clear all per-frame tracking memory so the next chunk starts fresh.

    Sam3TrackerPredictor has no reset_state() method; we replicate its effect
    by clearing every mutable tracking field while keeping the static fields
    (images, dimensions, device, offload flags).
    """
    from collections import OrderedDict
    inference_state["output_dict"]["cond_frame_outputs"].clear()
    inference_state["output_dict"]["non_cond_frame_outputs"].clear()
    inference_state["first_ann_frame_idx"] = None
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
    inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
    inference_state["cached_features"].clear()
    inference_state["constants"].clear()
    # Reset object registrations so add_new_points_or_box can re-register obj_id=1
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}


def propagate_with_guard(predictor, state, start_frame_idx=0,
                         max_frame_num_to_track=None, reverse=False):
    if max_frame_num_to_track is None:
        max_frame_num_to_track = state["num_frames"]
    total_mem = (torch.cuda.get_device_properties(0).total_memory
                 if torch.cuda.is_available() else 0)
    gpu_limit = total_mem * 0.85

    for frame_num, result in enumerate(
        predictor.propagate_in_video(
            state, start_frame_idx, max_frame_num_to_track, reverse,
            propagate_preflight=True,
        ), 1
    ):
        if torch.cuda.is_available() and torch.cuda.memory_allocated(0) > gpu_limit:
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated(0) > gpu_limit:
                print(f"[FATAL] GPU guard at frame {frame_num}. Exiting.", flush=True)
                sys.exit(1)
        if frame_num % 50 == 0 and _cpu_mem_available_gb() < 8.0:
            print(f"[FATAL] CPU RAM guard at frame {frame_num}. Exiting.", flush=True)
            sys.exit(1)
        yield result


# ── Clip ──────────────────────────────────────────────────────────────────────

def clip_video(src: str, start_s: float, end_s: float, out_path: str) -> str:
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", src,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        out_path,
    ]
    print(f"  ffmpeg clip: {start_s}s – {end_s}s  ({duration:.1f}s)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    return out_path


# ── Detector: load once, call per chunk ───────────────────────────────────────

def load_detector(detector_ckpt: str, device: torch.device):
    """
    Build and load the fine-tuned SAM3 detector. Pre-encodes the text query.
    Returns (model, lang_feats, lang_mask) — keep these alive for all chunks.
    """
    from sam3.model_builder import build_sam3_image_model

    print("  Building detector (image model)...")
    detector = build_sam3_image_model(
        bpe_path=BPE_PATH,
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    ).to(device)

    print(f"  Loading fine-tuned weights: {detector_ckpt}")
    raw = torch.load(detector_ckpt, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw)
    missing, _ = detector.load_state_dict(state_dict, strict=False)
    n_loaded   = len(state_dict) - len(missing)
    val_iou    = raw.get("val_iou", float("nan"))
    epoch      = raw.get("epoch", "?")
    print(f"  Loaded {n_loaded} keys  |  epoch={epoch}  val_iou={val_iou:.3f}")

    with torch.no_grad():
        text_out = detector.backbone.forward_text(captions=[TEXT_QUERY], device=device)
    lang_feats = text_out["language_features"]   # (seq_len, 1, dim)
    lang_mask  = text_out["language_mask"]       # (1, seq_len)

    return detector, lang_feats, lang_mask


def detect_on_frame(detector, lang_feats, lang_mask,
                    frame_bgr: np.ndarray, w_vid: int, h_vid: int,
                    device: torch.device):
    """
    Run one detector forward pass on a pre-read BGR frame.
    Returns (box_norm_xyxy [0,1], box_abs_xyxy [pixels], conf).
    """
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = TF.resize(img, [DETECTOR_IMG_SIZE, DETECTOR_IMG_SIZE])
    img_t = TF.to_tensor(img)
    img_t = (img_t - _IMG_MEAN) / _IMG_STD
    img_t = img_t.unsqueeze(0).to(device)

    B = 1
    with torch.no_grad():
        vis_out = detector.backbone.forward_image(img_t)
        backbone_out = {
            **vis_out,
            "language_features": lang_feats.expand(-1, B, -1),
            "language_mask":     lang_mask.expand(B, -1),
        }
        find_input = FindStage(
            img_ids           = torch.arange(B, device=device),
            text_ids          = torch.zeros(B, dtype=torch.long, device=device),
            input_boxes       = torch.zeros(B, 0, 4, device=device),
            input_boxes_mask  = torch.zeros(B, 0, dtype=torch.bool, device=device),
            input_boxes_label = torch.zeros(B, 0, dtype=torch.long, device=device),
            input_points      = torch.zeros(B, 0, 2, device=device),
            input_points_mask = torch.zeros(B, 0, dtype=torch.bool, device=device),
        )
        geo_prompt = Prompt(
            box_embeddings = torch.zeros(0, B, 4, device=device),
            box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
        )
        prompt, prompt_mask, backbone_out = detector._encode_prompt(
            backbone_out, find_input, geo_prompt
        )
        backbone_out, encoder_out, _ = detector._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )
        out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
        out, _ = detector._run_decoder(
            memory      = out["encoder_hidden_states"],
            pos_embed   = encoder_out["pos_embed"],
            src_mask    = encoder_out["padding_mask"],
            out         = out,
            prompt      = prompt,
            prompt_mask = prompt_mask,
            encoder_out = encoder_out,
        )

    pred_logits = out["pred_logits"][0]
    pred_boxes  = out["pred_boxes"][0]
    scores = pred_logits.squeeze(-1).sigmoid()
    best_q = scores.argmax().item()
    conf   = scores[best_q].item()

    cx, cy, w, h = pred_boxes[best_q].cpu().tolist()
    box_norm_xyxy = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)
    box_abs_xyxy  = np.array([
        (cx - w/2) * w_vid, (cy - h/2) * h_vid,
        (cx + w/2) * w_vid, (cy + h/2) * h_vid,
    ], dtype=np.float32)

    return box_norm_xyxy, box_abs_xyxy, conf


# ── Tracker: load once ────────────────────────────────────────────────────────

def load_tracker(device: torch.device):
    """Build and load the SAM3 tracker from the pretrained checkpoint."""
    from sam3.model_builder import build_tracker

    predictor = build_tracker(apply_temporal_disambiguation=True, with_backbone=True)

    ckpt = torch.load(SAM3_CKPT, map_location="cpu", weights_only=True)
    ckpt = ckpt.get("model", ckpt)
    tracker_state  = {k[len("tracker."):]: v
                      for k, v in ckpt.items() if k.startswith("tracker.")}
    backbone_state = {k[len("detector."):]: v
                      for k, v in ckpt.items() if k.startswith("detector.backbone.")}
    predictor.load_state_dict({**tracker_state, **backbone_state}, strict=False)
    del ckpt, tracker_state, backbone_state

    predictor = predictor.to(device, dtype=torch.bfloat16)
    predictor.eval()
    predictor.teacher_force_obj_scores_for_mem = False
    return predictor


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chunked reinit prostate segmentation (Phase 1)"
    )
    parser.add_argument("--video",           required=True,
                        help="Input video path (.mp4)")
    parser.add_argument("--detector_ckpt",   required=True,
                        help="Fine-tuned detector checkpoint (.pth)")
    parser.add_argument("--output_dir",      default=None,
                        help="Output directory")
    parser.add_argument("--clip_start",      type=float, default=None,
                        help="Clip start time in seconds")
    parser.add_argument("--clip_end",        type=float, default=None,
                        help="Clip end time in seconds")
    parser.add_argument("--reinit_interval", type=int, default=150,
                        help="Re-detect + reset tracker every N frames "
                             "(default 150 ≈ 5 s at 30 fps)")
    parser.add_argument("--conf_threshold",  type=float, default=0.1,
                        help="Warn if detection confidence falls below this value")
    parser.add_argument("--mask_alpha",      type=float, default=0.35,
                        help="Overlay mask opacity, 0–1 (default: 0.35)")
    parser.add_argument("--mask_color",      default="0,255,0",
                        help="Mask overlay colour as R,G,B integers (default: 0,255,0)")
    args = parser.parse_args()

    assert os.path.isfile(args.video),         f"Video not found: {args.video}"
    assert os.path.isfile(args.detector_ckpt), f"Checkpoint not found: {args.detector_ckpt}"
    assert torch.cuda.is_available(), "CUDA required"
    if (args.clip_start is None) != (args.clip_end is None):
        parser.error("--clip_start and --clip_end must be given together")

    try:
        r, g, b = [int(x.strip()) for x in args.mask_color.split(",")]
    except ValueError:
        parser.error("--mask_color must be three comma-separated integers, e.g. '0,255,0'")
    mask_color_bgr = (b, g, r)

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = os.path.join(
            os.path.dirname(args.video), stem + "_sam3_chunked"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(SCRATCH_TMP, exist_ok=True)

    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Video:           {args.video}")
    print(f"Detector ckpt:   {args.detector_ckpt}")
    print(f"Reinit interval: {args.reinit_interval} frames")
    print(f"Mask color RGB:  {r},{g},{b}  alpha: {args.mask_alpha}")
    print(f"Output:          {args.output_dir}")

    # ── [0] Clip ──────────────────────────────────────────────────────────
    video_path = args.video
    if args.clip_start is not None:
        stem      = os.path.splitext(os.path.basename(args.video))[0]
        clip_name = f"{stem}_{int(args.clip_start)}-{int(args.clip_end)}.mp4"
        clip_path = os.path.join(args.output_dir, clip_name)
        print(f"\n[0] Clipping video  {args.clip_start}s – {args.clip_end}s → {clip_path}")
        t0 = time.time()
        clip_video(args.video, args.clip_start, args.clip_end, clip_path)
        print(f"  Clip written in {time.time()-t0:.1f}s  ({os.path.getsize(clip_path)/1e6:.1f} MB)")
        video_path = clip_path

    # ── [1] Video metadata ────────────────────────────────────────────────
    print(f"\n[1] Reading video metadata...")
    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    w_vid       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_chunks = (frame_count + args.reinit_interval - 1) // args.reinit_interval
    print(f"  {frame_count} frames, {w_vid}x{h_vid} @ {fps:.1f} fps")
    print(f"  Reinit every {args.reinit_interval} frames "
          f"({args.reinit_interval/fps:.1f} s at {fps:.1f} fps) → {n_chunks} chunks")

    # ── [2] Load detector (stays in memory for all chunks) ────────────────
    print(f"\n[2] Loading detector...")
    t0 = time.time()
    detector, lang_feats, lang_mask = load_detector(args.detector_ckpt, device)
    print(f"  Detector ready in {time.time()-t0:.1f}s")
    gpu_check("after detector load")

    # ── [3] Load tracker (stays in memory for all chunks) ─────────────────
    print(f"\n[3] Loading tracker...")
    t0 = time.time()
    predictor = load_tracker(device)
    print(f"  Tracker ready in {time.time()-t0:.1f}s")
    gpu_check("after tracker load")
    cpu_check("after tracker load")

    # ── [4] Init inference state (streaming) ─────────────────────────────
    print(f"\n[4] Initializing inference state (streaming)...")
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state, loader = init_state_streaming(predictor, video_path)
    n_frames = loader.n_frames
    print(f"  State initialized in {time.time()-t0:.1f}s  ({n_frames} frames)")

    # ── [5] Chunked detect + track loop ──────────────────────────────────
    print(f"\n[5] Chunked detect+track  ({n_chunks} chunks)...")
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    chunk_detections = {}    # chunk_start_frame -> (box_abs_xyxy, conf)
    total_mask_count = 0
    t_all = time.time()

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * args.reinit_interval
        chunk_end   = min(chunk_start + args.reinit_interval, n_frames)
        chunk_size  = chunk_end - chunk_start

        print(f"\n  -- Chunk {chunk_idx+1}/{n_chunks}  "
              f"frames {chunk_start}–{chunk_end-1}  "
              f"({chunk_size} frames, {chunk_size/fps:.1f} s) --", flush=True)

        # Read detect frame (first frame of chunk)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            print(f"    [WARN] Could not read frame {chunk_start}, skipping chunk.")
            continue

        # Detect
        t0 = time.time()
        box_norm, box_abs, conf = detect_on_frame(
            detector, lang_feats, lang_mask, frame_bgr, w_vid, h_vid, device
        )
        conf_tag = "  [LOW]" if conf < args.conf_threshold else ""
        print(f"    Detection: conf={conf:.3f}{conf_tag}  "
              f"box_norm={[round(x,3) for x in box_norm.tolist()]}  "
              f"({time.time()-t0:.1f}s)")
        chunk_detections[chunk_start] = (box_abs, conf)

        # Reset tracker memory; re-inject the streaming loader
        reset_tracker_state(inference_state)
        inference_state["images"]     = loader
        inference_state["num_frames"] = n_frames

        # Prompt with the new box at the chunk's start frame
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _low_res, prompt_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=chunk_start,
                obj_id=1,
                box=box_norm,
            )
        prompt_mask = (prompt_masks[0] > 0.0).cpu().numpy().squeeze()
        print(f"    Prompt mask: {prompt_mask.sum()} px, "
              f"coverage {prompt_mask.mean()*100:.1f}%")

        # Propagate through this chunk only
        chunk_mask_count = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, _obj_ids, _low_res, out_video_masks, _scores in \
                    propagate_with_guard(
                        predictor, inference_state,
                        start_frame_idx=chunk_start,
                        max_frame_num_to_track=chunk_size,
                    ):
                mask = (out_video_masks[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                if mask.any():
                    np.save(os.path.join(masks_dir, f"{out_frame_idx:06d}.npy"), mask)
                    chunk_mask_count += 1
                if out_frame_idx == chunk_start or out_frame_idx % 50 == 0:
                    print(f"      frame {out_frame_idx}: {int(mask.sum())} px  "
                          f"coverage {mask.mean()*100:.1f}%")

        elapsed = time.time() - t0
        total_mask_count += chunk_mask_count
        print(f"    Chunk done: {chunk_mask_count}/{chunk_size} frames masked  "
              f"({elapsed:.1f}s)", flush=True)
        gpu_check(f"chunk {chunk_idx+1}")

    print(f"\n  All chunks done in {time.time()-t_all:.1f}s — "
          f"{total_mask_count} masks total")
    cpu_check("after all chunks")

    # ── [6] Render overlay ────────────────────────────────────────────────
    print(f"\nRendering overlay video...")
    t0 = time.time()
    out_video_path = os.path.join(args.output_dir, "overlay.mp4")
    writer = cv2.VideoWriter(
        out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid)
    )
    cap = cv2.VideoCapture(video_path)
    for fidx in range(frame_count):
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
                overlay[mask.astype(bool)] = mask_color_bgr
                frame = cv2.addWeighted(frame, 1 - args.mask_alpha,
                                        overlay, args.mask_alpha, 0)
        # Annotate the first frame of each chunk with its detection box
        if fidx in chunk_detections:
            box_abs, conf = chunk_detections[fidx]
            x1, y1, x2, y2 = box_abs.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"reinit conf={conf:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"  Render done in {time.time()-t0:.1f}s → {out_video_path}")

    try:
        reset_tracker_state(inference_state)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Source:        {args.video}")
    print(f"  Processed:     {video_path}")
    print(f"  Reinit every:  {args.reinit_interval} frames ({args.reinit_interval/fps:.1f} s)")
    print(f"  Chunks:        {n_chunks}")
    print(f"  Masks saved:   {total_mask_count} / {frame_count} frames")
    print(f"  Output:        {out_video_path}")
    print(f"  GPU mem:       {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
