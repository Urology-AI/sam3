#!/usr/bin/env python3
"""
track_nerves_2d.py
==================

2D version of track_nerves_ffmpeg_v2.py — propagates brush masks
annotated by annotate_brush_2d.html through a plain 2D video using
SAM2.

Drops the side-by-side eye cropping (no --sbs_eye flag, no other-eye
mirror) so the whole frame is encoded, propagated, and overlaid. Both
fps-ratio fixes from v2 are kept:

FIX 1 — Sub-chunk sizing in actual frames, not annotation frames, so
        a 59.94 fps video annotated at 30 fps does not OOM.
FIX 2 — Renderer iterates over ACTUAL frames and computes a fractional
        annotation-frame coordinate per frame for mask lookup, so masks
        line up with the underlying video content at any fps ratio.

Tune MAX_ACTUAL_FRAMES after running benchmark_sam2_frames.py.

Usage:

python3 aua_tracking/track_nerves_2d.py \\
  --video aua_videos/case_2d.mp4 \\
  --tracks aua_boxes/case_2d_mask_tracks.json \\
  --mask_alpha 0.10 \\
  --render_from_first_track
"""

import argparse
import base64
import bisect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple  # noqa: F401

import cv2
import numpy as np
import torch


# ── Paths ──────────────────────────────────────────────────────────────────────

SAM3_DIR = os.path.dirname(os.path.abspath(__file__))

SAM2_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam2_mask_tracks"

sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

ENCODE_BATCH_SIZE = 8

# Maximum ACTUAL video frames to encode per sub-chunk.
# Tune this with benchmark_sam2_frames.py.  500 is conservative for H100 80 GB.
MAX_ACTUAL_FRAMES = 500


# ── ffmpeg video helpers ───────────────────────────────────────────────────────

def _ffmpeg_read_bgr(video_path: str, time_s: float,
                     video_w: int, video_h: int) -> Optional[np.ndarray]:
    """Read a single BGR frame at presentation time time_s via ffmpeg."""
    cmd = ["ffmpeg", "-v", "quiet", "-ss", f"{time_s:.6f}", "-i", video_path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    expected = video_w * video_h * 3
    if len(result.stdout) < expected:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(video_h, video_w, 3).copy()


def _ffmpeg_frame_gen(video_path: str, start_time_s: float, duration_s: float,
                      video_w: int, video_h: int):
    """Yield BGR frames sequentially via ffmpeg pipe."""
    frame_bytes = video_w * video_h * 3
    cmd = ["ffmpeg", "-v", "quiet",
           "-ss", f"{start_time_s:.6f}",
           "-i", video_path,
           "-t", f"{duration_s:.6f}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(video_h, video_w, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait()


# ── FFmpeg chunk loader ────────────────────────────────────────────────────────

class FFmpegChunkLoader:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(
        self,
        video_path: str,
        image_size: int,
        start_frame: int,
        n_frames: int,
        fps_ann: float,
        video_w: int,
        video_h: int,
        step: int = 1,
    ):
        self.start_frame = start_frame
        self.step = step
        self.image_size = image_size

        start_time_s = start_frame / fps_ann
        duration_s   = n_frames    / fps_ann

        cmd = ["ffmpeg", "-v", "quiet",
               "-ss", f"{start_time_s:.6f}",
               "-i", video_path,
               "-t", f"{duration_s:.6f}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

        frame_bytes = video_w * video_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        all_frames: List[np.ndarray] = []
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(video_h, video_w, 3).copy()
            f = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            all_frames.append(f)
        proc.stdout.close()
        proc.wait()

        self._frames = all_frames[::step] if step > 1 else all_frames
        self.n_frames = max(1, len(self._frames))
        self._last_tensor: Optional[torch.Tensor] = None

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx: int):
        if local_idx >= len(self._frames):
            if self._last_tensor is not None:
                return self._last_tensor
            local_idx = len(self._frames) - 1

        frame = self._frames[local_idx]

        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        t -= self._MEAN
        t /= self._STD

        self._last_tensor = t
        return t


# ── SAM2 pre-encoding ──────────────────────────────────────────────────────────

def pre_encode_chunk(predictor, state: dict, loader: FFmpegChunkLoader, device: torch.device):
    n = len(loader)

    for batch_start in range(0, n, ENCODE_BATCH_SIZE):
        batch_end = min(batch_start + ENCODE_BATCH_SIZE, n)

        frames = [
            loader[i].to(device).float()
            for i in range(batch_start, batch_end)
        ]

        batch = torch.stack(frames, dim=0)
        bb = predictor.forward_image(batch)

        for j, frame_idx in enumerate(range(batch_start, batch_end)):
            state["cached_features"][frame_idx] = (
                batch[j:j + 1].clone(),
                {
                    "backbone_fpn": [
                        f[j:j + 1].clone()
                        for f in bb["backbone_fpn"]
                    ],
                    "vision_pos_enc": [
                        p[j:j + 1].clone()
                        for p in bb["vision_pos_enc"]
                    ],
                },
            )

        del batch, bb


# ── Mask helpers ───────────────────────────────────────────────────────────────

def decode_mask_png(mask_png: str) -> np.ndarray:
    if "," in mask_png:
        mask_png = mask_png.split(",", 1)[1]
    raw = base64.b64decode(mask_png)
    arr = np.frombuffer(raw, dtype=np.uint8)
    rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError("Could not decode mask PNG")
    if rgba.ndim == 2:
        alpha = rgba
    elif rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:
        gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
        alpha = gray
    return (alpha > 0).astype(np.uint8)


def resize_mask_if_needed(mask_full: np.ndarray, video_h: int, video_w: int) -> np.ndarray:
    if mask_full.shape[:2] != (video_h, video_w):
        mask_full = cv2.resize(
            mask_full, (video_w, video_h), interpolation=cv2.INTER_NEAREST,
        )
    return (mask_full > 0).astype(np.uint8)


class LazyMaskCache:
    """
    Loads mask .npy files on demand during sequential frame rendering.
    Keeps at most two keyframes in RAM at a time for linear interpolation.
    """

    def __init__(self, mask_dir: str, video_h: int, video_w: int):
        self._video_h = video_h
        self._video_w = video_w
        self._paths: Dict[int, str] = {}
        self.kf_idxs: List[int] = []

        if os.path.isdir(mask_dir):
            for fname in sorted(os.listdir(mask_dir)):
                if fname.endswith(".npy"):
                    fidx = int(os.path.splitext(fname)[0])
                    self.kf_idxs.append(fidx)
                    self._paths[fidx] = os.path.join(mask_dir, fname)

        self._cache: Dict[int, np.ndarray] = {}

    def _load(self, fidx: int) -> np.ndarray:
        if fidx not in self._cache:
            m = np.load(self._paths[fidx]).astype(np.float32)
            if m.shape[:2] != (self._video_h, self._video_w):
                m = cv2.resize(m, (self._video_w, self._video_h),
                               interpolation=cv2.INTER_NEAREST)
            self._cache[fidx] = m
        return self._cache[fidx]

    def _evict_before(self, fidx: int) -> None:
        for k in [k for k in self._cache if k < fidx]:
            del self._cache[k]

    def get_mask_at(self, fidx: float) -> Optional[np.ndarray]:
        if not self.kf_idxs:
            return None
        pos = bisect.bisect_left(self.kf_idxs, fidx)
        if pos == len(self.kf_idxs):
            last = self.kf_idxs[-1]
            m = self._load(last)
            self._evict_before(last)
            return m
        if pos == 0 or self.kf_idxs[pos] == fidx:
            cur = self.kf_idxs[pos]
            m = self._load(cur)
            self._evict_before(cur)
            return m
        prev_i = self.kf_idxs[pos - 1]
        next_i = self.kf_idxs[pos]
        a = (fidx - prev_i) / max(1, next_i - prev_i)
        prev_m = self._load(prev_i)
        next_m = self._load(next_i)
        self._evict_before(prev_i)
        return (1 - a) * prev_m + a * next_m


# ── Rendering helpers ──────────────────────────────────────────────────────────

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0, 255, 255)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def draw_label_panel(frame: np.ndarray, entries, panel_x: int, panel_y: int):
    if not entries:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    dot_r = 5
    dot_gap = 8
    pad_x = 12
    pad_y = 8
    line_h = 24
    text_widths = [cv2.getTextSize(lbl, font, font_scale, thickness)[0][0] for lbl, _ in entries]
    panel_w = pad_x + dot_r * 2 + dot_gap + max(text_widths) + pad_x
    panel_h = pad_y + len(entries) * line_h + pad_y
    h, w = frame.shape[:2]
    x0 = max(0, min(panel_x, w - 2))
    y0 = max(0, min(panel_y, h - 2))
    x1 = min(x0 + panel_w, w - 1)
    y1 = min(y0 + panel_h, h - 1)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    dark = np.full_like(roi, 20.0)
    frame[y0:y1, x0:x1] = (dark * 0.72 + roi * 0.28).astype(np.uint8)
    for i, (label, color_bgr) in enumerate(entries):
        row_cy = y0 + pad_y + i * line_h + line_h // 2
        cx = x0 + pad_x + dot_r
        cv2.circle(frame, (cx, row_cy), dot_r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, row_cy), dot_r, (240, 240, 240), 1, cv2.LINE_AA)
        tx = cx + dot_r + dot_gap
        ty = row_cy + 5
        cv2.putText(frame, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--mask_alpha", type=float, default=0.10)
    ap.add_argument("--clip_to_tracks", action="store_true")
    ap.add_argument("--clip_tail_s", type=float, default=2.0)
    ap.add_argument("--render_from_first_track", action="store_true")
    ap.add_argument("--mask_color", default="#00ff00")
    ap.add_argument("--track_ids", default=None)
    ap.add_argument("--no_compile", action="store_true")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(SCRATCH_TMP, exist_ok=True)

    with open(args.tracks, "r") as f:
        ann = json.load(f)

    fps_ann = float(ann.get("fps", 30))
    video_w_ann = int(ann["video_w"])
    video_h_ann = int(ann["video_h"])
    tracks = ann.get("tracks", [])

    if not tracks:
        print("No tracks in JSON — nothing to do.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or fps_ann
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or video_w_ann
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or video_h_ann
    cap.release()

    if video_w != video_w_ann or video_h != video_h_ann:
        print(
            f"WARNING: annotation video size {video_w_ann}x{video_h_ann} "
            f"differs from actual video {video_w}x{video_h}. Masks will be resized."
        )

    fps_ratio = fps / fps_ann
    ann_chunk_size = max(1, round(MAX_ACTUAL_FRAMES / fps_ratio))

    if args.output_dir is None:
        tracks_stem = os.path.splitext(os.path.basename(args.tracks))[0]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            "inferred_videos",
            tracks_stem,
        )

    masks_root = os.path.join(args.output_dir, "masks")
    for d in [args.output_dir, masks_root, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    print(f"Video      : {args.video}")
    print(f"Frames     : {n_frames}")
    print(f"FPS        : {fps:.3f}  (ann fps={fps_ann:.1f}  ratio={fps_ratio:.3f})")
    print(f"Size       : {video_w}x{video_h}")
    print(f"Tracks     : {len(tracks)}")
    print(f"Chunk size : {ann_chunk_size} ann-frames = ~{round(ann_chunk_size * fps_ratio)} actual frames")
    print(f"Output dir : {args.output_dir}")

    prepared_tracks = []

    for t in tracks:
        masks = sorted(t.get("masks", []), key=lambda m: int(m["frame"]))
        if not masks:
            print(f"  Track {t.get('id')} {t.get('label')}: no masks — will skip")
            continue

        start = t.get("start_frame")
        end   = t.get("end_frame")
        if start is None:
            start = int(masks[0]["frame"])
        if end is None:
            end = int(masks[-1]["frame"])

        start = max(0, int(start))
        end   = min(n_frames - 1, int(end))

        if end < start:
            print(f"  Track {t.get('id')} {t.get('label')}: invalid range — skipping")
            continue

        prepared = {
            "id": int(t["id"]),
            "label": t.get("label", f"track_{t['id']}"),
            "color": t.get("color", "#4fc3f7"),
            "start_frame": start,
            "end_frame": end,
            "masks": masks,
        }
        prepared_tracks.append(prepared)

        print(
            f"  [{prepared['id']:02d}] {prepared['label']:<30s} "
            f"frames [{start}–{end}] "
            f"{len(masks)} mask keyframe(s) "
            f"color={prepared['color']}"
        )

    if not prepared_tracks:
        print("No usable tracks after filtering.")
        return

    # ── GPU / SAM2 ─────────────────────────────────────────────────────────────

    assert torch.cuda.is_available(), "CUDA required for this SAM2 tracker"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("\nLoading SAM2 video predictor...")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)

    if not args.no_compile:
        print("Compiling SAM2 modules...")
        predictor.image_encoder   = torch.compile(predictor.image_encoder,   mode="default")
        predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
        predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    else:
        print("torch.compile disabled")

    print(f"Ready. image_size={image_size}")

    # ── Per-track SAM2 inference ───────────────────────────────────────────────

    total_mask_count = 0
    t0_total = time.time()

    for track in prepared_tracks:
        tid   = track["id"]
        label = track["label"]
        t_start = track["start_frame"]
        t_end   = track["end_frame"]

        track_masks_dir = os.path.join(masks_root, f"track_{tid:02d}")
        os.makedirs(track_masks_dir, exist_ok=True)

        mask_entries = sorted(track["masks"], key=lambda m: int(m["frame"]))

        print("\n" + "─" * 70)
        print(
            f"Track {tid:02d} | {label} | "
            f"frames [{t_start}–{t_end}] | "
            f"{len(mask_entries)} mask keyframe(s)"
        )

        key_masks_full = []
        for mi, m in enumerate(mask_entries):
            fidx = int(m["frame"])
            if fidx < t_start or fidx > t_end:
                print(f"  key mask {mi}: frame {fidx} outside track range — skipping")
                continue
            mask_full = decode_mask_png(m["mask_png"])
            mask_full = resize_mask_if_needed(mask_full, video_h, video_w)
            if not mask_full.any():
                print(f"  key mask {mi}: frame {fidx} empty — skipping")
                continue
            key_masks_full.append((fidx, mask_full))

        if not key_masks_full:
            print(f"  Track {tid:02d}: no usable non-empty masks — skipping")
            continue

        key_masks_full.sort(key=lambda x: x[0])
        seg_starts = [f for f, _ in key_masks_full]
        seg_ends   = seg_starts[1:] + [t_end + 1]

        track_mask_count = 0

        for si, ((seed_frame, seed_mask), seg_start, seg_end) in enumerate(
            zip(key_masks_full, seg_starts, seg_ends), start=1,
        ):
            seg_start = max(seg_start, t_start)
            seg_end   = min(seg_end,   t_end + 1)
            if seg_start >= seg_end:
                continue

            seg_len = seg_end - seg_start
            print(
                f"\n  Segment {si}/{len(key_masks_full)} "
                f"frames [{seg_start}–{seg_end}) "
                f"({seg_len} ann-frames, {seg_len / fps_ann:.1f}s ann / "
                f"{round(seg_len * fps_ratio)} actual frames, "
                f"{round(seg_len * fps_ratio) / fps:.1f}s actual) "
                f"seed_frame={seed_frame} seed_pixels={int(seed_mask.sum())}",
                flush=True,
            )

            sub_starts     = list(range(seg_start, seg_end, ann_chunk_size))
            carry_seed_mask = None

            for sub_i, sub_start in enumerate(sub_starts):
                sub_end = min(sub_start + ann_chunk_size, seg_end)
                sub_len = sub_end - sub_start

                loader = FFmpegChunkLoader(
                    args.video, image_size, sub_start, sub_len, fps_ann,
                    video_w, video_h,
                    step=args.frame_step,
                )

                frame0 = _ffmpeg_read_bgr(
                    args.video, sub_start / fps_ann, video_w, video_h,
                )
                if frame0 is None:
                    print(f"    Cannot read frame {sub_start} — skipping sub-chunk")
                    continue

                tmp_dir = tempfile.mkdtemp(prefix="sam2_mask_track_", dir=SCRATCH_TMP)
                try:
                    cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame0)
                    state = predictor.init_state(video_path=tmp_dir)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                state["images"]       = loader
                state["num_frames"]   = loader.n_frames
                state["video_height"] = video_h
                state["video_width"]  = video_w

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_enc = time.perf_counter()
                    pre_encode_chunk(predictor, state, loader, device)
                    torch.cuda.synchronize()
                    print(
                        f"    sub {sub_i + 1}/{len(sub_starts)} "
                        f"ann [{sub_start}–{sub_end})  "
                        f"{len(loader)} actual frames encoded in "
                        f"{time.perf_counter() - t_enc:.2f}s",
                        flush=True,
                    )

                    seed = seed_mask if (sub_i == 0 or carry_seed_mask is None) else carry_seed_mask
                    seed_tensor = torch.from_numpy((seed > 0).astype(np.uint8))
                    predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=1, mask=seed_tensor,
                    )

                    t_prop = time.perf_counter()

                    for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                        mask = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)

                        global_idx = sub_start + round(local_idx * args.frame_step * fps_ann / fps)

                        carry_seed_mask = mask

                        if mask.any():
                            np.save(
                                os.path.join(track_masks_dir, f"{global_idx:06d}.npy"),
                                mask,
                            )
                            track_mask_count += 1
                            total_mask_count += 1

                    torch.cuda.synchronize()
                    print(f"    propagated in {time.perf_counter() - t_prop:.2f}s", flush=True)

                    try:
                        predictor.reset_state(state)
                    except Exception:
                        pass
                    state["cached_features"].clear()

                del loader, state
                torch.cuda.empty_cache()

        print(f"\n  Track {tid:02d} complete: {track_mask_count} masks saved")

    del predictor
    torch.cuda.empty_cache()

    elapsed = time.time() - t0_total
    print(f"\nInference complete: {total_mask_count} total masks in {elapsed:.1f}s")

    # ── Load saved masks for rendering ─────────────────────────────────────────

    print("\nRendering overlay video...")

    track_render_data = {}
    for track in prepared_tracks:
        tid = track["id"]
        tmd = os.path.join(masks_root, f"track_{tid:02d}")
        mask_cache = LazyMaskCache(tmd, video_h, video_w)
        color_bgr = (
            hex_to_bgr(args.mask_color) if args.mask_color is not None
            else hex_to_bgr(track["color"])
        )
        track_render_data[tid] = {
            "mask_cache": mask_cache,
            "color_bgr":  color_bgr,
            "label":      track["label"],
            "start":      track["start_frame"],
            "end":        track["end_frame"],
        }
        print(f"  Track {tid:02d} {track['label']:<30s}: {len(mask_cache.kf_idxs)} rendered mask frames")

    out_raw  = os.path.join(args.output_dir, "_raw_overlay.mp4")
    out_path = os.path.join(args.output_dir, "overlay.mp4")

    render_start = 0
    render_end   = n_frames

    if args.render_from_first_track and prepared_tracks:
        first_ann_frame = min(t["start_frame"] for t in prepared_tracks)
        last_ann_frame  = max(t["end_frame"]   for t in prepared_tracks)
        render_start = max(0,        first_ann_frame)
        render_end   = min(n_frames, last_ann_frame + 1)
        print(
            f"render_from_first_track: frames {render_start}–{render_end - 1} "
            f"(first={first_ann_frame}, last={last_ann_frame})"
        )

    elif args.clip_to_tracks and prepared_tracks:
        last_ann_frame = max(t["end_frame"] for t in prepared_tracks)
        render_start = 0
        render_end   = min(n_frames, last_ann_frame + int(fps * args.clip_tail_s) + 1)
        print(
            f"clip_to_tracks: frames 0–{render_end - 1} "
            f"(last annotation ends at {last_ann_frame}, +{args.clip_tail_s:.1f}s tail)"
        )

    else:
        print(f"Rendering full video: frames 0–{n_frames - 1}")

    if render_end <= render_start:
        raise RuntimeError(f"Invalid render range: start={render_start}, end={render_end}")

    render_start_s  = render_start / fps_ann
    render_duration = (render_end - render_start) / fps_ann

    writer = cv2.VideoWriter(
        out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_w, video_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_raw}")

    t0_render = time.time()

    _render_gen = _ffmpeg_frame_gen(
        args.video, render_start_s, render_duration, video_w, video_h,
    )

    # Iterate over ACTUAL frames consumed from ffmpeg (not ann-frame indices).
    # Each consumed actual frame maps to a fractional ann-frame coordinate for
    # mask lookup and track-active range checks.
    n_render_actual = int(round((render_end - render_start) * fps / fps_ann))

    for out_idx in range(n_render_actual):
        frame = next(_render_gen, None)
        if frame is None:
            print(f"Could not read actual frame {out_idx}; stopping render.")
            break

        ann_fidx = render_start + out_idx * fps_ann / fps

        active_labels = []
        for track in prepared_tracks:
            tid = track["id"]
            td  = track_render_data[tid]
            if ann_fidx < td["start"] or ann_fidx > td["end"]:
                continue
            active_labels.append((td["label"], td["color_bgr"]))
            mask_f = td["mask_cache"].get_mask_at(ann_fidx)
            if mask_f is None:
                continue
            mask_f = np.clip(mask_f, 0.0, 1.0)
            alpha = (mask_f * args.mask_alpha)[:, :, None]
            color_layer = np.full((video_h, video_w, 3), td["color_bgr"], dtype=np.float32)
            frame_f = frame.astype(np.float32)
            frame[:] = (frame_f * (1.0 - alpha) + color_layer * alpha).astype(np.uint8)

        if active_labels:
            draw_label_panel(frame, active_labels, 16, 16)

        writer.write(frame)

        if out_idx % 600 == 0:
            print(
                f"  actual {out_idx}/{n_render_actual} "
                f"(ann ~{ann_fidx:.1f}/{render_end - 1})",
                flush=True,
            )

    writer.release()

    print("Re-muxing to H.264...")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", out_raw,
         "-c:v", "libx264", "-crf", "18", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("ffmpeg error:")
        print(r.stderr[:1000])
        print(f"Keeping raw video at: {out_raw}")
    else:
        try:
            os.remove(out_raw)
        except OSError:
            pass

    print(f"Rendered in {time.time() - t0_render:.1f}s")
    print("\n" + "=" * 70)
    print("DONE")
    print(f"Masks : {masks_root}")
    print(f"Video : {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
