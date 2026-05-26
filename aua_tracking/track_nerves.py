#!/usr/bin/env python3
"""
track_multi_structure_masks.py
==============================

Runs SAM2 tracking from brush-drawn mask keyframes exported by
annotate_mask_tracks.html.

This version supports:
  - Mask prompts instead of boxes
  - 3D side-by-side videos with --sbs_eye left/right
  - Drawing masks on the left eye and propagating only that eye
  - Rendering overlay back onto both eyes
  - Optional rendering only from first track start to last track end using:
        --render_from_first_track

Usage:

python3 track_multi_structure_masks.py \
  --video aua_videos/case.mp4 \
  --tracks aua_boxes/case_mask_tracks.json \
  --sbs_eye left \
  --frame_step 1 \
  --mask_alpha 0.30 \
  --render_from_first_track

For full video rendering, omit --render_from_first_track.
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
from typing import Dict, List, Optional, Tuple

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
MAX_CACHED_FRAMES = 600


# ── Video chunk loader ─────────────────────────────────────────────────────────

class VideoChunkLoader:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(
        self,
        video_path: str,
        image_size: int,
        start_frame: int,
        n_frames: int,
        step: int = 1,
        sbs_eye_x: int = 0,
        sbs_eye_w: Optional[int] = None,
    ):
        self.start_frame = start_frame
        self.step = step
        self.image_size = image_size
        self.sbs_eye_x = sbs_eye_x
        self.sbs_eye_w = sbs_eye_w

        self._cap = cv2.VideoCapture(video_path)
        self._last_tensor = None

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = min(n_frames, max(0, total_frames - start_frame))

        self.n_frames = max(1, (n_frames + step - 1) // step)
        self._next_raw = start_frame

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx: int):
        raw = self.start_frame + local_idx * self.step

        if raw != self._next_raw:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
            self._next_raw = raw

        ret, frame = self._cap.read()
        self._next_raw += 1

        if not ret:
            if self._last_tensor is not None:
                return self._last_tensor
            raise IndexError(f"Frame {raw} unreadable with no prior frame fallback")

        if self.sbs_eye_w is not None:
            frame = frame[:, self.sbs_eye_x:self.sbs_eye_x + self.sbs_eye_w]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        t -= self._MEAN
        t /= self._STD

        self._last_tensor = t
        return t

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


# ── SAM2 pre-encoding ──────────────────────────────────────────────────────────

def pre_encode_chunk(predictor, state: dict, loader: VideoChunkLoader, device: torch.device):
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
    """
    Decode data:image/png;base64,... into uint8 binary mask.

    Returns:
        HxW uint8 mask with values 0/1.
    """
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


def crop_mask_to_eye(
    mask_full: np.ndarray,
    video_h: int,
    video_w: int,
    eye_x: int,
    eye_w: int,
    sbs: bool,
) -> np.ndarray:
    """
    Annotation mask is stored in full video coordinates.

    For SBS videos, SAM2 tracks only the selected eye.
    Example:
      --sbs_eye left  => crop mask_full[:, 0:video_w//2]
      --sbs_eye right => crop mask_full[:, video_w//2:video_w]
    """
    if mask_full.shape[:2] != (video_h, video_w):
        mask_full = cv2.resize(
            mask_full,
            (video_w, video_h),
            interpolation=cv2.INTER_NEAREST,
        )

    if sbs:
        mask_eye = mask_full[:, eye_x:eye_x + eye_w]
    else:
        mask_eye = mask_full

    return (mask_eye > 0).astype(np.uint8)


def get_mask_at(kf_idxs: List[int], kf_masks: Dict[int, np.ndarray], fidx: int):
    """
    Get a mask for rendering.

    Since actual propagated masks are usually saved for many frames, this often
    returns the exact frame. If frame_step > 1, it linearly blends between nearby
    saved masks for smoother display.
    """
    if not kf_idxs:
        return None

    pos = bisect.bisect_left(kf_idxs, fidx)

    if pos == len(kf_idxs):
        return kf_masks[kf_idxs[-1]]

    if pos == 0 or kf_idxs[pos] == fidx:
        return kf_masks[kf_idxs[pos]]

    prev_i = kf_idxs[pos - 1]
    next_i = kf_idxs[pos]

    a = (fidx - prev_i) / max(1, next_i - prev_i)

    return (1 - a) * kf_masks[prev_i] + a * kf_masks[next_i]


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
    """
    entries: list of (label_text, color_bgr)
    """
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

    text_widths = [
        cv2.getTextSize(lbl, font, font_scale, thickness)[0][0]
        for lbl, _ in entries
    ]

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

        cv2.putText(
            frame,
            label,
            (tx + 1, ty + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            label,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    ap.add_argument(
        "--tracks",
        required=True,
        help="JSON exported from annotate_mask_tracks.html",
    )

    ap.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. If omitted, creates inferred_videos/<video>_mask_tracks",
    )

    ap.add_argument(
        "--sbs_eye",
        default="left",
        choices=["left", "right", "none"],
        help="Which SBS eye the mask was drawn on. Use none for plain 2D video.",
    )

    ap.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Propagate every Nth frame. Default: 1",
    )

    ap.add_argument(
        "--mask_alpha",
        type=float,
        default=0.10,
        help="Overlay opacity in output video. Default: 0.10",
    )

    ap.add_argument(
        "--clip_to_tracks",
        action="store_true",
        help="Render from frame 0 until a few seconds after the last annotated track.",
    )

    ap.add_argument(
        "--clip_tail_s",
        type=float,
        default=2.0,
        help="Tail seconds after last track if --clip_to_tracks is set. Default: 2.0",
    )

    ap.add_argument(
        "--render_from_first_track",
        action="store_true",
        help="Render only from first track start frame to last track end frame.",
    )

    ap.add_argument(
        "--mask_color",
        default=None,
        help="Override all track colors with a hex value, e.g. '#00ff00' for pure green.",
    )

    ap.add_argument(
        "--track_ids",
        default=None,
        help="Comma-separated list of track IDs to process, e.g. '0,1,2,3,4,5,6'. "
             "If omitted, all tracks are processed.",
    )

    ap.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for easier debugging.",
    )

    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(SCRATCH_TMP, exist_ok=True)

    with open(args.tracks, "r") as f:
        ann = json.load(f)

    fps_ann = ann.get("fps", 30)
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
            f"differs from actual video {video_w}x{video_h}. "
            f"Masks will be resized."
        )

    # ── SBS setup ──────────────────────────────────────────────────────────────

    sbs = args.sbs_eye != "none"

    if sbs:
        eye_w = video_w // 2
        eye_x = 0 if args.sbs_eye == "left" else eye_w
        other_x = eye_w if args.sbs_eye == "left" else 0
    else:
        eye_x = 0
        eye_w = video_w
        other_x = None

    # ── Output dirs ────────────────────────────────────────────────────────────

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
    print(f"FPS        : {fps:.3f}")
    print(f"Size       : {video_w}x{video_h}")
    print(f"SBS eye    : {args.sbs_eye}  eye_x={eye_x}  eye_w={eye_w}")
    print(f"Tracks     : {len(tracks)}")
    print(f"Output dir : {args.output_dir}")

    # ── Validate / prepare track metadata ──────────────────────────────────────

    prepared_tracks = []

    for t in tracks:
        masks = sorted(t.get("masks", []), key=lambda m: int(m["frame"]))

        if not masks:
            print(f"  Track {t.get('id')} {t.get('label')}: no masks — will skip")
            continue

        start = t.get("start_frame")
        end = t.get("end_frame")

        if start is None:
            start = int(masks[0]["frame"])

        if end is None:
            end = int(masks[-1]["frame"])

        start = max(0, int(start))
        end = min(n_frames - 1, int(end))

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
        predictor.image_encoder = torch.compile(predictor.image_encoder, mode="default")
        predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
        predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    else:
        print("torch.compile disabled")

    print(f"Ready. image_size={image_size}")

    # ── Per-track SAM2 inference ───────────────────────────────────────────────

    total_mask_count = 0
    t0_total = time.time()

    for track in prepared_tracks:
        tid = track["id"]
        label = track["label"]
        t_start = track["start_frame"]
        t_end = track["end_frame"]

        track_masks_dir = os.path.join(masks_root, f"track_{tid:02d}")
        os.makedirs(track_masks_dir, exist_ok=True)

        mask_entries = sorted(track["masks"], key=lambda m: int(m["frame"]))

        print("\n" + "─" * 70)
        print(
            f"Track {tid:02d} | {label} | "
            f"frames [{t_start}–{t_end}] | "
            f"{len(mask_entries)} mask keyframe(s)"
        )

        key_masks_eye = []

        for mi, m in enumerate(mask_entries):
            fidx = int(m["frame"])

            if fidx < t_start or fidx > t_end:
                print(f"  key mask {mi}: frame {fidx} outside track range — skipping")
                continue

            mask_full = decode_mask_png(m["mask_png"])

            mask_eye = crop_mask_to_eye(
                mask_full=mask_full,
                video_h=video_h,
                video_w=video_w,
                eye_x=eye_x,
                eye_w=eye_w,
                sbs=sbs,
            )

            if not mask_eye.any():
                print(f"  key mask {mi}: frame {fidx} empty after crop — skipping")
                continue

            key_masks_eye.append((fidx, mask_eye))

        if not key_masks_eye:
            print(f"  Track {tid:02d}: no usable non-empty masks — skipping")
            continue

        key_masks_eye.sort(key=lambda x: x[0])

        seg_starts = [f for f, _ in key_masks_eye]
        seg_ends = seg_starts[1:] + [t_end + 1]

        track_mask_count = 0

        for si, ((seed_frame, seed_mask), seg_start, seg_end) in enumerate(
            zip(key_masks_eye, seg_starts, seg_ends),
            start=1,
        ):
            seg_start = max(seg_start, t_start)
            seg_end = min(seg_end, t_end + 1)

            if seg_start >= seg_end:
                continue

            seg_len = seg_end - seg_start

            print(
                f"\n  Segment {si}/{len(key_masks_eye)} "
                f"frames [{seg_start}–{seg_end}) "
                f"({seg_len} frames, {seg_len / fps:.1f}s) "
                f"seed_frame={seed_frame} "
                f"seed_pixels={int(seed_mask.sum())}",
                flush=True,
            )

            sub_starts = list(range(seg_start, seg_end, MAX_CACHED_FRAMES))
            carry_seed_mask = None

            for sub_i, sub_start in enumerate(sub_starts):
                sub_end = min(sub_start + MAX_CACHED_FRAMES, seg_end)
                sub_len = sub_end - sub_start

                loader = VideoChunkLoader(
                    args.video,
                    image_size,
                    sub_start,
                    sub_len,
                    args.frame_step,
                    sbs_eye_x=eye_x,
                    sbs_eye_w=(eye_w if sbs else None),
                )

                cap_tmp = cv2.VideoCapture(args.video)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, sub_start)
                ret, frame0 = cap_tmp.read()
                cap_tmp.release()

                if not ret:
                    print(f"    Cannot read frame {sub_start} — skipping sub-chunk")
                    continue

                if sbs:
                    frame0 = frame0[:, eye_x:eye_x + eye_w]

                tmp_dir = tempfile.mkdtemp(prefix="sam2_mask_track_", dir=SCRATCH_TMP)

                try:
                    cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame0)
                    state = predictor.init_state(video_path=tmp_dir)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                state["images"] = loader
                state["num_frames"] = loader.n_frames
                state["video_height"] = video_h
                state["video_width"] = eye_w

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_enc = time.perf_counter()

                    pre_encode_chunk(predictor, state, loader, device)
                    torch.cuda.synchronize()

                    print(
                        f"    sub {sub_i + 1}/{len(sub_starts)} "
                        f"[{sub_start}–{sub_end}) "
                        f"encoded {len(loader)} frames in "
                        f"{time.perf_counter() - t_enc:.2f}s",
                        flush=True,
                    )

                    if sub_i == 0 or carry_seed_mask is None:
                        seed = seed_mask
                    else:
                        seed = carry_seed_mask

                    seed_tensor = torch.from_numpy((seed > 0).astype(np.uint8))

                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=1,
                        mask=seed_tensor,
                    )

                    t_prop = time.perf_counter()

                    for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                        mask = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                        global_idx = sub_start + local_idx * args.frame_step

                        carry_seed_mask = mask

                        if mask.any():
                            np.save(
                                os.path.join(track_masks_dir, f"{global_idx:06d}.npy"),
                                mask,
                            )

                            track_mask_count += 1
                            total_mask_count += 1

                    torch.cuda.synchronize()

                    print(
                        f"    propagated in {time.perf_counter() - t_prop:.2f}s",
                        flush=True,
                    )

                    try:
                        predictor.reset_state(state)
                    except Exception:
                        pass

                del loader
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

        kf_masks = {}

        if os.path.isdir(tmd):
            for fname in sorted(os.listdir(tmd)):
                if not fname.endswith(".npy"):
                    continue

                fidx = int(os.path.splitext(fname)[0])
                m = np.load(os.path.join(tmd, fname)).astype(np.float32)

                if m.shape[:2] != (video_h, eye_w):
                    m = cv2.resize(
                        m,
                        (eye_w, video_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                kf_masks[fidx] = m

        color_bgr = (
            hex_to_bgr(args.mask_color)
            if args.mask_color is not None
            else hex_to_bgr(track["color"])
        )

        track_render_data[tid] = {
            "kf_idxs": sorted(kf_masks.keys()),
            "kf_masks": kf_masks,
            "color_bgr": color_bgr,
            "label": track["label"],
            "start": track["start_frame"],
            "end": track["end_frame"],
        }

        print(
            f"  Track {tid:02d} {track['label']:<30s}: "
            f"{len(kf_masks)} rendered mask frames"
        )

    out_raw = os.path.join(args.output_dir, "_raw_overlay.mp4")
    out_path = os.path.join(
        args.output_dir,
        "overlay_sbs.mp4" if sbs else "overlay.mp4",
    )

    # ── Rendering range logic ──────────────────────────────────────────────────

    render_start = 0
    render_end = n_frames

    if args.render_from_first_track and prepared_tracks:
        first_ann_frame = min(t["start_frame"] for t in prepared_tracks)
        last_ann_frame = max(t["end_frame"] for t in prepared_tracks)

        render_start = max(0, first_ann_frame)
        render_end = min(n_frames, last_ann_frame + 1)

        print(
            f"render_from_first_track: rendering frames "
            f"{render_start}–{render_end - 1} "
            f"(first track starts at {first_ann_frame}, "
            f"last track ends at {last_ann_frame})"
        )

    elif args.clip_to_tracks and prepared_tracks:
        last_ann_frame = max(t["end_frame"] for t in prepared_tracks)

        render_start = 0
        render_end = min(
            n_frames,
            last_ann_frame + int(fps * args.clip_tail_s) + 1,
        )

        print(
            f"clip_to_tracks: rendering frames 0–{render_end - 1} "
            f"(last annotation ends at {last_ann_frame}, "
            f"+{args.clip_tail_s:.1f}s tail)"
        )

    else:
        print(f"Rendering full video: frames 0–{n_frames - 1}")

    if render_end <= render_start:
        raise RuntimeError(
            f"Invalid render range: start={render_start}, end={render_end}"
        )

    # ── Render overlay video ───────────────────────────────────────────────────

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, render_start)

    writer = cv2.VideoWriter(
        out_raw,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_w, video_h),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_raw}")

    t0_render = time.time()

    for fidx in range(render_start, render_end):
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame {fidx}; stopping render.")
            break

        active_labels = []

        for track in prepared_tracks:
            tid = track["id"]
            td = track_render_data[tid]

            if fidx < td["start"] or fidx > td["end"]:
                continue

            active_labels.append((td["label"], td["color_bgr"]))

            mask_f = get_mask_at(td["kf_idxs"], td["kf_masks"], fidx)

            if mask_f is None:
                continue

            mask_f = np.clip(mask_f, 0.0, 1.0)

            alpha = (mask_f * args.mask_alpha)[:, :, None]
            color_layer = np.full(
                (video_h, eye_w, 3),
                td["color_bgr"],
                dtype=np.float32,
            )

            # Overlay on the annotated/propagated eye.
            eye_slice = frame[:, eye_x:eye_x + eye_w].astype(np.float32)

            frame[:, eye_x:eye_x + eye_w] = (
                eye_slice * (1.0 - alpha) + color_layer * alpha
            ).astype(np.uint8)

            # For SBS videos, mirror the same mask overlay onto the other eye.
            if sbs and other_x is not None:
                other_slice = frame[:, other_x:other_x + eye_w].astype(np.float32)

                frame[:, other_x:other_x + eye_w] = (
                    other_slice * (1.0 - alpha) + color_layer * alpha
                ).astype(np.uint8)

        if active_labels:
            draw_label_panel(frame, active_labels, eye_x + 16, 16)

            if sbs and other_x is not None:
                draw_label_panel(frame, active_labels, other_x + 16, 16)

        writer.write(frame)

        if (fidx - render_start) % 300 == 0:
            print(
                f"  frame {fidx}/{render_end - 1} "
                f"({fidx - render_start + 1}/{render_end - render_start})",
                flush=True,
            )

    cap.release()
    writer.release()

    # ── Re-encode to H.264 ─────────────────────────────────────────────────────

    print("Re-muxing to H.264...")

    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            out_raw,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ],
        capture_output=True,
        text=True,
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