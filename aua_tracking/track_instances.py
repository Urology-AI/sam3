#!/usr/bin/env python3
"""
track_instances.py
==================
Tracks multiple simultaneous instances of the same structure using SAM2.
Reads annotations exported by annotate_instances.html.

Each annotated frame can have N boxes (N >= 1), one per visible instance.
All instances within a track are tracked as separate SAM2 objects in the same
propagation pass and rendered with the same track colour.

Usage:
    python3 track_instances.py \
        --video  aua_videos/case.mov \
        --tracks aua_boxes/case_instances.json \
        --sbs_eye left \
        --render_from_first_track

JSON format (from annotate_instances.html):
    annotation_type: "multi_instance_boxes"
    tracks[].boxes[]: {frame, time_s, instance_idx, box: [x1,y1,x2,y2]}
    Multiple boxes per frame allowed (different instance_idx values).

Output masks:
    masks/track_00_obj_1/000100.npy   track 0, instance 0, frame 100
    masks/track_00_obj_2/000100.npy   track 0, instance 1, frame 100
"""

import argparse, bisect, json, os, sys, shutil, subprocess, tempfile, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam2_instances"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

ENCODE_BATCH_SIZE = 8
MAX_CACHED_FRAMES = 600


# ── Video chunk loader ─────────────────────────────────────────────────────────

class VideoChunkLoader:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path, image_size, start_frame, n_frames, step=1,
                 sbs_eye_x=0, sbs_eye_w=None):
        self.start_frame  = start_frame
        self.step         = step
        self.image_size   = image_size
        self.sbs_eye_x    = sbs_eye_x
        self.sbs_eye_w    = sbs_eye_w
        self._cap         = cv2.VideoCapture(video_path)
        self._last_tensor = None
        n_frames          = min(n_frames,
                                int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame)
        self.n_frames     = max(1, (n_frames + step - 1) // step)
        self._next_raw    = start_frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx):
        raw = self.start_frame + local_idx * self.step
        if raw != self._next_raw:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
            self._next_raw = raw
        ret, frame = self._cap.read()
        self._next_raw += 1
        if not ret:
            if self._last_tensor is not None:
                return self._last_tensor
            raise IndexError(f"Frame {raw} unreadable with no prior frame to fall back to")
        if self.sbs_eye_w is not None:
            frame = frame[:, self.sbs_eye_x:self.sbs_eye_x + self.sbs_eye_w]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        t -= self._MEAN
        t /= self._STD
        self._last_tensor = t
        return t

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


# ── Batch pre-encoding ─────────────────────────────────────────────────────────

def pre_encode_chunk(predictor, state, loader, device):
    n = len(loader)
    for batch_start in range(0, n, ENCODE_BATCH_SIZE):
        batch_end = min(batch_start + ENCODE_BATCH_SIZE, n)
        frames = [loader[i].to(device).float() for i in range(batch_start, batch_end)]
        batch  = torch.stack(frames, dim=0)
        bb     = predictor.forward_image(batch)
        for j, frame_idx in enumerate(range(batch_start, batch_end)):
            state["cached_features"][frame_idx] = (
                batch[j:j+1].clone(),
                {
                    "backbone_fpn":   [f[j:j+1].clone() for f in bb["backbone_fpn"]],
                    "vision_pos_enc": [p[j:j+1].clone() for p in bb["vision_pos_enc"]],
                },
            )
        del batch, bb


# ── Helpers ────────────────────────────────────────────────────────────────────

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def get_mask_at(kf_idxs, kf_masks, fidx):
    if not kf_idxs:
        return None
    pos = bisect.bisect_left(kf_idxs, fidx)
    if pos == len(kf_idxs):
        return kf_masks[kf_idxs[-1]]
    if pos == 0 or kf_idxs[pos] == fidx:
        return kf_masks[kf_idxs[pos]]
    prev_i, next_i = kf_idxs[pos - 1], kf_idxs[pos]
    a = (fidx - prev_i) / (next_i - prev_i)
    return (1 - a) * kf_masks[prev_i] + a * kf_masks[next_i]


def draw_label_panel(frame, entries, panel_x, panel_y):
    if not entries:
        return
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    dot_r, dot_gap, pad_x, pad_y, line_h = 5, 8, 12, 8, 24
    text_widths = [cv2.getTextSize(lbl, font, font_scale, thickness)[0][0] for lbl, _ in entries]
    panel_w = pad_x + dot_r * 2 + dot_gap + max(text_widths) + pad_x
    panel_h = pad_y + len(entries) * line_h + pad_y
    h, w = frame.shape[:2]
    x0, y0 = panel_x, panel_y
    x1 = min(x0 + panel_w, w - 1)
    y1 = min(y0 + panel_h, h - 1)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (np.full_like(roi, 20.0) * 0.72 + roi * 0.28).astype(np.uint8)
    for i, (label, color_bgr) in enumerate(entries):
        row_cy = y0 + pad_y + i * line_h + line_h // 2
        cx = x0 + pad_x + dot_r
        cv2.circle(frame, (cx, row_cy), dot_r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, row_cy), dot_r, (240, 240, 240), 1, cv2.LINE_AA)
        tx, ty = cx + dot_r + dot_gap, row_cy + 5
        cv2.putText(frame, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty),         font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--tracks",     required=True, help="JSON from annotate_instances.html")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sbs_eye",    default="left", choices=["left", "right", "none"])
    ap.add_argument("--frame_step", type=int,   default=1)
    ap.add_argument("--mask_alpha", type=float, default=0.30)
    ap.add_argument("--mask_color", default=None,
                    help="Override all track colors with a hex value e.g. '#00ff00'")
    ap.add_argument("--render_from_first_track", action="store_true",
                    help="Render only the annotated range instead of the full video")
    ap.add_argument("--no_compile", action="store_true")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(SCRATCH_TMP, exist_ok=True)

    with open(args.tracks) as f:
        ann = json.load(f)

    fps_ann    = ann.get("fps", 30)
    video_w_ann = int(ann["video_w"])
    video_h_ann = int(ann["video_h"])
    tracks_ann  = ann.get("tracks", [])

    if not tracks_ann:
        print("No tracks in JSON — nothing to do.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps      = cap.get(cv2.CAP_PROP_FPS) or fps_ann
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or video_w_ann
    video_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or video_h_ann
    cap.release()

    if video_w != video_w_ann or video_h != video_h_ann:
        print(f"WARNING: annotation size {video_w_ann}x{video_h_ann} != video {video_w}x{video_h} — masks will be resized")

    sbs = args.sbs_eye != "none"
    if sbs:
        eye_w   = video_w // 2
        eye_x   = 0 if args.sbs_eye == "left" else eye_w
        other_x = eye_w if args.sbs_eye == "left" else 0
    else:
        eye_w = video_w; eye_x = 0; other_x = None

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            "inferred_videos", stem + "_instances",
        )
    masks_root = os.path.join(args.output_dir, "masks")
    for d in [args.output_dir, masks_root, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    # ── Validate tracks ────────────────────────────────────────────────────────

    prepared_tracks = []
    for t in tracks_ann:
        boxes = sorted(t.get("boxes", []), key=lambda b: (b["frame"], b["instance_idx"]))
        if not boxes:
            print(f"  Track {t.get('id')} {t.get('label')}: no boxes — skipping")
            continue

        start = t.get("start_frame") or int(boxes[0]["frame"])
        end   = t.get("end_frame")   or int(boxes[-1]["frame"])
        start = max(0, int(start))
        end   = min(n_frames - 1, int(end))

        # Group boxes by seed frame
        by_frame = defaultdict(list)
        for b in boxes:
            by_frame[int(b["frame"])].append(b)
        seed_frames = sorted(by_frame.keys())

        max_inst = max(len(v) for v in by_frame.values())
        print(f"  [{int(t['id']):02d}] {t.get('label',''):<30s} "
              f"frames [{start}–{end}]  "
              f"{len(seed_frames)} seed frame(s)  "
              f"max {max_inst} simultaneous  "
              f"color={t.get('color','#ffffff')}")

        prepared_tracks.append({
            "id":          int(t["id"]),
            "label":       t.get("label", f"track_{t['id']}"),
            "color":       t.get("color", "#4fc3f7"),
            "start_frame": start,
            "end_frame":   end,
            "seed_frames": seed_frames,
            "by_frame":    by_frame,
        })

    if not prepared_tracks:
        print("No usable tracks after filtering.")
        return

    print(f"\nVideo   : {args.video}  ({n_frames} frames @ {fps:.1f} fps  {video_w}x{video_h})")
    print(f"SBS eye : {args.sbs_eye}  eye_x={eye_x}  eye_w={eye_w}")

    # ── GPU / SAM2 ─────────────────────────────────────────────────────────────

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)

    if not args.no_compile:
        print("Compiling SAM2 modules...")
        predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
        predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
        predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"Ready. image_size={image_size}")

    # ── Per-track inference ────────────────────────────────────────────────────

    t0_total = time.time()
    total_masks = 0

    for track in prepared_tracks:
        tid          = track["id"]
        t_start      = track["start_frame"]
        t_end        = track["end_frame"]
        seed_frames  = track["seed_frames"]
        by_frame     = track["by_frame"]

        seg_starts = seed_frames
        seg_ends   = seed_frames[1:] + [t_end + 1]

        print(f"\n{'─'*70}")
        print(f"Track {tid:02d} | {track['label']} | frames [{t_start}–{t_end}] | {len(seed_frames)} segment(s)")

        track_mask_count = 0

        for si, (seed_frame, seg_start, seg_end) in enumerate(zip(seed_frames, seg_starts, seg_ends), 1):
            seg_start = max(seg_start, t_start)
            seg_end   = min(seg_end, t_end + 1)
            if seg_start >= seg_end:
                continue

            instances_at_seed = sorted(by_frame[seed_frame], key=lambda b: b["instance_idx"])
            n_inst = len(instances_at_seed)
            seg_len = seg_end - seg_start

            print(f"\n  seg {si}/{len(seed_frames)}  frames [{seg_start}–{seg_end})  "
                  f"({seg_len} frames, {seg_len/fps:.1f}s)  "
                  f"{n_inst} instance(s) at seed frame {seed_frame}", flush=True)

            # Instance directories: obj_id is instance_idx + 1
            inst_dirs = {}
            for inst in instances_at_seed:
                obj_id = inst["instance_idx"] + 1
                d = os.path.join(masks_root, f"track_{tid:02d}_obj_{obj_id}")
                os.makedirs(d, exist_ok=True)
                inst_dirs[obj_id] = d

            sub_starts   = list(range(seg_start, seg_end, MAX_CACHED_FRAMES))
            carry_masks  = {}  # obj_id -> last propagated mask array

            for sub_i, sub_start in enumerate(sub_starts):
                sub_end = min(sub_start + MAX_CACHED_FRAMES, seg_end)
                sub_len = sub_end - sub_start

                loader = VideoChunkLoader(
                    args.video, image_size,
                    sub_start, sub_len, args.frame_step,
                    sbs_eye_x=eye_x, sbs_eye_w=(eye_w if sbs else None),
                )

                cap_tmp = cv2.VideoCapture(args.video)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, sub_start)
                ret, frame0 = cap_tmp.read()
                cap_tmp.release()
                if not ret:
                    print(f"    Cannot read frame {sub_start} — skipping")
                    continue
                if sbs:
                    frame0 = frame0[:, eye_x:eye_x + eye_w]

                tmp_dir = tempfile.mkdtemp(prefix="sam2_inst_", dir=SCRATCH_TMP)
                try:
                    cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame0)
                    state = predictor.init_state(video_path=tmp_dir)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                state["images"]       = loader
                state["num_frames"]   = loader.n_frames
                state["video_height"] = video_h
                state["video_width"]  = eye_w

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_enc = time.perf_counter()
                    pre_encode_chunk(predictor, state, loader, device)
                    torch.cuda.synchronize()
                    print(f"    sub {sub_i+1}/{len(sub_starts)} [{sub_start}–{sub_end}) "
                          f"encoded {len(loader)} frames in {time.perf_counter()-t_enc:.2f}s",
                          flush=True)

                    if sub_i == 0 or not carry_masks:
                        # First sub-chunk: initialize from annotated boxes
                        for inst in instances_at_seed:
                            obj_id = inst["instance_idx"] + 1
                            x1, y1, x2, y2 = inst["box"]
                            x1e = float(max(0,     x1 - eye_x))
                            x2e = float(min(eye_w, x2 - eye_x))
                            box_eye = np.array([x1e, float(y1), x2e, float(y2)], dtype=np.float32)
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=0, obj_id=obj_id,
                                box=box_eye,
                            )
                    else:
                        # Subsequent sub-chunks: carry masks from previous chunk
                        for obj_id, cmask in carry_masks.items():
                            seed_tensor = torch.from_numpy((cmask > 0).astype(np.uint8))
                            predictor.add_new_mask(
                                inference_state=state,
                                frame_idx=0, obj_id=obj_id,
                                mask=seed_tensor,
                            )

                    t_prop = time.perf_counter()
                    carry_masks = {}

                    for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                        global_idx = sub_start + local_idx * args.frame_step
                        for k, oid in enumerate(obj_ids):
                            mask = (mask_logits[k][0] > 0.0).cpu().numpy().astype(np.uint8)
                            carry_masks[oid] = mask
                            if mask.any() and oid in inst_dirs:
                                np.save(os.path.join(inst_dirs[oid], f"{global_idx:06d}.npy"), mask)
                                track_mask_count += 1
                                total_masks      += 1

                    torch.cuda.synchronize()
                    print(f"    propagated in {time.perf_counter()-t_prop:.2f}s", flush=True)

                    try:
                        predictor.reset_state(state)
                    except Exception:
                        pass

                del loader
                torch.cuda.empty_cache()

        print(f"\n  Track {tid:02d} complete: {track_mask_count} masks saved")

    del predictor
    torch.cuda.empty_cache()
    print(f"\nInference complete: {total_masks} total masks in {time.time()-t0_total:.1f}s")

    # ── Load saved masks for rendering ─────────────────────────────────────────

    print("\nRendering overlay video...")

    track_render_data = {}
    for track in prepared_tracks:
        tid = track["id"]
        color_bgr = (hex_to_bgr(args.mask_color) if args.mask_color
                     else hex_to_bgr(track["color"]))

        # Find all instance directories for this track
        instances = {}
        for entry in sorted(os.listdir(masks_root)):
            if not entry.startswith(f"track_{tid:02d}_obj_"):
                continue
            obj_id   = int(entry.split("_obj_")[1])
            inst_dir = os.path.join(masks_root, entry)
            kf_masks = {}
            for fname in sorted(os.listdir(inst_dir)):
                if not fname.endswith(".npy"):
                    continue
                fidx = int(os.path.splitext(fname)[0])
                m = np.load(os.path.join(inst_dir, fname)).astype(np.float32)
                if m.shape[:2] != (video_h, eye_w):
                    m = cv2.resize(m, (eye_w, video_h), interpolation=cv2.INTER_NEAREST)
                kf_masks[fidx] = m
            if kf_masks:
                instances[obj_id] = {
                    "kf_idxs":    sorted(kf_masks.keys()),
                    "kf_masks":   kf_masks,
                    "first_frame": min(kf_masks.keys()),
                    "last_frame":  max(kf_masks.keys()),
                }

        track_render_data[tid] = {
            "instances": instances,
            "color_bgr": color_bgr,
            "label":     track["label"],
            "start":     track["start_frame"],
            "end":       track["end_frame"],
        }
        n_inst_frames = sum(len(v["kf_idxs"]) for v in instances.values())
        print(f"  Track {tid:02d} {track['label']:<30s}: "
              f"{len(instances)} instance(s)  {n_inst_frames} total mask frames")

    # ── Rendering range ────────────────────────────────────────────────────────

    render_start = 0
    render_end   = n_frames

    if args.render_from_first_track and prepared_tracks:
        first_ann = min(t["start_frame"] for t in prepared_tracks)
        last_ann  = max(t["end_frame"]   for t in prepared_tracks)
        render_start = max(0, first_ann)
        render_end   = min(n_frames, last_ann + 1)
        print(f"render_from_first_track: frames {render_start}–{render_end-1}")
    else:
        print(f"Rendering full video: frames 0–{n_frames-1}")

    # ── Render ─────────────────────────────────────────────────────────────────

    out_raw  = os.path.join(args.output_dir, "_raw_overlay.mp4")
    out_path = os.path.join(args.output_dir, "overlay_sbs.mp4" if sbs else "overlay.mp4")

    cap    = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, render_start)
    writer = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (video_w, video_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_raw}")

    t0_render = time.time()

    for fidx in range(render_start, render_end):
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {fidx}; stopping.")
            break

        active_labels = []

        for track in prepared_tracks:
            tid = track["id"]
            td  = track_render_data[tid]

            if fidx < td["start"] or fidx > td["end"]:
                continue

            active_labels.append((td["label"], td["color_bgr"]))
            color_layer = np.full((video_h, eye_w, 3), td["color_bgr"], dtype=np.float32)

            for obj_id, inst in td["instances"].items():
                # Only render within this instance's tracked range
                if fidx < inst["first_frame"] or fidx > inst["last_frame"]:
                    continue

                mask_f = get_mask_at(inst["kf_idxs"], inst["kf_masks"], fidx)
                if mask_f is None:
                    continue

                alpha = (np.clip(mask_f, 0.0, 1.0) * args.mask_alpha)[:, :, None]

                eye_sl = frame[:, eye_x:eye_x + eye_w].astype(np.float32)
                frame[:, eye_x:eye_x + eye_w] = (
                    eye_sl * (1.0 - alpha) + color_layer * alpha
                ).astype(np.uint8)

                if sbs and other_x is not None:
                    other_sl = frame[:, other_x:other_x + eye_w].astype(np.float32)
                    frame[:, other_x:other_x + eye_w] = (
                        other_sl * (1.0 - alpha) + color_layer * alpha
                    ).astype(np.uint8)

        if active_labels:
            draw_label_panel(frame, active_labels, eye_x + 16, 16)
            if sbs and other_x is not None:
                draw_label_panel(frame, active_labels, other_x + 16, 16)

        writer.write(frame)

        if (fidx - render_start) % 300 == 0:
            print(f"  frame {fidx}/{render_end-1}", flush=True)

    cap.release()
    writer.release()

    print("Re-muxing to H.264...")
    r = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ], capture_output=True, text=True)

    if r.returncode != 0:
        print("ffmpeg error:", r.stderr[:500])
        print(f"Keeping raw video at: {out_raw}")
    else:
        try:
            os.remove(out_raw)
        except OSError:
            pass

    print(f"Rendered in {time.time()-t0_render:.1f}s")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"Masks : {masks_root}")
    print(f"Video : {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
