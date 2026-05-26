#!/usr/bin/env python3
"""
track_multi_structure.py
========================
Takes multi-structure annotations (from annotate_multi.html) and runs SAM2
tracking for each structure independently, producing a single composite overlay
video with professional structure labels.

Usage:
    python3 track_multi_structure.py \
        --video  aua_videos/HHY1_3D_750s_870s.mp4 \
        --tracks aua_boxes/HHY1_multi_tracks.json \
        [--output_dir  aua_videos/inferred_videos/HHY1_multi] \
        [--sbs_eye     left]      # left | right | none
        [--frame_step  1]
        [--mask_alpha  0.30]
"""

import argparse, bisect, json, os, sys, shutil, subprocess, tempfile, time
import cv2
import numpy as np
import torch

SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"
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


# ── Mask interpolation ─────────────────────────────────────────────────────────

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


# ── Rendering helpers ──────────────────────────────────────────────────────────

def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_label_panel(frame, entries, panel_x, panel_y):
    """entries: list of (label_text, color_bgr). Draws a professional label chip."""
    if not entries:
        return

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    dot_r      = 5
    dot_gap    = 8
    pad_x      = 12
    pad_y      = 8
    line_h     = 24

    text_widths = [
        cv2.getTextSize(lbl, font, font_scale, thickness)[0][0]
        for lbl, _ in entries
    ]
    panel_w = pad_x + dot_r * 2 + dot_gap + max(text_widths) + pad_x
    panel_h = pad_y + len(entries) * line_h + pad_y

    h, w = frame.shape[:2]
    x0, y0 = panel_x, panel_y
    x1 = min(x0 + panel_w, w - 1)
    y1 = min(y0 + panel_h, h - 1)

    # Semi-transparent dark background
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    dark = np.full_like(roi, 20.0)
    frame[y0:y1, x0:x1] = (dark * 0.72 + roi * 0.28).astype(np.uint8)

    for i, (label, color_bgr) in enumerate(entries):
        row_cy = y0 + pad_y + i * line_h + line_h // 2
        cx     = x0 + pad_x + dot_r
        # Colored dot with thin white outline
        cv2.circle(frame, (cx, row_cy), dot_r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, row_cy), dot_r, (240, 240, 240), 1, cv2.LINE_AA)
        # Text: shadow + white
        tx = cx + dot_r + dot_gap
        ty = row_cy + 5
        cv2.putText(frame, label, (tx + 1, ty + 1), font, font_scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True,  help="Input video path")
    ap.add_argument("--tracks",     required=True,  help="JSON from annotate_multi.html")
    ap.add_argument("--output_dir", default=None,   help="Output directory (auto if omitted)")
    ap.add_argument("--sbs_eye",    default="left", choices=["left", "right", "none"],
                    help="Which SBS eye the boxes were drawn on (none = plain 2D)")
    ap.add_argument("--frame_step",      type=int,   default=1,    help="Propagate every Nth frame")
    ap.add_argument("--mask_alpha",      type=float, default=0.30, help="Overlay opacity")
    ap.add_argument("--clip_to_tracks",  action="store_true",
                    help="Stop rendering a few seconds after the last annotation end frame")
    ap.add_argument("--clip_tail_s",     type=float, default=2.0,
                    help="Seconds of tail to include after last annotation end (default 2)")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.tracks) as f:
        ann = json.load(f)

    fps_ann = ann.get("fps", 30)
    video_w = ann["video_w"]
    video_h = ann["video_h"]
    tracks  = ann["tracks"]

    if not tracks:
        print("No tracks in JSON — nothing to do."); return

    # SBS setup
    sbs = args.sbs_eye != "none"
    if sbs:
        eye_x   = 0 if args.sbs_eye == "left" else video_w // 2
        w_eye   = video_w // 2
        other_x = (video_w // 2) if args.sbs_eye == "left" else 0
    else:
        eye_x = 0; w_eye = video_w; other_x = None

    # Video metadata
    cap      = cv2.VideoCapture(args.video)
    fps      = cap.get(cv2.CAP_PROP_FPS) or fps_ann
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Output dirs
    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            "inferred_videos", stem + "_multi",
        )
    masks_root = os.path.join(args.output_dir, "masks")
    for d in [args.output_dir, masks_root, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    print(f"Video    : {args.video}  ({n_frames} frames @ {fps:.1f} fps)")
    print(f"SBS eye  : {args.sbs_eye}  w_eye={w_eye}  eye_x={eye_x}")
    print(f"Tracks   : {len(tracks)}")
    for t in tracks:
        print(f"  [{t['id']:02d}] {t['label']:<30s}  "
              f"frames [{t['start_frame']}–{t['end_frame']}]  "
              f"{len(t['boxes'])} box(es)  color={t['color']}")

    # GPU
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Load SAM2
    print("\nLoading SAM2 large video predictor...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
    predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
    predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"  Ready  (image_size={image_size}, compile=default)")

    total_mask_count = 0
    t0_total = time.time()

    # ── Per-track inference ────────────────────────────────────────────────────
    for track in tracks:
        tid     = track["id"]
        label   = track["label"]
        t_start = track["start_frame"]
        t_end   = min(track["end_frame"], n_frames - 1)

        boxes_sorted = sorted(track["boxes"], key=lambda b: b["frame"])
        if not boxes_sorted:
            print(f"\n  Track {tid:02d} ({label}): no boxes — skipping"); continue

        track_masks_dir = os.path.join(masks_root, f"track_{tid:02d}")
        os.makedirs(track_masks_dir, exist_ok=True)

        print(f"\n{'─'*62}")
        print(f"  Track {tid:02d}  {label}  "
              f"frames [{t_start}–{t_end}]  {len(boxes_sorted)} box(es)")

        # Each annotated box initiates a segment running to the next box (or t_end+1)
        seg_starts = [b["frame"] for b in boxes_sorted]
        seg_ends   = [b["frame"] for b in boxes_sorted[1:]] + [t_end + 1]

        track_mask_count = 0

        for si, (box_entry, seg_start, seg_end) in enumerate(
                zip(boxes_sorted, seg_starts, seg_ends)):

            seg_start = max(seg_start, t_start)
            seg_end   = min(seg_end,   t_end + 1)
            if seg_start >= seg_end:
                continue

            x1, y1, x2, y2 = box_entry["box"]
            x1_eye = float(max(0,     x1 - eye_x))
            x2_eye = float(min(w_eye, x2 - eye_x))
            box_eye = np.array([x1_eye, float(y1), x2_eye, float(y2)], dtype=np.float32)

            seg_len = seg_end - seg_start
            print(f"\n  seg {si+1}/{len(boxes_sorted)}  "
                  f"frames [{seg_start}–{seg_end})  "
                  f"({seg_len} frames, {seg_len/fps:.1f}s)  "
                  f"box_eye=[{x1_eye:.0f},{y1:.0f},{x2_eye:.0f},{y2:.0f}]",
                  flush=True)

            sub_starts = list(range(seg_start, seg_end, MAX_CACHED_FRAMES))
            seed_mask  = None

            for sub_i, sub_start in enumerate(sub_starts):
                sub_end = min(sub_start + MAX_CACHED_FRAMES, seg_end)
                sub_len = sub_end - sub_start

                loader = VideoChunkLoader(
                    args.video, image_size,
                    sub_start, sub_len, args.frame_step,
                    sbs_eye_x=eye_x, sbs_eye_w=(w_eye if sbs else None),
                )

                cap_tmp = cv2.VideoCapture(args.video)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, sub_start)
                ret, frame0 = cap_tmp.read()
                cap_tmp.release()
                if not ret:
                    print(f"    Cannot read frame {sub_start} — skipping sub-chunk")
                    continue
                if sbs:
                    frame0 = frame0[:, eye_x:eye_x + w_eye]

                tmp_dir = tempfile.mkdtemp(prefix="track_multi_", dir=SCRATCH_TMP)
                try:
                    cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame0)
                    state = predictor.init_state(video_path=tmp_dir)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                state["images"]       = loader
                state["num_frames"]   = loader.n_frames
                state["video_height"] = video_h
                state["video_width"]  = w_eye

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_enc = time.perf_counter()
                    pre_encode_chunk(predictor, state, loader, device)
                    torch.cuda.synchronize()
                    print(f"    sub {sub_i+1}/{len(sub_starts)}  "
                          f"[{sub_start}–{sub_end})  "
                          f"encode {len(loader)} frames  "
                          f"{time.perf_counter()-t_enc:.2f}s",
                          flush=True)

                    if sub_i == 0 or seed_mask is None:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=0, obj_id=1,
                            box=box_eye,
                        )
                    else:
                        predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=0, obj_id=1,
                            mask=torch.from_numpy(seed_mask),
                        )

                    t_prop = time.perf_counter()
                    for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                        mask       = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                        global_idx = sub_start + local_idx * args.frame_step
                        seed_mask  = mask
                        if mask.any():
                            np.save(
                                os.path.join(track_masks_dir, f"{global_idx:06d}.npy"),
                                mask,
                            )
                            track_mask_count  += 1
                            total_mask_count  += 1

                    torch.cuda.synchronize()
                    print(f"    propagate {time.perf_counter()-t_prop:.2f}s",
                          flush=True)

                    try:
                        predictor.reset_state(state)
                    except Exception:
                        pass

                del loader
                torch.cuda.empty_cache()

        print(f"\n  Track {tid:02d} complete: {track_mask_count} masks")

    del predictor
    torch.cuda.empty_cache()
    elapsed = time.time() - t0_total
    print(f"\n  Inference done: {total_mask_count} total masks in {elapsed:.1f}s")

    # ── Render overlay ─────────────────────────────────────────────────────────
    print("\nRendering overlay video...")

    track_render_data = {}
    for track in tracks:
        tid = track["id"]
        tmd = os.path.join(masks_root, f"track_{tid:02d}")
        kf_masks = {}
        if os.path.isdir(tmd):
            for fname in sorted(os.listdir(tmd)):
                if not fname.endswith(".npy"): continue
                fidx = int(os.path.splitext(fname)[0])
                m = np.load(os.path.join(tmd, fname)).astype(np.float32)
                if m.shape[:2] != (video_h, w_eye):
                    m = cv2.resize(m, (w_eye, video_h), interpolation=cv2.INTER_NEAREST)
                kf_masks[fidx] = m
        track_render_data[tid] = {
            "kf_idxs":   sorted(kf_masks.keys()),
            "kf_masks":  kf_masks,
            "color_bgr": hex_to_bgr(track["color"]),
            "label":     track["label"],
            "start":     track["start_frame"],
            "end":       track["end_frame"],
        }
        print(f"  Track {tid:02d} {track['label']:<30s}: "
              f"{len(kf_masks)} keyframe masks")

    out_raw  = os.path.join(args.output_dir, "_raw_overlay.mp4")
    out_path = os.path.join(args.output_dir,
                            "overlay_sbs.mp4" if sbs else "overlay.mp4")

    if args.clip_to_tracks and tracks:
        last_ann_frame = max(t["end_frame"] for t in tracks)
        render_end     = min(n_frames, last_ann_frame + int(fps * args.clip_tail_s) + 1)
        print(f"  clip_to_tracks: rendering frames 0–{render_end-1} "
              f"(last annotation ends at {last_ann_frame}, +{args.clip_tail_s:.1f}s tail)")
    else:
        render_end = n_frames

    cap    = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (video_w, video_h))
    t0 = time.time()

    for fidx in range(render_end):
        ret, frame = cap.read()
        if not ret: break

        active_labels = []

        for track in tracks:
            tid = track["id"]
            td  = track_render_data[tid]

            if fidx < td["start"] or fidx > td["end"]:
                continue

            # Always add to label stack when within the track's window
            active_labels.append((td["label"], td["color_bgr"]))

            # Apply mask overlay only when mask exists
            mask_f = get_mask_at(td["kf_idxs"], td["kf_masks"], fidx)
            if mask_f is not None:
                alpha       = (mask_f * args.mask_alpha)[:, :, None]
                color_layer = np.full((video_h, w_eye, 3),
                                      td["color_bgr"], dtype=np.float32)
                sl = frame[:, eye_x:eye_x + w_eye].astype(np.float32)
                frame[:, eye_x:eye_x + w_eye] = (
                    sl * (1 - alpha) + color_layer * alpha
                ).astype(np.uint8)
                if sbs and other_x is not None:
                    ol = frame[:, other_x:other_x + w_eye].astype(np.float32)
                    frame[:, other_x:other_x + w_eye] = (
                        ol * (1 - alpha) + color_layer * alpha
                    ).astype(np.uint8)

        if active_labels:
            draw_label_panel(frame, active_labels, eye_x + 16, 16)
            if sbs and other_x is not None:
                draw_label_panel(frame, active_labels, other_x + 16, 16)

        writer.write(frame)
        if fidx % 300 == 0:
            print(f"  frame {fidx}/{render_end}", flush=True)

    cap.release()
    writer.release()

    print("  Re-muxing to H.264...")
    r = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("  ffmpeg error:", r.stderr[:400])
    os.remove(out_raw)
    print(f"  Rendered in {time.time()-t0:.1f}s")

    print(f"\n{'='*62}")
    print("DONE")
    print(f"  Masks : {masks_root}")
    print(f"  Video : {out_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
