#!/usr/bin/env python3
"""
track_from_boxes.py
===================
Takes manual bounding-box annotations (from annotate_boxes.html) and runs
SAM2 tracking, reinitialising at each annotated frame with the provided box.

No SAM3 detector, no fine-tuned decoder — plain SAM2 large with box prompts.

Usage:
    python3 track_from_boxes.py \
        --video  aua_videos/HHY1_3D_85s_140s.mp4 \
        --boxes  aua_boxes/HHY1_3D_85s_140s_boxes.json \
        [--output_dir  aua_videos/inferred_videos/HHY1_3D_85s_140s_manual] \
        [--sbs_eye     left]      # left | right | none
        [--frame_step  1]         # 1 = every frame, 3 = every 3rd
        [--mask_alpha  0.35]
        [--mask_color  0,255,0]
"""

import argparse, bisect, json, os, sys, shutil, subprocess, tempfile, time
import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

ENCODE_BATCH_SIZE = 8


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


# ── Overlay helpers ────────────────────────────────────────────────────────────

def get_mask_at(kf_idxs, keyframe_masks, fidx):
    if not kf_idxs:
        return None
    pos = bisect.bisect_left(kf_idxs, fidx)
    if pos == len(kf_idxs):
        return keyframe_masks[kf_idxs[-1]]
    if pos == 0 or kf_idxs[pos] == fidx:
        return keyframe_masks[kf_idxs[pos]]
    prev_i, next_i = kf_idxs[pos - 1], kf_idxs[pos]
    a = (fidx - prev_i) / (next_i - prev_i)
    return (1 - a) * keyframe_masks[prev_i] + a * keyframe_masks[next_i]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True,  help="Input video path")
    ap.add_argument("--boxes",      required=True,  help="JSON from annotate_boxes.html")
    ap.add_argument("--output_dir", default=None,   help="Output directory (auto if omitted)")
    ap.add_argument("--sbs_eye",    default="left", choices=["left", "right", "none"],
                    help="Which SBS eye the boxes were drawn on (none = plain 2D)")
    ap.add_argument("--frame_step", type=int,   default=1,    help="Propagate every Nth frame")
    ap.add_argument("--mask_alpha", type=float, default=0.35, help="Overlay opacity")
    ap.add_argument("--mask_color", default="0,255,0",        help="Overlay colour R,G,B")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Load annotations ──────────────────────────────────────────────────────
    with open(args.boxes) as f:
        ann = json.load(f)

    fps_ann  = ann.get("fps", 30)
    video_w  = ann["video_w"]
    video_h  = ann["video_h"]
    ann_list = sorted(ann["boxes"], key=lambda b: b["frame"])

    if not ann_list:
        print("No boxes in JSON — nothing to do."); return

    # ── SBS setup ─────────────────────────────────────────────────────────────
    sbs = args.sbs_eye != "none"
    if sbs:
        eye_x   = 0 if args.sbs_eye == "left" else video_w // 2
        w_eye   = video_w // 2
        other_x = (video_w // 2) if args.sbs_eye == "left" else 0
    else:
        eye_x = 0; w_eye = video_w; other_x = None

    mask_color_bgr = tuple(int(v) for v in reversed(args.mask_color.split(",")))

    # ── Video metadata ────────────────────────────────────────────────────────
    cap      = cv2.VideoCapture(args.video)
    fps      = cap.get(cv2.CAP_PROP_FPS) or fps_ann
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── Output dirs ───────────────────────────────────────────────────────────
    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            "inferred_videos", stem + "_manual",
        )
    masks_dir = os.path.join(args.output_dir, "masks")
    for d in [args.output_dir, masks_dir, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Video    : {args.video}  ({n_frames} frames @ {fps:.1f} fps)")
    print(f"SBS eye  : {args.sbs_eye}  w_eye={w_eye}  eye_x={eye_x}")
    print(f"Step     : {args.frame_step}")
    print(f"Boxes    : {len(ann_list)}")
    for a in ann_list:
        print(f"  frame {a['frame']:5d}  t={a['time_s']:.2f}s  box={a['box']}")

    # ── Segments: from each annotated frame to the next (or end of video) ─────
    seg_starts = [a["frame"] for a in ann_list]
    seg_ends   = [a["frame"] for a in ann_list[1:]] + [n_frames]

    # ── GPU ───────────────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # ── Load SAM2 ─────────────────────────────────────────────────────────────
    print("\nLoading SAM2 large video predictor...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
    predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
    predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"  Ready  (image_size={image_size}, compile=default)")

    # ── Segment loop ──────────────────────────────────────────────────────────
    print(f"\nTracking {len(ann_list)} segment(s)...\n")
    t0         = time.time()
    mask_count = 0

    for si, (ann_entry, seg_start, seg_end) in enumerate(
            zip(ann_list, seg_starts, seg_ends)):

        chunk_frames = seg_end - seg_start

        # Convert full-video box coords → eye-space coords
        x1, y1, x2, y2 = ann_entry["box"]
        x1_eye = float(max(0,     x1 - eye_x))
        x2_eye = float(min(w_eye, x2 - eye_x))
        y1f, y2f = float(y1), float(y2)
        box_eye = np.array([x1_eye, y1f, x2_eye, y2f], dtype=np.float32)

        print(f"  seg {si+1}/{len(ann_list)}  "
              f"frames [{seg_start}–{seg_end})  "
              f"({chunk_frames} frames, {chunk_frames/fps:.1f}s)  "
              f"box_eye=[{x1_eye:.0f},{y1f:.0f},{x2_eye:.0f},{y2f:.0f}]",
              flush=True)

        # Read first frame for dummy init_state directory
        cap_tmp = cv2.VideoCapture(args.video)
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        ret, frame0 = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            print(f"    Cannot read frame {seg_start} — skipping segment"); continue
        if sbs:
            frame0 = frame0[:, eye_x:eye_x + w_eye]

        tmp_dir = tempfile.mkdtemp(prefix="track_boxes_", dir=SCRATCH_TMP)
        try:
            cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame0)
            state = predictor.init_state(video_path=tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        loader = VideoChunkLoader(
            args.video, image_size,
            seg_start, chunk_frames, args.frame_step,
            sbs_eye_x=eye_x, sbs_eye_w=(w_eye if sbs else None),
        )
        state["images"]       = loader
        state["num_frames"]   = loader.n_frames
        state["video_height"] = video_h
        state["video_width"]  = w_eye

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            # Pre-encode
            t_enc = time.perf_counter()
            pre_encode_chunk(predictor, state, loader, device)
            torch.cuda.synchronize()
            print(f"    pre-encode {len(loader)} keyframes  "
                  f"{time.perf_counter()-t_enc:.2f}s", flush=True)

            # Initialise with box at the first frame of this segment
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box_eye,
            )

            # Propagate
            t_prop = time.perf_counter()
            seg_masks = 0
            for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                mask       = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                global_idx = seg_start + local_idx * args.frame_step
                if mask.any():
                    np.save(os.path.join(masks_dir, f"{global_idx:06d}.npy"), mask)
                    seg_masks  += 1
                    mask_count += 1

            try:
                predictor.reset_state(state)
            except Exception:
                pass

        torch.cuda.synchronize()
        print(f"    propagate {time.perf_counter()-t_prop:.2f}s  "
              f"→ {seg_masks} masks", flush=True)
        del loader

    del predictor
    torch.cuda.empty_cache()
    print(f"\n  Inference done: {mask_count} masks in {time.time()-t0:.1f}s")

    # ── Render overlay ────────────────────────────────────────────────────────
    print("\nRendering overlay video...")

    keyframe_masks = {}
    for fname in sorted(os.listdir(masks_dir)):
        if not fname.endswith(".npy"): continue
        fidx = int(os.path.splitext(fname)[0])
        m = np.load(os.path.join(masks_dir, fname)).astype(np.float32)
        if m.shape[:2] != (video_h, w_eye):
            m = cv2.resize(m, (w_eye, video_h), interpolation=cv2.INTER_NEAREST)
        keyframe_masks[fidx] = m
    kf_idxs = sorted(keyframe_masks.keys())
    print(f"  {len(kf_idxs)} keyframe masks  eye={w_eye}×{video_h}")

    out_raw  = os.path.join(args.output_dir, "_raw_overlay.mp4")
    out_path = os.path.join(args.output_dir,
                            "overlay_sbs.mp4" if sbs else "overlay.mp4")

    cap         = cv2.VideoCapture(args.video)
    writer      = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (video_w, video_h))
    color_layer = np.full((video_h, w_eye, 3), mask_color_bgr, dtype=np.float32)
    box_dur     = max(1, round(fps))   # show reinit box for 1 second
    t0          = time.time()

    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret: break

        # Segmentation overlay
        mask_f = get_mask_at(kf_idxs, keyframe_masks, fidx)
        if mask_f is not None:
            alpha = (mask_f * args.mask_alpha)[:, :, None]
            sl = frame[:, eye_x:eye_x + w_eye].astype(np.float32)
            frame[:, eye_x:eye_x + w_eye] = (
                sl * (1 - alpha) + color_layer * alpha
            ).astype(np.uint8)
            if sbs and other_x is not None:
                ol = frame[:, other_x:other_x + w_eye].astype(np.float32)
                frame[:, other_x:other_x + w_eye] = (
                    ol * (1 - alpha) + color_layer * alpha
                ).astype(np.uint8)

        # Draw reinit box for box_dur frames after each annotated frame
        for a in ann_list:
            if a["frame"] <= fidx < a["frame"] + box_dur:
                bx1, by1, bx2, by2 = [int(v) for v in a["box"]]
                # Primary eye (box is stored in full-video coords)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, f"init f{a['frame']}",
                            (bx1, max(by1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                # Mirror to other eye for SBS
                if sbs and other_x is not None:
                    dx = other_x - eye_x
                    cv2.rectangle(frame, (bx1 + dx, by1), (bx2 + dx, by2), (0, 0, 255), 2)
                break

        writer.write(frame)
        if fidx % 300 == 0:
            print(f"  frame {fidx}/{n_frames}", flush=True)

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

    print(f"\n{'='*58}")
    print("DONE")
    print(f"  Masks : {masks_dir}")
    print(f"  Video : {out_path}")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
