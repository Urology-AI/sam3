#!/usr/bin/env python3
"""
track_from_boxes_finetuned.py
=============================
Same as track_from_boxes.py but uses the fine-tuned SAM2 mask decoder to
convert each bounding box into a clean initial mask before handing off to
SAM2 video propagation.  Plain add_new_points_or_box is replaced by
decoder_box_to_mask → add_new_mask.

Usage:
    python3 track_from_boxes_finetuned.py \
        --video      aua_videos/HHY1_3D_750s_870s.mp4 \
        --boxes      aua_boxes/HHY1_3D_750s_870s_boxes.json \
        [--output_dir  aua_videos/inferred_videos/HHY1_3D_750s_870s_finetuned] \
        [--sbs_eye     left]
        [--frame_step  1]
        [--mask_alpha  0.15]
        [--mask_color  0,255,0]
"""

import argparse, bisect, json, os, sys, shutil, subprocess, tempfile, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sam3/
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM2_CKPT    = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
DECODER_CKPT = os.path.join(SAM3_DIR,
               "sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth")

ENCODE_BATCH_SIZE  = 8
MAX_CACHED_FRAMES  = 600   # max frames pre-encoded per sub-chunk; limits GPU memory
SAM2_IMG_SIZE     = 1024
_BB_FEAT_SIZES    = [(256, 256), (128, 128), (64, 64)]


# ── Fine-tuned decoder ─────────────────────────────────────────────────────────

def load_finetuned_decoder(device):
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    raw = torch.load(DECODER_CKPT, map_location="cpu", weights_only=True)
    model.sam_mask_decoder.load_state_dict(raw.get("state_dict", raw))
    print(f"  Fine-tuned decoder loaded  epoch={raw.get('epoch','?')}  "
          f"val_iou={raw.get('val_iou', float('nan')):.3f}")
    return model


def decoder_box_to_mask(still_model, frame_bgr_eye, box_eye, w_eye, h_vid, device):
    """Run fine-tuned decoder on one eye-crop frame with a box prompt.

    frame_bgr_eye : BGR image already cropped to eye width (w_eye × h_vid)
    box_eye       : [x1, y1, x2, y2] in eye-space pixel coords
    Returns       : bool mask (h_vid, w_eye)
    """
    img = cv2.cvtColor(frame_bgr_eye, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SAM2_IMG_SIZE, SAM2_IMG_SIZE))
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_t = torch.from_numpy(
        ((img.astype(np.float32) / 255.0) - img_mean) / img_std
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    sx = SAM2_IMG_SIZE / w_eye
    sy = SAM2_IMG_SIZE / h_vid
    x1, y1, x2, y2 = box_eye
    box_1024 = torch.tensor(
        [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        backbone_out = still_model.forward_image(img_t)
        _, vision_feats, _, _ = still_model._prepare_backbone_features(backbone_out)
        if getattr(still_model, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + still_model.no_mem_embed
        B = 1
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], _BB_FEAT_SIZES[::-1])
        ][::-1]
        image_embed    = feats[-1]
        high_res_feats = feats[:-1]

        box_coords = box_1024.reshape(1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
        sparse_emb, dense_emb = still_model.sam_prompt_encoder(
            points=(box_coords, box_labels), boxes=None, masks=None,
        )
        low_res_masks, _, _, _ = still_model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=still_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        mask_full = F.interpolate(
            low_res_masks, size=(h_vid, w_eye),
            mode="bilinear", align_corners=False,
        )
    return (mask_full[0, 0] > 0.0).cpu().numpy()


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


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--boxes",      required=True,  help="JSON from annotate_boxes.html")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sbs_eye",    default="left", choices=["left", "right", "none"])
    ap.add_argument("--frame_step", type=int,   default=1)
    ap.add_argument("--mask_alpha", type=float, default=0.35)
    ap.add_argument("--mask_color", default="0,255,0")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.boxes) as f:
        ann = json.load(f)

    fps_ann  = ann.get("fps", 30)
    video_w  = ann["video_w"]
    video_h  = ann["video_h"]
    ann_list = sorted(ann["boxes"], key=lambda b: b["frame"])

    if not ann_list:
        print("No boxes in JSON — nothing to do."); return

    # ── SBS ───────────────────────────────────────────────────────────────────
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
            "inferred_videos", stem + "_finetuned",
        )
    masks_dir = os.path.join(args.output_dir, "masks")
    for d in [args.output_dir, masks_dir, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    print(f"Video    : {args.video}  ({n_frames} frames @ {fps:.1f} fps)")
    print(f"SBS eye  : {args.sbs_eye}  w_eye={w_eye}  eye_x={eye_x}")
    print(f"Step     : {args.frame_step}")
    print(f"Boxes    : {len(ann_list)}")
    for a in ann_list:
        print(f"  frame {a['frame']:5d}  t={a['time_s']:.2f}s  box={a['box']}")

    seg_starts = [a["frame"] for a in ann_list]
    seg_ends   = [a["frame"] for a in ann_list[1:]] + [n_frames]

    # ── GPU ───────────────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # ── Load fine-tuned still-image decoder ───────────────────────────────────
    print("\nLoading fine-tuned SAM2 decoder...")
    still_model = load_finetuned_decoder(device)

    # ── Load SAM2 video predictor ─────────────────────────────────────────────
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
        x1, y1, x2, y2 = ann_entry["box"]
        x1_eye = float(max(0,     x1 - eye_x))
        x2_eye = float(min(w_eye, x2 - eye_x))
        box_eye = [x1_eye, float(y1), x2_eye, float(y2)]

        print(f"  seg {si+1}/{len(ann_list)}  "
              f"frames [{seg_start}–{seg_end})  ({chunk_frames} frames, {chunk_frames/fps:.1f}s)  "
              f"box_eye=[{x1_eye:.0f},{y1:.0f},{x2_eye:.0f},{y2:.0f}]", flush=True)

        # Read first frame (BGR, eye-cropped) for the decoder
        cap_tmp = cv2.VideoCapture(args.video)
        cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        ret, frame0_full = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            print(f"    Cannot read frame {seg_start} — skipping"); continue
        frame0_eye = frame0_full[:, eye_x:eye_x + w_eye] if sbs else frame0_full

        # Fine-tuned decoder → clean initial mask for this annotation
        init_mask = decoder_box_to_mask(
            still_model, frame0_eye, box_eye, w_eye, video_h, device,
        )
        print(f"    decoder mask coverage: {init_mask.mean()*100:.1f}%", flush=True)

        # Break long segments into sub-chunks to stay within GPU memory.
        # First sub-chunk seeds from the decoder mask; subsequent ones seed
        # from the last propagated mask to maintain tracking continuity.
        sub_starts  = list(range(seg_start, seg_end, MAX_CACHED_FRAMES))
        seed_mask   = init_mask
        seg_masks   = 0

        for sub_i, sub_start in enumerate(sub_starts):
            sub_end      = min(sub_start + MAX_CACHED_FRAMES, seg_end)
            sub_frames   = sub_end - sub_start
            n_sub_chunks = len(sub_starts)

            if n_sub_chunks > 1:
                print(f"    sub-chunk {sub_i+1}/{n_sub_chunks}  "
                      f"frames [{sub_start}–{sub_end})", flush=True)

            tmp_dir = tempfile.mkdtemp(prefix="track_ft_", dir=SCRATCH_TMP)
            try:
                # Write the sub-chunk's first frame for dummy init_state
                cap_tmp = cv2.VideoCapture(args.video)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, sub_start)
                _, f0 = cap_tmp.read()
                cap_tmp.release()
                f0_eye = f0[:, eye_x:eye_x + w_eye] if sbs else f0
                cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), f0_eye)
                state = predictor.init_state(video_path=tmp_dir)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            loader = VideoChunkLoader(
                args.video, image_size,
                sub_start, sub_frames, args.frame_step,
                sbs_eye_x=eye_x, sbs_eye_w=(w_eye if sbs else None),
            )
            state["images"]       = loader
            state["num_frames"]   = loader.n_frames
            state["video_height"] = video_h
            state["video_width"]  = w_eye

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

                t_enc = time.perf_counter()
                pre_encode_chunk(predictor, state, loader, device)
                torch.cuda.synchronize()
                print(f"      pre-encode {len(loader)} keyframes  "
                      f"{time.perf_counter()-t_enc:.2f}s", flush=True)

                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    mask=torch.from_numpy(seed_mask),
                )

                t_prop = time.perf_counter()
                for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                    mask       = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                    global_idx = sub_start + local_idx * args.frame_step
                    if mask.any():
                        np.save(os.path.join(masks_dir, f"{global_idx:06d}.npy"), mask)
                        seg_masks  += 1
                        mask_count += 1
                    seed_mask = mask  # carry forward for next sub-chunk

                try:
                    predictor.reset_state(state)
                except Exception:
                    pass

            torch.cuda.synchronize()
            print(f"      propagate {time.perf_counter()-t_prop:.2f}s  "
                  f"→ {seg_masks} masks so far", flush=True)
            del loader

        print(f"    seg total → {seg_masks} masks", flush=True)

    del still_model, predictor
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
    box_dur     = max(1, round(fps))
    t0          = time.time()

    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret: break

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

        for a in ann_list:
            if a["frame"] <= fidx < a["frame"] + box_dur:
                bx1, by1, bx2, by2 = [int(v) for v in a["box"]]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, f"init f{a['frame']}",
                            (bx1, max(by1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
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
