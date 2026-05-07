#!/usr/bin/env python3
"""
detect_segment_fast.py
======================
infer_prostate.py + two speedups:
  1. FRAME_STEP=3  — propagate every 3rd frame; hold last mask for skipped frames
  2. torch.compile  — inductor kernel fusion on the SAM2 video predictor

Pipeline per chunk (identical to infer_prostate.py):
  1. SAM3 fine-tuned detector    → bounding box
  2. Fine-tuned SAM2 decoder     → clean initial mask from noisy detector box
  3. add_new_mask                → seed video predictor with clean mask
  4. Base SAM2 video predictor   → propagate every FRAME_STEP frames

Run:
  python3 detect_segment_fast.py
"""

import os, sys, shutil, subprocess, tempfile, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM3_CKPT    = ("/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
                "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")
BPE_PATH     = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
DET_CKPT     = os.path.join(SAM3_DIR,
                "detector_training/checkpoints/20260428_0942/checkpoint_best.pth")
DECODER_CKPT = os.path.join(SAM3_DIR,
                "sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth")
SAM2_CKPT    = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

VIDEO_CLIP = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/"
              "M_04222026081653_U013419042221253_2_002_0002-01_1290-1380.mp4")

OUTPUT_DIR = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/"
              "fast_inference")

TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
SAM2_IMG_SIZE     = 1024
REINIT_INTERVAL   = 150    # frames between detector reinits (~2.5 s at 60 fps)
FRAME_STEP        = 3      # propagate every Nth frame; hold last mask for skipped frames
WARMUP_CHUNKS     = 1      # first N chunks trigger torch.compile JIT; not counted in timing
MASK_ALPHA        = 0.35
MASK_COLOR_BGR    = (0, 255, 0)
BOX_COLOR_BGR     = (0, 0, 255)

_DET_MEAN      = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_DET_STD       = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
_BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]


# ── SAM3 detector ──────────────────────────────────────────────────────────────

def load_detector(device):
    from sam3.model_builder import build_sam3_image_model
    det = build_sam3_image_model(
        bpe_path=BPE_PATH, checkpoint_path=SAM3_CKPT,
        load_from_HF=False, eval_mode=True, enable_segmentation=True,
    ).to(device)
    raw = torch.load(DET_CKPT, map_location="cpu", weights_only=True)
    det.load_state_dict(raw.get("state_dict", raw), strict=False)
    with torch.no_grad():
        txt = det.backbone.forward_text(captions=[TEXT_QUERY], device=device)
    print(f"  SAM3 detector loaded  epoch={raw.get('epoch','?')}")
    return det, txt["language_features"], txt["language_mask"]


def run_detector(det, lang_feats, lang_mask, frame_bgr, w_vid, h_vid, device):
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt

    img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img   = TF.resize(img, [DETECTOR_IMG_SIZE, DETECTOR_IMG_SIZE])
    img_t = (TF.to_tensor(img) - _DET_MEAN) / _DET_STD
    img_t = img_t.unsqueeze(0).to(device)
    B = 1
    with torch.no_grad():
        vis = det.backbone.forward_image(img_t)
        bb  = {**vis,
               "language_features": lang_feats.expand(-1, B, -1),
               "language_mask":     lang_mask.expand(B, -1)}
        fi  = FindStage(
            img_ids           = torch.arange(B, device=device),
            text_ids          = torch.zeros(B, dtype=torch.long, device=device),
            input_boxes       = torch.zeros(B, 0, 4, device=device),
            input_boxes_mask  = torch.zeros(B, 0, dtype=torch.bool, device=device),
            input_boxes_label = torch.zeros(B, 0, dtype=torch.long, device=device),
            input_points      = torch.zeros(B, 0, 2, device=device),
            input_points_mask = torch.zeros(B, 0, dtype=torch.bool, device=device),
        )
        gp  = Prompt(
            box_embeddings = torch.zeros(0, B, 4, device=device),
            box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
        )
        prompt, pm, bb = det._encode_prompt(bb, fi, gp)
        bb, eo, _      = det._run_encoder(bb, fi, prompt, pm)
        out = {"encoder_hidden_states": eo["encoder_hidden_states"]}
        out, _ = det._run_decoder(
            memory=out["encoder_hidden_states"], pos_embed=eo["pos_embed"],
            src_mask=eo["padding_mask"], out=out,
            prompt=prompt, prompt_mask=pm, encoder_out=eo,
        )
    scores  = out["pred_logits"][0].squeeze(-1).sigmoid()
    best_q  = scores.argmax().item()
    conf    = scores[best_q].item()
    cx, cy, bw, bh = out["pred_boxes"][0][best_q].cpu().tolist()
    box_abs = np.array(
        [(cx - bw/2)*w_vid, (cy - bh/2)*h_vid,
         (cx + bw/2)*w_vid, (cy + bh/2)*h_vid],
        dtype=np.float32,
    )
    return box_abs, conf


# ── Fine-tuned SAM2 decoder (still-image, for init mask) ──────────────────────

def load_finetuned_decoder(device):
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    raw = torch.load(DECODER_CKPT, map_location="cpu", weights_only=True)
    model.sam_mask_decoder.load_state_dict(raw.get("state_dict", raw))
    print(f"  Fine-tuned decoder loaded  epoch={raw.get('epoch','?')}  "
          f"val_iou={raw.get('val_iou', float('nan')):.3f}")
    return model


def decoder_box_to_mask(still_model, frame_bgr, box_abs, device):
    h_vid, w_vid = frame_bgr.shape[:2]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SAM2_IMG_SIZE, SAM2_IMG_SIZE))
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_t = torch.from_numpy(
        ((img.astype(np.float32) / 255.0) - img_mean) / img_std
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    sx = SAM2_IMG_SIZE / w_vid
    sy = SAM2_IMG_SIZE / h_vid
    box_1024 = torch.tensor(
        [box_abs[0]*sx, box_abs[1]*sy, box_abs[2]*sx, box_abs[3]*sy],
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
            low_res_masks, size=(h_vid, w_vid), mode="bilinear", align_corners=False,
        )
    return (mask_full[0, 0] > 0.0).cpu().numpy()


# ── Streaming loader with step support ────────────────────────────────────────

class VideoChunkLoader:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path, image_size, start_frame, n_frames, step=1):
        self.start_frame = start_frame
        self.step        = step
        self.image_size  = image_size
        self._cap        = cv2.VideoCapture(video_path)
        n_frames         = min(n_frames,
                               int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame)
        self.n_frames    = (n_frames + step - 1) // step
        self._next_raw   = start_frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx):
        raw = self.start_frame + local_idx * self.step
        if raw < self._next_raw:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
            self._next_raw = raw
        elif raw > self._next_raw:
            for _ in range(raw - self._next_raw):
                self._cap.grab()
            self._next_raw = raw
        ret, frame = self._cap.read()
        self._next_raw += 1
        if not ret:
            raise IndexError(f"Frame {raw} unreadable")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        t -= self._MEAN
        t /= self._STD
        return t

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    assert os.path.isfile(VIDEO_CLIP),    f"Video not found: {VIDEO_CLIP}"
    assert os.path.isfile(DET_CKPT),      f"Detector ckpt not found: {DET_CKPT}"
    assert os.path.isfile(DECODER_CKPT),  f"Decoder ckpt not found: {DECODER_CKPT}"
    assert torch.cuda.is_available(),      "CUDA required"

    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    for d in [OUTPUT_DIR, masks_dir, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FRAME_STEP={FRAME_STEP}  REINIT={REINIT_INTERVAL}")

    # ── [1] Video metadata ────────────────────────────────────────────────
    cap      = cv2.VideoCapture(VIDEO_CLIP)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_starts = list(range(0, n_frames, REINIT_INTERVAL))
    box_dur      = max(1, round(fps))
    print(f"\n[1] {n_frames} frames @ {fps:.1f} fps  →  {len(chunk_starts)} chunks")

    # ── [2] Load SAM3 detector ────────────────────────────────────────────
    print("\n[2] Loading SAM3 fine-tuned detector...")
    det, lang_feats, lang_mask = load_detector(device)

    # ── [3] Load fine-tuned SAM2 still-image decoder ─────────────────────
    print("\n[3] Loading fine-tuned SAM2 decoder (still-image)...")
    still_model = load_finetuned_decoder(device)

    # ── [4] Load base SAM2 video predictor + torch.compile ────────────────
    print("\n[4] Loading base SAM2 large video predictor + torch.compile...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
    predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
    predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"  Ready  (image_size={image_size}, compile=default/inductor)")

    # ── [5] Chunked detect → decode → track ──────────────────────────────
    print(f"\n[5] Chunked pipeline ({len(chunk_starts)} chunks, step={FRAME_STEP}, "
          f"warmup={WARMUP_CHUNKS} chunk(s))...")
    t0            = time.time()
    mask_count    = 0
    reinit_log    = {}
    infer_time_s  = 0.0  # pure GPU inference seconds (post-warmup)
    infer_frames  = 0    # raw video frames covered (post-warmup)

    cap = cv2.VideoCapture(VIDEO_CLIP)
    for ci, cs in enumerate(chunk_starts):
        chunk_size = min(REINIT_INTERVAL, n_frames - cs)
        is_warmup  = ci < WARMUP_CHUNKS

        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"  chunk {ci+1}: cannot read frame {cs}, skipping")
            continue

        tag = "WARMUP" if is_warmup else f"{ci+1:3d}/{len(chunk_starts)}"
        if is_warmup:
            print(f"  chunk [WARMUP]  frame {cs:5d}  (torch.compile JIT triggering...)",
                  flush=True)

        # ── GPU inference start ───────────────────────────────────────────
        if not is_warmup:
            torch.cuda.synchronize()
            t_infer = time.perf_counter()

        # Detect
        box_abs, conf = run_detector(det, lang_feats, lang_mask,
                                     frame_bgr, w_vid, h_vid, device)
        reinit_log[cs] = (box_abs, conf)
        if not is_warmup:
            print(f"  chunk {tag}  frame {cs:5d}  "
                  f"conf={conf:.3f}  box=[{box_abs[0]:.0f},{box_abs[1]:.0f},"
                  f"{box_abs[2]:.0f},{box_abs[3]:.0f}]", flush=True)

        # Fine-tuned decoder → clean initial mask
        init_mask = decoder_box_to_mask(still_model, frame_bgr, box_abs, device)

        # Init tracker state
        tmp_dir = tempfile.mkdtemp(prefix="fast_seg_", dir=SCRATCH_TMP)
        try:
            cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame_bgr)
            state = predictor.init_state(video_path=tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        loader                = VideoChunkLoader(VIDEO_CLIP, image_size,
                                                 cs, chunk_size, FRAME_STEP)
        state["images"]       = loader
        state["num_frames"]   = loader.n_frames
        state["video_height"] = h_vid
        state["video_width"]  = w_vid

        # Seed with fine-tuned mask, propagate every FRAME_STEP frames
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                mask=torch.from_numpy(init_mask),
            )
            chunk_masks = 0
            for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                mask       = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                global_idx = cs + local_idx * FRAME_STEP
                if mask.any():
                    np.save(os.path.join(masks_dir, f"{global_idx:06d}.npy"), mask)
                    chunk_masks += 1
                    mask_count  += 1

            try:
                predictor.reset_state(state)
            except Exception:
                pass

        # ── GPU inference end ─────────────────────────────────────────────
        if not is_warmup:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_infer
            infer_time_s += dt
            infer_frames += chunk_size
            print(f"    → {chunk_masks} masks  chunk_infer={dt:.2f}s  "
                  f"({time.time()-t0:.0f}s wall)", flush=True)

        del loader

    cap.release()

    video_dur_s = infer_frames / fps
    print(f"\n  ┌─ Pure inference summary (post-warmup) ──────────────────┐")
    print(f"  │  Video covered : {video_dur_s:.1f}s  ({infer_frames} raw frames, "
          f"{len(chunk_starts) - WARMUP_CHUNKS} chunks)")
    print(f"  │  Inference time: {infer_time_s:.2f}s")
    print(f"  │  Throughput    : {infer_frames / infer_time_s:.1f} FPS  "
          f"(source = {fps:.1f} FPS,  {video_dur_s / infer_time_s:.2f}× real-time)")
    print(f"  └─────────────────────────────────────────────────────────┘")
    del det, lang_feats, lang_mask, still_model, predictor
    torch.cuda.empty_cache()
    print(f"\n  Done: {mask_count} masks in {time.time()-t0:.1f}s")

    # ── [6] Render overlay (hold last mask for skipped frames) ────────────
    print("\n[6] Rendering overlay video...")
    out_raw  = os.path.join(OUTPUT_DIR, "_raw_overlay.mp4")
    out_path = os.path.join(OUTPUT_DIR, "overlay.mp4")
    writer   = cv2.VideoWriter(
        out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid),
    )
    cap       = cv2.VideoCapture(VIDEO_CLIP)
    last_mask = None
    t0        = time.time()

    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        npy = os.path.join(masks_dir, f"{fidx:06d}.npy")
        if os.path.exists(npy):
            m = np.load(npy)
            if m.shape[:2] != (h_vid, w_vid):
                m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
            last_mask = m

        if last_mask is not None and last_mask.any():
            overlay = frame.copy()
            overlay[last_mask.astype(bool)] = MASK_COLOR_BGR
            frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

        for cs, (box_abs, conf) in reinit_log.items():
            if cs <= fidx < cs + box_dur:
                x1, y1, x2, y2 = box_abs.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)
                lbl = f"detector  conf={conf:.2f}"
                cv2.putText(frame, lbl, (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, lbl, (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, BOX_COLOR_BGR, 2, cv2.LINE_AA)
                break

        writer.write(frame)
        if fidx % 600 == 0:
            print(f"  frame {fidx}/{n_frames}", flush=True)

    cap.release()
    writer.release()

    print("  Re-muxing to H.264...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("  ffmpeg stderr:", result.stderr[:400])
    os.remove(out_raw)
    print(f"  Overlay written in {time.time()-t0:.1f}s → {out_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Frames:    {n_frames} @ {fps:.1f} fps")
    print(f"  Chunks:    {len(chunk_starts)}  (reinit every {REINIT_INTERVAL/fps:.1f}s, "
          f"warmup={WARMUP_CHUNKS})")
    print(f"  Step:      {FRAME_STEP}  ({mask_count} masks computed, "
          f"{n_frames - mask_count} held from last)")
    print(f"  Inference: {infer_time_s:.2f}s for {video_dur_s:.1f}s of video  "
          f"({video_dur_s / infer_time_s:.2f}× real-time)")
    print(f"  Output:    {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
