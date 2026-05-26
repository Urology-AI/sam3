#!/usr/bin/env python3
"""
heatmap_fast.py
===============
Prostate segmentation heatmap using the full SAM3 + fine-tuned SAM2 pipeline.

Pipeline (identical to detect_segment_fast.py):
  1. SAM3 fine-tuned detector    → bounding box per chunk reinit
  2. Fine-tuned SAM2 decoder     → clean initial mask from detector box
  3. add_new_mask                → seed video predictor
  4. Base SAM2 video predictor   → propagate (FRAME_STEP=3, batch pre-encoded)

Rendering (new vs detect_segment_fast.py):
  - Heatmap centre: random foreground pixel from frame-0 segmentation mask
  - Heatmap colour: red (near centre) → green (mid) → blue (far), per pixel
  - Heatmap alpha: 60/255 (~0.24) — intentionally very transparent
  - Toggle schedule: off (1st third) / on+fade (2nd third) / off (3rd third)

Run:
  python3 heatmap_fast.py
"""

import bisect
import os
import sys
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sam3/
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

VIDEO_CLIP  = os.path.join(SAM3_DIR, "aua_videos", "media7_0_13.mp4")
OUTPUT_DIR  = os.path.join(SAM3_DIR, "aua_videos", "inferred_videos", "media7_heatmap")

# ── Hyperparameters ────────────────────────────────────────────────────────────
TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
SAM2_IMG_SIZE     = 1024
REINIT_INTERVAL   = 150    # frames between detector reinits
FRAME_STEP        = 3      # propagate every Nth frame
WARMUP_CHUNKS     = 1      # first N chunks trigger torch.compile JIT
MAX_CHUNKS        = None   # None = process entire video
ENCODE_BATCH_SIZE = 8

# Heatmap rendering
HEATMAP_ALPHA     = 120     # 0–255; intentionally low for transparency (~0.24)
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
    print(f"  SAM3 detector loaded  epoch={raw.get('epoch', '?')}")
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


# ── Fine-tuned SAM2 decoder ────────────────────────────────────────────────────

def load_finetuned_decoder(device):
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    raw = torch.load(DECODER_CKPT, map_location="cpu", weights_only=True)
    model.sam_mask_decoder.load_state_dict(raw.get("state_dict", raw))
    print(f"  Fine-tuned decoder loaded  epoch={raw.get('epoch', '?')}  "
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


# ── Batch pre-encoding ─────────────────────────────────────────────────────────

def pre_encode_chunk(predictor, state, loader, device, batch_size=ENCODE_BATCH_SIZE):
    n = len(loader)
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        frames = [loader[i].to(device).float() for i in range(batch_start, batch_end)]
        batch_tensor   = torch.stack(frames, dim=0)
        backbone_batch = predictor.forward_image(batch_tensor)
        for j, frame_idx in enumerate(range(batch_start, batch_end)):
            state["cached_features"][frame_idx] = (
                batch_tensor[j:j+1].clone(),
                {
                    "backbone_fpn":   [f[j:j+1].clone() for f in backbone_batch["backbone_fpn"]],
                    "vision_pos_enc": [p[j:j+1].clone() for p in backbone_batch["vision_pos_enc"]],
                },
            )
        del batch_tensor, backbone_batch


# ── VideoChunkLoader ───────────────────────────────────────────────────────────

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


# ── Heatmap ────────────────────────────────────────────────────────────────────

def build_heatmap_rgba(mask_float: np.ndarray,
                       center_xy: Tuple[float, float]) -> np.ndarray:
    """
    Build an RGBA heatmap (H, W, 4) from a float mask [0, 1].

    Foreground pixels (mask > 0.5):
        distance = 0 at center_xy  →  red   (R=255, B=0)
        distance = max             →  blue  (R=0,   B=255)
        green peaks at mid-distance.
    Alpha = HEATMAP_ALPHA for foreground, 0 outside.
    """
    H, W = mask_float.shape
    cx, cy = center_xy
    fg = mask_float > 0.5
    if not fg.any():
        return np.zeros((H, W, 4), dtype=np.uint8)

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    max_dist = float(dist[fg].max()) or 1.0
    t = np.clip(dist / max_dist, 0.0, 1.0)   # 0 = hot centre, 1 = cool edge

    R = ((1.0 - t) * 255).astype(np.uint8)
    G = (np.maximum(0.0, 1.0 - np.abs(t - 0.5) * 2) * 200).astype(np.uint8)
    B = (t * 255).astype(np.uint8)
    A = np.where(fg, HEATMAP_ALPHA, 0).astype(np.uint8)

    return np.stack([R, G, B, A], axis=-1)


def overlay_heatmap_on_frame(frame_bgr: np.ndarray,
                             heatmap_rgba: np.ndarray,
                             fade_alpha: float) -> np.ndarray:
    """
    Alpha-composite heatmap_rgba over frame_bgr, scaled by fade_alpha [0, 1].
    Returns BGR frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    a     = (heatmap_rgba[:, :, 3:4].astype(np.float32) / 255.0) * fade_alpha
    h_rgb = heatmap_rgba[:, :, :3].astype(np.float32)
    out   = (frame_rgb * (1.0 - a) + h_rgb * a).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def make_schedule(N: int) -> List[Tuple[bool, float]]:
    """
    1st third : heatmap OFF
    2nd third : heatmap ON  (5% fade-in + 5% fade-out at edges)
    3rd third : heatmap OFF
    """
    on_start = int(round(N / 3.0))
    on_end   = int(round(2 * N / 3.0))
    fade_len = max(1, int(N * 0.05))

    schedule = []
    for i in range(N):
        if i < on_start or i >= on_end:
            schedule.append((False, 0.0))
            continue
        rel  = i - on_start
        tail = on_end - 1 - i
        if rel < fade_len:
            alpha = rel / fade_len
        elif tail < fade_len:
            alpha = tail / fade_len
        else:
            alpha = 1.0
        schedule.append((True, float(alpha)))
    return schedule


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    assert os.path.isfile(VIDEO_CLIP),    f"Video not found: {VIDEO_CLIP}"
    assert os.path.isfile(DET_CKPT),      f"Detector ckpt not found: {DET_CKPT}"
    assert os.path.isfile(DECODER_CKPT),  f"Decoder ckpt not found: {DECODER_CKPT}"
    assert torch.cuda.is_available(),     "CUDA required"

    np.random.seed(42)

    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    for d in [OUTPUT_DIR, masks_dir, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FRAME_STEP={FRAME_STEP}  REINIT={REINIT_INTERVAL}")

    # ── [1] Video metadata ────────────────────────────────────────────────────
    cap      = cv2.VideoCapture(VIDEO_CLIP)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_starts = list(range(0, n_frames, REINIT_INTERVAL))
    if MAX_CHUNKS is not None:
        chunk_starts = chunk_starts[:MAX_CHUNKS]
    box_dur = max(1, round(fps))
    print(f"\n[1] {n_frames} frames @ {fps:.1f} fps  "
          f"({n_frames/fps:.1f}s)  →  {len(chunk_starts)} chunks")

    # ── [2] Load models ───────────────────────────────────────────────────────
    print("\n[2] Loading SAM3 fine-tuned detector...")
    det, lang_feats, lang_mask = load_detector(device)

    print("\n[3] Loading fine-tuned SAM2 decoder (still-image)...")
    still_model = load_finetuned_decoder(device)

    print("\n[4] Loading base SAM2 large video predictor + torch.compile...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
    predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
    predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"  Ready  (image_size={image_size})")

    # ── [5] Chunked detect → decode → track ──────────────────────────────────
    print(f"\n[5] Chunked pipeline ({len(chunk_starts)} chunks, step={FRAME_STEP}, "
          f"warmup={WARMUP_CHUNKS})...")
    t0           = time.time()
    mask_count   = 0
    reinit_log   = {}        # frame_idx → (box_abs, conf)
    frame0_mask  = None      # bool array (H, W), frame 0 foreground mask

    cap = cv2.VideoCapture(VIDEO_CLIP)
    for ci, cs in enumerate(chunk_starts):
        chunk_size = min(REINIT_INTERVAL, n_frames - cs)
        is_warmup  = ci < WARMUP_CHUNKS

        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"  chunk {ci+1}: cannot read frame {cs}, skipping")
            continue

        if is_warmup:
            print(f"  chunk [WARMUP]  frame {cs:5d}  (torch.compile JIT...)", flush=True)

        box_abs, conf = run_detector(det, lang_feats, lang_mask,
                                     frame_bgr, w_vid, h_vid, device)
        reinit_log[cs] = (box_abs, conf)

        if not is_warmup:
            print(f"  chunk {ci+1:3d}/{len(chunk_starts)}  frame {cs:5d}  "
                  f"conf={conf:.3f}  box=[{box_abs[0]:.0f},{box_abs[1]:.0f},"
                  f"{box_abs[2]:.0f},{box_abs[3]:.0f}]", flush=True)

        init_mask = decoder_box_to_mask(still_model, frame_bgr, box_abs, device)

        # Keep frame-0 mask for heatmap centre selection
        if ci == 0:
            frame0_mask = init_mask

        tmp_dir = tempfile.mkdtemp(prefix="heatmap_seg_", dir=SCRATCH_TMP)
        try:
            cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), frame_bgr)
            state = predictor.init_state(video_path=tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        loader              = VideoChunkLoader(VIDEO_CLIP, image_size,
                                               cs, chunk_size, FRAME_STEP)
        state["images"]     = loader
        state["num_frames"] = loader.n_frames
        state["video_height"] = h_vid
        state["video_width"]  = w_vid

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pre_encode_chunk(predictor, state, loader, device)
            predictor.add_new_mask(
                inference_state=state, frame_idx=0, obj_id=1,
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

        if not is_warmup:
            print(f"    → {chunk_masks} masks  ({time.time()-t0:.0f}s wall)", flush=True)

        del loader

    cap.release()
    del det, lang_feats, lang_mask, still_model, predictor
    torch.cuda.empty_cache()
    print(f"\n  Segmentation done: {mask_count} keyframe masks in {time.time()-t0:.1f}s")

    # ── [6] Heatmap centre: random foreground pixel from frame 0 ─────────────
    if frame0_mask is not None and frame0_mask.any():
        ys_fg, xs_fg = np.where(frame0_mask)
        pick = np.random.randint(len(xs_fg))
        cx, cy = float(xs_fg[pick]), float(ys_fg[pick])
        print(f"\n[6] Heatmap centre: random fg pixel from frame 0 → ({cx:.0f}, {cy:.0f})")
    else:
        cx, cy = w_vid / 2.0, h_vid / 2.0
        print(f"\n[6] Frame-0 mask empty — using frame centre ({cx:.0f}, {cy:.0f})")

    # ── [7] Load keyframe masks ───────────────────────────────────────────────
    keyframe_masks = {}
    for fname in sorted(os.listdir(masks_dir)):
        if fname.endswith(".npy"):
            fidx_k = int(os.path.splitext(fname)[0])
            m = np.load(os.path.join(masks_dir, fname)).astype(np.float32)
            if m.shape[:2] != (h_vid, w_vid):
                m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
            keyframe_masks[fidx_k] = m
    kf_idxs = sorted(keyframe_masks.keys())
    print(f"  {len(kf_idxs)} keyframe masks loaded")

    def get_mask_at(fidx):
        if not kf_idxs:
            return None
        pos = bisect.bisect_left(kf_idxs, fidx)
        if pos == len(kf_idxs):
            return keyframe_masks[kf_idxs[-1]]
        if pos == 0 or kf_idxs[pos] == fidx:
            return keyframe_masks[kf_idxs[pos]]
        prev_idx = kf_idxs[pos - 1]
        next_idx = kf_idxs[pos]
        alpha = (fidx - prev_idx) / (next_idx - prev_idx)
        return (1.0 - alpha) * keyframe_masks[prev_idx] + alpha * keyframe_masks[next_idx]

    # ── [8] Render ────────────────────────────────────────────────────────────
    print("\n[7] Rendering heatmap overlay...")
    out_raw  = os.path.join(OUTPUT_DIR, "_raw_heatmap.mp4")
    out_path = os.path.join(OUTPUT_DIR, "heatmap_overlay.mp4")
    writer   = cv2.VideoWriter(
        out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid),
    )

    cap = cv2.VideoCapture(VIDEO_CLIP)
    t0  = time.time()
    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        mask_f = get_mask_at(fidx)
        if mask_f is not None:
            hm = build_heatmap_rgba(mask_f, (cx, cy))
            frame = overlay_heatmap_on_frame(frame, hm, 1.0)

        writer.write(frame)
        if fidx % 500 == 0:
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

    print(f"\n{'='*60}")
    print(f"DONE  ({time.time()-t0:.1f}s render)")
    print(f"{'='*60}")
    print(f"  Video    : {VIDEO_CLIP}")
    print(f"  Frames   : {n_frames} @ {fps:.1f} fps")
    print(f"  Masks    : {mask_count} keyframes  (FRAME_STEP={FRAME_STEP})")
    print(f"  Heatmap  : alpha={HEATMAP_ALPHA}/255  centre=({cx:.0f}, {cy:.0f})")
    print(f"  Output   : {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
