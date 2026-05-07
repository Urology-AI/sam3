#!/usr/bin/env python3
"""
infer_prostate.py
=================
Full prostate segmentation pipeline:

  [1] (Optional) ffmpeg clip from a longer source video
  [2] SAM3 fine-tuned detector  → bounding box every REINIT_INTERVAL frames
  [3] Fine-tuned SAM2 decoder   → clean initial mask from the (noisy) detector box
  [4] Base SAM2 video predictor → propagate mask through REINIT_INTERVAL frames
  Repeat [2-4] for each chunk, then render an overlay video.

Two separate decoder instances, as designed:
  • still-image model  (build_sam2)        loads fine-tuned decoder weights  → init mask
  • video predictor    (build_sam2_video_predictor) uses base decoder weights → propagation

Usage
-----
  cd /sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3

  python3 infer_prostate.py \\
      --video /path/to/source.mp4 \\
      --detector_ckpt detector_training/checkpoints/20260428_0942/checkpoint_best.pth \\
      --decoder_ckpt  sam2_decoder_training/checkpoints/<run_id>/checkpoint_best.pth \\
      --clip_start 3959 --clip_end 3969 \\
      --output_dir short_clips \\
      --reinit_interval 150
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM3_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR    = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM3_CKPT = ("/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
             "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")
BPE_PATH  = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
SAM2_IMG_SIZE     = 1024

_DET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_DET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
_BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]


# ── SAM3 detector ─────────────────────────────────────────────────────────────

def load_detector(detector_ckpt: str, device):
    from sam3.model_builder import build_sam3_image_model
    det = build_sam3_image_model(
        bpe_path=BPE_PATH, checkpoint_path=SAM3_CKPT,
        load_from_HF=False, eval_mode=True, enable_segmentation=True,
    ).to(device)
    raw = torch.load(detector_ckpt, map_location="cpu", weights_only=True)
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
        fi = FindStage(
            img_ids           = torch.arange(B, device=device),
            text_ids          = torch.zeros(B, dtype=torch.long, device=device),
            input_boxes       = torch.zeros(B, 0, 4, device=device),
            input_boxes_mask  = torch.zeros(B, 0, dtype=torch.bool, device=device),
            input_boxes_label = torch.zeros(B, 0, dtype=torch.long, device=device),
            input_points      = torch.zeros(B, 0, 2, device=device),
            input_points_mask = torch.zeros(B, 0, dtype=torch.bool, device=device),
        )
        gp = Prompt(
            box_embeddings = torch.zeros(0, B, 4, device=device),
            box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
        )
        prompt, pm, bb = det._encode_prompt(bb, fi, gp)
        bb, eo, _      = det._run_encoder(bb, fi, prompt, pm)
        out = {"encoder_hidden_states": eo["encoder_hidden_states"]}
        out, _ = det._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=eo["pos_embed"],
            src_mask=eo["padding_mask"],
            out=out, prompt=prompt, prompt_mask=pm, encoder_out=eo,
        )
    scores  = out["pred_logits"][0].squeeze(-1).sigmoid()
    best_q  = scores.argmax().item()
    conf    = scores[best_q].item()
    cx, cy, bw, bh = out["pred_boxes"][0][best_q].cpu().tolist()
    box_abs = np.array(
        [(cx - bw/2) * w_vid, (cy - bh/2) * h_vid,
         (cx + bw/2) * w_vid, (cy + bh/2) * h_vid],
        dtype=np.float32,
    )
    return box_abs, conf


# ── Fine-tuned SAM2 decoder (still-image mode) ────────────────────────────────

def load_finetuned_decoder(decoder_ckpt: str, device):
    """Build base SAM2 and override just the mask decoder with fine-tuned weights."""
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()
    raw = torch.load(decoder_ckpt, map_location="cpu", weights_only=True)
    model.sam_mask_decoder.load_state_dict(raw.get("state_dict", raw))
    print(f"  Fine-tuned SAM2 decoder loaded  epoch={raw.get('epoch', '?')}  "
          f"val_iou={raw.get('val_iou', '?'):.3f}" if isinstance(raw.get('val_iou'), float)
          else f"  Fine-tuned SAM2 decoder loaded  epoch={raw.get('epoch', '?')}")
    return model


def decoder_box_to_mask(still_model, frame_bgr, box_abs, device):
    """
    Run the fine-tuned still-image SAM2 decoder.
    box_abs: xyxy in original frame pixel coords.
    Returns a (H, W) boolean numpy mask at original frame resolution.
    """
    h_vid, w_vid = frame_bgr.shape[:2]

    # Preprocess image to 1024×1024
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SAM2_IMG_SIZE, SAM2_IMG_SIZE))
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_t = torch.from_numpy(
        ((img.astype(np.float32) / 255.0) - img_mean) / img_std
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # Scale box to 1024 space
    sx = SAM2_IMG_SIZE / w_vid
    sy = SAM2_IMG_SIZE / h_vid
    box_1024 = torch.tensor(
        [box_abs[0]*sx, box_abs[1]*sy, box_abs[2]*sx, box_abs[3]*sy],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)   # (1, 4)

    with torch.no_grad():
        # Encode image (frozen)
        backbone_out = still_model.forward_image(img_t)
        _, vision_feats, _, _ = still_model._prepare_backbone_features(backbone_out)
        if getattr(still_model, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + still_model.no_mem_embed
        B = 1
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], _BB_FEAT_SIZES[::-1])
        ][::-1]
        image_embed   = feats[-1]
        high_res_feats = feats[:-1]

        # Encode box as point pair (labels 2=TL, 3=BR)
        box_coords = box_1024.reshape(1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
        sparse_emb, dense_emb = still_model.sam_prompt_encoder(
            points=(box_coords, box_labels), boxes=None, masks=None,
        )

        # Decode with fine-tuned decoder
        low_res_masks, _, _, _ = still_model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=still_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        # Upsample to original video resolution
        mask_full = F.interpolate(
            low_res_masks, size=(h_vid, w_vid), mode="bilinear", align_corners=False,
        )
    return (mask_full[0, 0] > 0.0).cpu().numpy()   # (H, W) bool


# ── Streaming video loader for SAM2 tracker ───────────────────────────────────

class VideoChunkLoader:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path, image_size, start_frame, n_frames):
        self.image_size  = image_size
        self.start_frame = start_frame
        self.n_frames    = n_frames
        self._cap        = cv2.VideoCapture(video_path)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._next_raw   = start_frame

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx):
        raw = self.start_frame + local_idx
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


def init_tracker_streaming(predictor, video_path, chunk_frame_bgr, w_vid, h_vid):
    """Init SAM2 video predictor with 1-frame temp dir, swap in chunk loader."""
    tmp = tempfile.mkdtemp(prefix="sam2_init_", dir=SCRATCH_TMP)
    try:
        cv2.imwrite(os.path.join(tmp, "000000.jpg"), chunk_frame_bgr)
        state = predictor.init_state(video_path=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    state["video_height"] = h_vid
    state["video_width"]  = w_vid
    return state


# ── Rendering helpers ─────────────────────────────────────────────────────────

def apply_mask(frame, mask_hw, color_bgr, alpha):
    if not mask_hw.any():
        return frame
    overlay = frame.copy()
    overlay[mask_hw] = color_bgr
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def put_label(frame, text, pos=(14, 36)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",          required=True,  help="Source video path")
    p.add_argument("--detector_ckpt",  required=True,  help="SAM3 fine-tuned detector .pth")
    p.add_argument("--decoder_ckpt",   required=True,  help="SAM2 fine-tuned decoder .pth")
    p.add_argument("--clip_start",     type=int, default=None, help="Clip start (seconds)")
    p.add_argument("--clip_end",       type=int, default=None, help="Clip end (seconds)")
    p.add_argument("--reinit_interval",type=int, default=150,  help="Frames between reinits")
    p.add_argument("--output_dir",     default=None,   help="Output directory (auto if omitted)")
    p.add_argument("--mask_color",     default="0,255,0", help="R,G,B overlay colour")
    p.add_argument("--mask_alpha",     type=float, default=0.35)
    p.add_argument("--conf_threshold", type=float, default=0.10,
                   help="Warn if detector confidence falls below this")
    return p.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    mask_color_bgr = tuple(int(x) for x in reversed(args.mask_color.split(",")))

    # ── [1] Optional clip ─────────────────────────────────────────────────────
    os.makedirs(SCRATCH_TMP, exist_ok=True)
    video_path = args.video
    if args.clip_start is not None and args.clip_end is not None:
        stem     = os.path.splitext(os.path.basename(args.video))[0]
        out_dir  = args.output_dir or os.path.join(SAM3_DIR, "short_clips")
        os.makedirs(out_dir, exist_ok=True)
        clip_path = os.path.join(out_dir, f"{stem}_{args.clip_start}-{args.clip_end}.mp4")
        if not os.path.exists(clip_path):
            print(f"\n[1/6] Clipping {args.clip_start}-{args.clip_end}s → {clip_path}")
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(args.clip_start), "-to", str(args.clip_end),
                "-i", args.video,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                clip_path,
            ], check=True, capture_output=True)
        else:
            print(f"\n[1/6] Clip exists: {clip_path}")
        video_path = clip_path
    else:
        out_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.video)),
            os.path.splitext(os.path.basename(args.video))[0] + "_prostate_seg",
        )
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[1/6] No clipping — using {video_path} directly")

    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # ── Video metadata ─────────────────────────────────────────────────────────
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_starts = list(range(0, n_frames, args.reinit_interval))
    box_dur = max(1, round(fps))
    print(f"Clip: {n_frames} frames  {w_vid}x{h_vid} @ {fps:.1f} fps")
    print(f"Reinit every {args.reinit_interval} frames (~{args.reinit_interval/fps:.1f}s)"
          f"  →  {len(chunk_starts)} chunks")

    # ── [2] Load SAM3 detector ─────────────────────────────────────────────────
    print("\n[2/6] Loading SAM3 fine-tuned detector...")
    det, lang_feats, lang_mask = load_detector(args.detector_ckpt, device)

    # ── [3] Load fine-tuned SAM2 still-image decoder ──────────────────────────
    print("\n[3/6] Loading fine-tuned SAM2 decoder (still-image)...")
    still_model = load_finetuned_decoder(args.decoder_ckpt, device)

    # ── [4] Load base SAM2 video predictor (for tracking) ─────────────────────
    print("\n[4/6] Loading base SAM2 large video predictor (tracker)...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    print("  Base SAM2 tracker loaded")

    # ── [5] Chunked detect → decode → track ───────────────────────────────────
    print(f"\n[5/6] Chunked pipeline ({len(chunk_starts)} chunks)...")
    t0         = time.time()
    mask_count = 0
    reinit_log = {}   # chunk_start → (box_abs, conf)

    cap = cv2.VideoCapture(video_path)

    for ci, cs in enumerate(chunk_starts):
        chunk_size = min(args.reinit_interval, n_frames - cs)

        # Read chunk-start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"  chunk {ci+1}: could not read frame {cs}, skipping")
            continue

        # ── Detect ────────────────────────────────────────────────────────────
        box_abs, conf = run_detector(det, lang_feats, lang_mask,
                                     frame_bgr, w_vid, h_vid, device)
        reinit_log[cs] = (box_abs, conf)
        if conf < args.conf_threshold:
            print(f"  chunk {ci+1:3d}  frame {cs:5d}  "
                  f"conf={conf:.3f} [LOW]  box=[{box_abs[0]:.0f},{box_abs[1]:.0f},"
                  f"{box_abs[2]:.0f},{box_abs[3]:.0f}]")
        else:
            print(f"  chunk {ci+1:3d}  frame {cs:5d}  "
                  f"conf={conf:.3f}  box=[{box_abs[0]:.0f},{box_abs[1]:.0f},"
                  f"{box_abs[2]:.0f},{box_abs[3]:.0f}]")

        # ── Fine-tuned decoder → clean initial mask ────────────────────────────
        init_mask = decoder_box_to_mask(still_model, frame_bgr, box_abs, device)
        # init_mask: (h_vid, w_vid) bool

        # ── Init SAM2 tracker with streaming loader ────────────────────────────
        state = init_tracker_streaming(predictor, video_path, frame_bgr, w_vid, h_vid)
        loader = VideoChunkLoader(video_path, image_size, cs, chunk_size)
        state["images"]     = loader
        state["num_frames"] = chunk_size

        # ── Seed tracker with fine-tuned mask (not a box) ─────────────────────
        with torch.inference_mode():
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                mask=torch.from_numpy(init_mask),
            )

            # ── Propagate chunk ────────────────────────────────────────────────
            chunk_masks = 0
            for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                global_idx = cs + local_idx
                if mask.any():
                    np.save(os.path.join(masks_dir, f"{global_idx:06d}.npy"), mask)
                    chunk_masks += 1
                    mask_count  += 1

            try:
                predictor.reset_state(state)
            except Exception:
                pass

        del loader
        elapsed = time.time() - t0
        print(f"    → {chunk_masks} masks  ({elapsed:.0f}s total)")

    cap.release()
    del det, lang_feats, lang_mask, still_model, predictor
    torch.cuda.empty_cache()
    print(f"  Propagation done: {mask_count} masks across {len(chunk_starts)} chunks")

    # ── [6] Render overlay video ───────────────────────────────────────────────
    print("\n[6/6] Rendering overlay video...")
    out_raw  = os.path.join(out_dir, "_raw_overlay.mp4")
    out_path = os.path.join(out_dir, "overlay.mp4")
    writer   = cv2.VideoWriter(
        out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid),
    )
    cap     = cv2.VideoCapture(video_path)
    t0      = time.time()
    written = 0
    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        npy = os.path.join(masks_dir, f"{fidx:06d}.npy")
        if os.path.exists(npy):
            m = np.load(npy)
            if m.shape[:2] != (h_vid, w_vid):
                m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
            frame = apply_mask(frame, m.astype(bool), mask_color_bgr, args.mask_alpha)

        # Flash detector box for 1 s at each reinit
        for cs_r, (box_r, conf_r) in reinit_log.items():
            if cs_r <= fidx < cs_r + box_dur:
                x1, y1, x2, y2 = box_r.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                lbl = f"detector  conf={conf_r:.2f}"
                cv2.putText(frame, lbl, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, lbl, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                break

        put_label(frame, f"SAM3 det + SAM2 seg  reinit={args.reinit_interval}f")
        writer.write(frame)
        written += 1

        if fidx % 500 == 0:
            print(f"  frame {fidx}/{n_frames}  ({written/max(time.time()-t0,1e-3):.0f} fps)")

    cap.release()
    writer.release()

    print("\nRe-muxing to H.264...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[:400])
    if os.path.exists(out_raw):
        os.remove(out_raw)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Overlay:  {out_path}")
    print(f"  Masks:    {masks_dir}/  ({mask_count} frames)")
    print(f"  Chunks:   {len(chunk_starts)}  every {args.reinit_interval} frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
