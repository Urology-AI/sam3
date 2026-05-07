#!/usr/bin/env python3
"""
compare_sam2track_vs_sam3.py
============================
Side-by-side comparison:
  Left  → SAM2 chunked reinit  (box from fine-tuned SAM3 detector, every SAM2_REINIT_INTERVAL frames)
  Right → SAM3 chunked reinit  (pre-computed .npy masks, every 150 frames)

Both sides re-detect with the same fine-tuned checkpoint at their respective
reinit intervals.  SAM2 actually re-initialises its tracker; SAM3 masks come
from a prior detect_and_segment_chunked.py run.

Usage:
  cd /sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
  python3 compare_sam2track_vs_sam3.py
"""

import os, sys, shutil, subprocess, tempfile, time
import cv2
import numpy as np
import torch
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
DET_CKPT  = os.path.join(SAM3_DIR,
             "detector_training/checkpoints/20260428_0942/checkpoint_best.pth")
SAM2_CKPT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

VIDEO_CLIP     = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
                  "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/"
                  "M_04222026081653_U013419042221253_2_002_0002-01_1290-1380.mp4")
MASKS_DIR      = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
                  "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/masks")
OUTPUT_DIR     = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
                  "comparison_sam2_vs_sam3")
SAM2_MASKS_TMP = os.path.join(OUTPUT_DIR, "_sam2_masks_tmp")

TEXT_QUERY           = "prostate gland"
DETECTOR_IMG_SIZE    = 1008
SAM2_REINIT_INTERVAL = 150   # match SAM3 side (150 frames ~2.5 s at 60 fps)
MASK_ALPHA           = 0.15
MASK_COLOR_BGR       = (0, 255, 0)
BOX_COLOR_BGR        = (0, 0, 255)
OUT_W_EACH           = 960
OUT_H_EACH           = 540

_DET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_DET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Streaming loader with chunk offset ────────────────────────────────────────
class VideoChunkLoader:
    """Lazy-load a contiguous slice [start_frame, start_frame+n_frames) of a video.

    state["images"][local_idx] → normalised tensor for global frame start_frame+local_idx.
    """
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path: str, image_size: int,
                 start_frame: int = 0, n_frames: int | None = None):
        self.image_size  = image_size
        self.start_frame = start_frame
        self._cap        = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(video_path)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames  = min(n_frames, total - start_frame) if n_frames else (total - start_frame)
        self._next_raw = start_frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, local_idx: int):
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


# ── Fine-tuned SAM3 detector ──────────────────────────────────────────────────

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
    lang_feats = txt["language_features"]
    lang_mask  = txt["language_mask"]
    print(f"  Detector loaded  epoch={raw.get('epoch', '?')}")
    return det, lang_feats, lang_mask


def detect_box(det, lang_feats, lang_mask, frame_bgr, w_vid, h_vid, device):
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt

    img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img   = TF.resize(img, [DETECTOR_IMG_SIZE, DETECTOR_IMG_SIZE])
    img_t = (TF.to_tensor(img) - _DET_MEAN) / _DET_STD
    img_t = img_t.unsqueeze(0).to(device)
    B     = 1
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


# ── Rendering helpers ─────────────────────────────────────────────────────────

def apply_mask(frame, mask_hw, color_bgr, alpha):
    if not mask_hw.any():
        return frame
    overlay = frame.copy()
    overlay[mask_hw.astype(bool)] = color_bgr
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def put_label(frame, text, pos=(16, 38)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_box_overlay(frame, box_abs, conf, color):
    x1, y1, x2, y2 = box_abs.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"reinit  conf={conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_TMP, exist_ok=True)
    os.makedirs(SAM2_MASKS_TMP, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Video metadata ─────────────────────────────────────────────────────────
    cap      = cv2.VideoCapture(VIDEO_CLIP)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    box_dur = max(1, round(fps))   # show box annotation for 1 s after each reinit
    chunk_starts = list(range(0, n_frames, SAM2_REINIT_INTERVAL))
    print(f"Clip:    {n_frames} frames  {w_vid}x{h_vid} @ {fps:.1f} fps")
    print(f"SAM2 reinit every {SAM2_REINIT_INTERVAL} frames ({SAM2_REINIT_INTERVAL/fps:.1f} s)"
          f"  →  {len(chunk_starts)} chunks")
    print(f"Output:  {OUT_W_EACH*2}x{OUT_H_EACH} side-by-side")

    # ── [1] Pre-detect boxes at all SAM2 reinit boundaries ────────────────────
    print(f"\n[1/5] Detecting prostate box at {len(chunk_starts)} reinit frames...")
    t0 = time.time()
    det, lang_feats, lang_mask = load_detector(device)
    reinit_boxes = {}   # chunk_start → (box_abs float32[4], conf float)
    cap = cv2.VideoCapture(VIDEO_CLIP)
    for cs in chunk_starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, f = cap.read()
        if not ret:
            continue
        box, conf = detect_box(det, lang_feats, lang_mask, f, w_vid, h_vid, device)
        reinit_boxes[cs] = (box, conf)
        print(f"  frame {cs:5d}  conf={conf:.3f}  "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
    cap.release()
    del det, lang_feats, lang_mask
    torch.cuda.empty_cache()
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── [2] SAM2 chunked propagation ───────────────────────────────────────────
    print(f"\n[2/5] Loading SAM2 large video predictor...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    print("  SAM2 large loaded")

    print(f"\n[3/5] SAM2 chunked propagation ({len(chunk_starts)} chunks)...")
    t0         = time.time()
    mask_count = 0
    image_size = getattr(predictor, "image_size", 1024)

    for ci, cs in enumerate(chunk_starts):
        if cs not in reinit_boxes:
            continue
        box_abs, conf = reinit_boxes[cs]
        chunk_size    = min(SAM2_REINIT_INTERVAL, n_frames - cs)

        # Grab the chunk-start frame for temp init dir
        cap_probe = cv2.VideoCapture(VIDEO_CLIP)
        cap_probe.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, init_frame = cap_probe.read()
        cap_probe.release()
        if not ret:
            continue

        # Init SAM2 state via 1-frame temp dir, then swap in chunk streaming loader
        tmp_dir = tempfile.mkdtemp(prefix="sam2_chunk_", dir=SCRATCH_TMP)
        try:
            cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), init_frame)
            state = predictor.init_state(video_path=tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        loader                  = VideoChunkLoader(VIDEO_CLIP, image_size, cs, chunk_size)
        state["images"]         = loader
        state["num_frames"]     = chunk_size
        state["video_height"]   = h_vid
        state["video_width"]    = w_vid

        with torch.inference_mode():
            # Pass box in absolute pixel coords; SAM2 normalises internally
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1, box=box_abs,
            )
            chunk_masks = 0
            for local_idx, _, mask_logits in predictor.propagate_in_video(state):
                mask = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                global_idx = cs + local_idx
                if mask.any():
                    np.save(os.path.join(SAM2_MASKS_TMP, f"{global_idx:06d}.npy"), mask)
                    chunk_masks += 1
                    mask_count  += 1
            try:
                predictor.reset_state(state)
            except Exception:
                pass

        del loader
        elapsed = time.time() - t0
        print(f"  chunk {ci+1}/{len(chunk_starts)}  "
              f"frames {cs}-{cs+chunk_size-1}  masks={chunk_masks}  "
              f"({elapsed:.0f}s elapsed)")

    del predictor
    torch.cuda.empty_cache()
    print(f"  Propagation done: {mask_count} SAM2 masks total")

    # ── [4] Render side-by-side ────────────────────────────────────────────────
    print(f"\n[4/5] Rendering side-by-side video...")
    out_raw  = os.path.join(OUTPUT_DIR, "_raw_comparison.mp4")
    out_path = os.path.join(OUTPUT_DIR, "comparison_sam2track_vs_sam3.mp4")
    writer   = cv2.VideoWriter(
        out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W_EACH * 2, OUT_H_EACH),
    )

    cap     = cv2.VideoCapture(VIDEO_CLIP)
    t0      = time.time()
    written = 0
    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # ── Left: SAM2 chunked reinit mask ───────────────────────────────────
        left = frame.copy()
        npy  = os.path.join(SAM2_MASKS_TMP, f"{fidx:06d}.npy")
        if os.path.exists(npy):
            m = np.load(npy)
            if m.shape[:2] != (h_vid, w_vid):
                m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
            left = apply_mask(left, m, MASK_COLOR_BGR, MASK_ALPHA)
        # Show box annotation for 1 s after each reinit
        for cs, (box_abs, conf) in reinit_boxes.items():
            if cs <= fidx < cs + box_dur:
                draw_box_overlay(left, box_abs, conf, BOX_COLOR_BGR)
                break
        put_label(left, f"SAM2 Tracker  (reinit every {SAM2_REINIT_INTERVAL/fps:.1f}s)")

        # ── Right: SAM3 chunked reinit mask ──────────────────────────────────
        right = frame.copy()
        npy   = os.path.join(MASKS_DIR, f"{fidx:06d}.npy")
        if os.path.exists(npy):
            m = np.load(npy)
            if m.shape[:2] != (h_vid, w_vid):
                m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
            right = apply_mask(right, m, MASK_COLOR_BGR, MASK_ALPHA)
        put_label(right, "SAM3 Chunked Reinit  (~2.5s interval)")

        # ── Combine ───────────────────────────────────────────────────────────
        l_s = cv2.resize(left,  (OUT_W_EACH, OUT_H_EACH))
        r_s = cv2.resize(right, (OUT_W_EACH, OUT_H_EACH))
        combined = np.concatenate([l_s, r_s], axis=1)
        cv2.line(combined, (OUT_W_EACH, 0), (OUT_W_EACH, OUT_H_EACH), (180, 180, 180), 2)
        writer.write(combined)
        written += 1

        if fidx % 600 == 0:
            elapsed = time.time() - t0
            print(f"  frame {fidx}/{n_frames}  ({written/max(elapsed,1e-3):.0f} fps)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  Render done in {elapsed:.1f}s  ({written/max(elapsed,1e-3):.0f} fps)")

    # ── [5] Re-mux to H.264 ───────────────────────────────────────────────────
    print(f"\n[5/5] Re-muxing to H.264...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[:500])
    if os.path.exists(out_raw):
        os.remove(out_raw)
    shutil.rmtree(SAM2_MASKS_TMP, ignore_errors=True)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Output:         {out_path}")
    print(f"  Frames written: {written}")
    print(f"  SAM2 masks:     {mask_count}/{n_frames}")
    print(f"  SAM2 chunks:    {len(reinit_boxes)} reinits every {SAM2_REINIT_INTERVAL} frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
