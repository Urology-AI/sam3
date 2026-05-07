#!/usr/bin/env python3
"""
benchmark_sam2large.py
======================
Compares full propagation (every frame) vs partial propagation (every Nth frame)
using SAM2-large + torch.compile.

For each chunk:
  1. SAM3 detector            → bounding box
  2. Full propagation  (step=1)          → mask every frame   → reference
  3. Partial propagation (step=FRAME_STEP) → mask every Nth frame → candidate

Summary reports:
  - FPS for full vs partial
  - IoU of partial vs full (at overlapping frames)
  - IoU of both vs AutoSam fine-tuned masks

Run:
  bash run_benchmark.sh
"""

import os, sys, shutil, tempfile, time
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
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

VIDEO_CLIP = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/"
              "M_04222026081653_U013419042221253_2_002_0002-01_1290-1380.mp4")

FINETUNED_MASKS_DIR = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
                       "comparison_sam2_vs_sam3/masks")

OUTPUT_DIR   = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
                "benchmark_sam2large_compiled")
FULL_DIR     = os.path.join(OUTPUT_DIR, "masks_full")
PARTIAL_DIR  = os.path.join(OUTPUT_DIR, "masks_partial")

TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
REINIT_INTERVAL   = 150   # frames between detector reinits
FRAME_STEP        = 3     # partial: process every Nth frame (3 → ~20 FPS from 60 FPS source)
MAX_CHUNKS        = 10    # None = all chunks
WARMUP_CHUNKS     = 1     # exclude from FPS (torch.compile traces on chunk 1)

_DET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_DET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ── Streaming loader ───────────────────────────────────────────────────────────

class VideoChunkLoader:
    """
    Maps local SAM2 indices 0, 1, 2, ... to video frames
    start_frame, start_frame+step, start_frame+2*step, ...
    """
    _MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, video_path, image_size, start_frame=0, n_frames=None, step=1):
        self.image_size  = image_size
        self.start_frame = start_frame
        self.step        = step
        self._cap        = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(video_path)
        total         = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_available = total - start_frame
        if n_frames is None:
            n_frames = raw_available
        else:
            n_frames = min(n_frames, raw_available)
        # number of frames SAM2 will request (local indices 0..n_logical-1)
        self.n_frames  = (n_frames + step - 1) // step
        self._next_raw = start_frame
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
    print(f"  Detector loaded  epoch={raw.get('epoch','?')}  "
          f"val_iou={raw.get('val_iou', float('nan')):.3f}")
    return det, txt["language_features"], txt["language_mask"]


def detect_box(det, lang_feats, lang_mask, frame_bgr, w_vid, h_vid, device):
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
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best_q = scores.argmax().item()
    conf   = scores[best_q].item()
    cx, cy, bw, bh = out["pred_boxes"][0][best_q].cpu().tolist()
    box_abs = np.array(
        [(cx - bw/2)*w_vid, (cy - bh/2)*h_vid,
         (cx + bw/2)*w_vid, (cy + bh/2)*h_vid],
        dtype=np.float32,
    )
    return box_abs, conf


# ── Propagation helper ─────────────────────────────────────────────────────────

def run_chunk(predictor, init_frame_bgr, box_abs, video_path,
              image_size, h_vid, w_vid, cs, chunk_size, step,
              masks_dir, device):
    """
    Init a fresh SAM2 state for one chunk, propagate with the given step,
    save masks, return (prop_ms, n_logical_frames, global_idx_list).
    global_idx = cs + local_idx * step.
    """
    tmp_dir = tempfile.mkdtemp(prefix="sam2bench_", dir=SCRATCH_TMP)
    try:
        cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), init_frame_bgr)
        state = predictor.init_state(video_path=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    loader                = VideoChunkLoader(video_path, image_size, cs, chunk_size, step)
    state["images"]       = loader
    state["num_frames"]   = loader.n_frames
    state["video_height"] = h_vid
    state["video_width"]  = w_vid

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=box_abs,
        )

    saved = []
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for local_idx, _, mask_logits in predictor.propagate_in_video(state):
            mask       = (mask_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
            global_idx = cs + local_idx * step
            if mask.any():
                np.save(os.path.join(masks_dir, f"{global_idx:06d}.npy"), mask)
                saved.append(global_idx)

    torch.cuda.synchronize()
    prop_ms = (time.time() - t0) * 1000

    n_logical = loader.n_frames
    try:
        predictor.reset_state(state)
    except Exception:
        pass
    del loader

    return prop_ms, n_logical, saved


# ── Metrics ────────────────────────────────────────────────────────────────────

def iou(pred, ref):
    p = pred.astype(bool)
    r = ref.astype(bool)
    inter = (p & r).sum()
    union = (p | r).sum()
    return float(inter / union) if union > 0 else 1.0


def dice_iou(pred, ref):
    p = pred.astype(bool)
    r = ref.astype(bool)
    inter = (p & r).sum()
    dice  = 2 * inter / (p.sum() + r.sum() + 1e-8)
    union = (p | r).sum()
    return float(dice), float(inter / union) if union > 0 else 1.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    assert os.path.isfile(VIDEO_CLIP), f"Video not found: {VIDEO_CLIP}"
    assert os.path.isfile(DET_CKPT),   f"Detector ckpt not found: {DET_CKPT}"
    assert os.path.isfile(SAM2_CKPT),  f"SAM2 ckpt not found: {SAM2_CKPT}"
    assert torch.cuda.is_available(),   "CUDA required"

    for d in [OUTPUT_DIR, FULL_DIR, PARTIAL_DIR, SCRATCH_TMP]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"FRAME_STEP={FRAME_STEP}  MAX_CHUNKS={MAX_CHUNKS}  REINIT={REINIT_INTERVAL}")

    # ── [1] Video metadata ────────────────────────────────────────────────
    cap      = cv2.VideoCapture(VIDEO_CLIP)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    chunk_starts  = list(range(0, n_frames, REINIT_INTERVAL))
    chunks_to_run = chunk_starts if MAX_CHUNKS is None else chunk_starts[:MAX_CHUNKS]
    print(f"\n[1] {n_frames} frames @ {fps:.1f} fps  →  "
          f"{len(chunks_to_run)}/{len(chunk_starts)} chunks to run")

    # ── [2] Detect boxes ──────────────────────────────────────────────────
    print(f"\n[2] Detecting at {len(chunks_to_run)} reinit frames...")
    det, lang_feats, lang_mask = load_detector(device)
    reinit_boxes = {}
    cap = cv2.VideoCapture(VIDEO_CLIP)
    for cs in chunks_to_run:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        ret, f = cap.read()
        if not ret:
            continue
        box_abs, conf = detect_box(det, lang_feats, lang_mask, f, w_vid, h_vid, device)
        reinit_boxes[cs] = (box_abs, conf, f)   # keep frame for init_state reuse
        print(f"  frame {cs:5d}  conf={conf:.3f}", flush=True)
    cap.release()
    del det, lang_feats, lang_mask
    torch.cuda.empty_cache()

    # ── [3] Load SAM2-large + compile ─────────────────────────────────────
    print(f"\n[3] Loading SAM2-large + torch.compile...")
    from sam2.build_sam import build_sam2_video_predictor
    predictor  = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    predictor.eval()
    image_size = getattr(predictor, "image_size", 1024)
    predictor.image_encoder    = torch.compile(predictor.image_encoder,    mode="default")
    predictor.memory_attention = torch.compile(predictor.memory_attention, mode="default")
    predictor.sam_mask_decoder = torch.compile(predictor.sam_mask_decoder, mode="default")
    print(f"  Ready  (image_size={image_size})")

    # ── [4] Per-chunk: full then partial propagation ───────────────────────
    print(f"\n[4] Running full (step=1) then partial (step={FRAME_STEP}) per chunk...")

    full_ms_list    = []   # ms/frame for full propagation    (post-warmup)
    partial_ms_list = []   # ms/frame for partial propagation (post-warmup)
    iou_partial_vs_full = []   # IoU(partial, full) at overlapping frames

    t_all = time.time()
    for ci, cs in enumerate(chunks_to_run):
        if cs not in reinit_boxes:
            continue
        box_abs, conf, init_frame_bgr = reinit_boxes[cs]
        chunk_size = min(REINIT_INTERVAL, n_frames - cs)

        # ── Full propagation ──────────────────────────────────────────────
        prop_ms_full, n_full, _ = run_chunk(
            predictor, init_frame_bgr, box_abs, VIDEO_CLIP,
            image_size, h_vid, w_vid, cs, chunk_size, step=1,
            masks_dir=FULL_DIR, device=device,
        )
        ms_per_frame_full = prop_ms_full / chunk_size

        # ── Partial propagation ───────────────────────────────────────────
        prop_ms_part, n_part, part_indices = run_chunk(
            predictor, init_frame_bgr, box_abs, VIDEO_CLIP,
            image_size, h_vid, w_vid, cs, chunk_size, step=FRAME_STEP,
            masks_dir=PARTIAL_DIR, device=device,
        )
        # effective ms per original video frame (not per processed frame)
        ms_per_frame_part = prop_ms_part / chunk_size

        # ── IoU: partial vs full at overlapping frames ────────────────────
        chunk_ious = []
        for gidx in part_indices:
            fp = os.path.join(PARTIAL_DIR, f"{gidx:06d}.npy")
            ff = os.path.join(FULL_DIR,    f"{gidx:06d}.npy")
            if os.path.exists(fp) and os.path.exists(ff):
                pm = np.load(fp)
                fm = np.load(ff)
                if fm.shape != pm.shape:
                    fm = cv2.resize(fm, (pm.shape[1], pm.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                chunk_ious.append(iou(pm, fm))
        mean_chunk_iou = float(np.mean(chunk_ious)) if chunk_ious else float("nan")

        is_warmup = ci < WARMUP_CHUNKS
        if not is_warmup:
            full_ms_list.append(ms_per_frame_full)
            partial_ms_list.append(ms_per_frame_part)
            iou_partial_vs_full.extend(chunk_ious)

        print(f"  chunk {ci+1:2d}/{len(chunks_to_run)}  frames {cs}-{cs+chunk_size-1}  "
              f"conf={conf:.3f}  "
              f"full={ms_per_frame_full:.1f}ms/f  "
              f"partial={ms_per_frame_part:.1f}ms/f  "
              f"IoU(part|full)={mean_chunk_iou:.3f}"
              f"{'  [warmup]' if is_warmup else ''}",
              flush=True)

    total_elapsed = time.time() - t_all

    # ── [5] Score both vs fine-tuned masks ────────────────────────────────
    print("\n[5] Scoring vs AutoSam fine-tuned masks...")
    full_dice_v, full_iou_v     = [], []
    partial_dice_v, partial_iou_v = [], []

    ft_files = {f for f in os.listdir(FINETUNED_MASKS_DIR) if f.endswith(".npy")} \
               if os.path.isdir(FINETUNED_MASKS_DIR) else set()

    for fname in sorted(ft_files):
        ref = np.load(os.path.join(FINETUNED_MASKS_DIR, fname))
        ff  = os.path.join(FULL_DIR,    fname)
        pf  = os.path.join(PARTIAL_DIR, fname)
        if os.path.exists(ff):
            pred = np.load(ff)
            if ref.shape != pred.shape:
                ref2 = cv2.resize(ref, (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                ref2 = ref
            d, v = dice_iou(pred, ref2)
            full_dice_v.append(d); full_iou_v.append(v)
        if os.path.exists(pf):
            pred = np.load(pf)
            if ref.shape != pred.shape:
                ref2 = cv2.resize(ref, (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                ref2 = ref
            d, v = dice_iou(pred, ref2)
            partial_dice_v.append(d); partial_iou_v.append(v)

    # ── [6] Summary ───────────────────────────────────────────────────────
    def fps_from_ms(ms_list):
        return 1000.0 / np.mean(ms_list) if ms_list else float("nan")

    full_fps    = fps_from_ms(full_ms_list)
    partial_fps = fps_from_ms(partial_ms_list)

    print("\n" + "=" * 68)
    print("BENCHMARK SUMMARY")
    print("=" * 68)
    print(f"  Model:       SAM2-large + torch.compile (default/inductor)")
    print(f"  Video:       {n_frames} frames @ {fps:.1f} fps")
    print(f"  Chunks:      {len(chunks_to_run)} run  "
          f"({WARMUP_CHUNKS} warmup excluded from FPS)")
    print(f"  Frame step:  full=1 (every frame)  partial={FRAME_STEP} "
          f"(every {FRAME_STEP}rd frame → {fps/FRAME_STEP:.1f} FPS equivalent)")
    print()
    print(f"  {'':30s}  {'full':>10}  {'partial':>10}  {'speedup':>8}")
    print(f"  {'─'*62}")
    print(f"  {'ms / original frame':30s}  "
          f"{np.mean(full_ms_list):>10.1f}  "
          f"{np.mean(partial_ms_list):>10.1f}  "
          f"{np.mean(full_ms_list)/np.mean(partial_ms_list) if partial_ms_list else float('nan'):>8.2f}x")
    print(f"  {'steady-state FPS':30s}  "
          f"{full_fps:>10.1f}  "
          f"{partial_fps:>10.1f}  "
          f"{'':>8}")
    print()
    print(f"  IoU(partial vs full)  at overlapping frames  "
          f"({len(iou_partial_vs_full)} frames):")
    if iou_partial_vs_full:
        print(f"    mean {np.mean(iou_partial_vs_full):.3f}  "
              f"median {np.median(iou_partial_vs_full):.3f}  "
              f"std {np.std(iou_partial_vs_full):.3f}")
    print()
    if full_dice_v or partial_dice_v:
        print(f"  vs AutoSam fine-tuned masks:")
        if full_dice_v:
            print(f"    full     Dice {np.mean(full_dice_v):.3f}  "
                  f"IoU {np.mean(full_iou_v):.3f}  ({len(full_dice_v)} frames)")
        if partial_dice_v:
            print(f"    partial  Dice {np.mean(partial_dice_v):.3f}  "
                  f"IoU {np.mean(partial_iou_v):.3f}  ({len(partial_dice_v)} frames)")
    print()
    print(f"  Total wall time:  {total_elapsed:.0f}s")
    print(f"  GPU mem peak:     {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=" * 68)


if __name__ == "__main__":
    main()
