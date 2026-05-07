# Prostate Segmentation — Custom Pipeline

This branch adds prostate gland segmentation and tracking for robotic surgery video (RARP)
on top of the base SAM3 codebase. See `README_SAM3.md` for the upstream SAM3 documentation.

---

## What This Adds

A three-stage pipeline for automatic prostate segmentation in surgical video — no manual
prompts at inference time:

1. **Fine-tuned SAM3 detector** — detects the prostate bounding box from a single frame
2. **Noise-robust SAM2 decoder** — generates a clean initial mask even from an imperfect box
3. **SAM2 video predictor** — propagates the mask through the video via maskmem

The pipeline runs in chunks (default 150 frames). At the start of each chunk the detector
reinitialises the tracker, preventing long-term drift.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `infer_prostate.py` | **Full pipeline** — detector → decoder → tracker, chunked |
| `train_detector.py` | Fine-tune SAM3 detector head on prostate data |
| `train_sam2_decoder.py` | Fine-tune SAM2 mask decoder with noisy-box augmentation |
| `detect_segment_fast.py` | infer_prostate.py + torch.compile + FRAME_STEP=3 skipping |
| `benchmark_sam2large.py` | FPS + IoU benchmark: full vs partial propagation |
| `masks_to_coco.py` | Convert binary mask PNGs to COCO JSON for detector training |
| `annotation_server.py` | Interactive annotation server (brush/click interface) |

---

## Checkpoints

| Model | Val IoU | Path |
|-------|---------|------|
| SAM3 fine-tuned detector | 0.907 | `detector_training/checkpoints/20260428_0942/checkpoint_best.pth` |
| SAM2 noise-robust decoder | 0.919 | `sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth` |

> Checkpoint files (`.pth`) are gitignored. Store them on the HPC filesystem or a separate
> artifact store.

---

## Dataset

**`prostate_tracker_long_qpr_19`** — 343 paired image/mask frames from robotic prostatectomy.
Located at: `../autosam-instruments-GraSP-trained/sam2/projects/prostate_tracker_long_qpr_19/`

Split: 291 train / 52 val (85/15, seed=42).

`detector_training/data/` contains:
- `annotations_train.json` / `annotations_val.json` — COCO-format box annotations
- `images/` — symlinks to the source JPEGs (HPC paths)

---

## Usage

### Full pipeline inference

```bash
cd /sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3

python3 infer_prostate.py \
    --video /path/to/source.mp4 \
    --detector_ckpt detector_training/checkpoints/20260428_0942/checkpoint_best.pth \
    --decoder_ckpt  sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth \
    --clip_start 3959 --clip_end 3969 \
    --output_dir short_clips \
    --reinit_interval 150
```

### Fast inference (torch.compile + frame-skipping)

```bash
python3 detect_segment_fast.py
```

### Train detector

```bash
python3 train_detector.py --epochs 50 --batch_size 32
```

### Train noise-robust decoder

```bash
python3 train_sam2_decoder.py --epochs 30 --batch_size 4
```

---

## Throughput

Benchmarked on a single NVIDIA H100 80GB:

| Configuration | FPS |
|---|---|
| Baseline (SAM2-large) | ~16 fps |
| + `torch.compile` `mode="default"` | ~22 fps |
| + FRAME_STEP=3 (every 3rd frame) | see `benchmark_sam2large.py` |
| Meta upstream target (FA3 + max-autotune) | 32 fps |

**Key bottleneck:** SAM2 maskmem is autoregressive — hard sequential dependency.
GPU is latency-bound, not memory-bound. Highest-leverage remaining speedup: batch
pre-encode all chunk frames through the ViT before `propagate_in_video` (zero temporal
dependency — can be fully parallelised).

---

## Environment

- Platform: Arion HPC (Linux, LSF)
- Container: `singularity shell --nv --writable /sc/arion/projects/video_rarp/neel_projects/sam3_dev`
- Temp dir: `/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg`
- No `nvidia-smi` in PATH — use `torch.cuda.get_device_properties(0)`

---

## Project Timeline

### Phase 1 — SAM2 with Manual Prompts
**What:** SAM2 video predictor + maskmem for tracking. Built a frontend for drawing bounding boxes or masks on frame 0 and propagating through the video at configurable sample rates.
**Worked:** SAM2 maskmem is robustly reliable once initialized with a clean prompt.
**Failed:** Manual prompt required on every new clip — not usable at scale.

---

### Phase 2 — AutoSAM2 (custom decoder, zero prompts)
**What:** Removed SAM2's prompt encoder entirely. Trained a custom `MaskDecoder` + `TwoWayTransformer` on top of the frozen SAM2 Hiera-ViT encoder for zero-shot prostate detection. 343 image/mask pairs.
**Result:** ~mIoU 0.86 on static frames.
**Why it failed for video:**
- No maskmem — each frame is independent, masks flicker/drift.
- The decoder simultaneously solves detection (where?) and segmentation (which pixels?) with only 343 training samples — a ceiling problem.
- Carrying a noisy frame-0 mask into the SAM2 tracker propagates that noise forward.

**Phase 2b — Hybrid (AutoSAM2 init → SAM2 maskmem):**
`prostate_hybrid_track.py` — AutoSAM2 generates frame-0 mask, SAM2 large propagates it. Better than pure AutoSAM2, but ~15% of badly-wrong frame-0 detections still corrupt the tracker seed.

---

### Phase 3 — SAM3 Out-of-the-Box (failed)
**What:** SAM3's CLIP-based language-conditioned detector with text prompt `"prostate gland"`.
**Why it failed:** SAM3's detector uses cross-attention between visual tokens and CLIP text tokens at every transformer decoder layer. `"prostate gland"` is far outside CLIP's training distribution — the text embedding has no geometric prior for surgical anatomy, so cross-attention adds noise rather than signal. Produces unreliable boxes, worse than a manual prompt.

---

### Phase 4 — Fine-tuned SAM3 Detector
**What:** Fine-tuned the SAM3 detector head only (transformer decoder, dot_prod_scoring, geometry_encoder, segmentation_head). ViT + CLIP backbones frozen. 32.7M / 840.5M params. Text query encoded once and cached.
**Result: val IoU = 0.907** (epoch 47/50).
**Why it worked:** Backbones already extract rich features. The decoder only needed to learn prostate geometry — a small learning problem. Fixed text embedding acts as a stable class bias rather than a meaningful semantic anchor.

---

### Phase 5 — SAM3 Detector → SAM2 Tracker Pipeline
**What:** Fine-tuned detector gives a bounding box → base SAM2 video predictor propagates masks via maskmem. Chunked reinit every ~500 frames.
**First result:** `case_213_clipped` 3959–3969s, 299/299 frames masked, conf=0.19.
**Why better than SAM3's own tracker:** SAM2 maskmem is purely geometric — no text cross-attention during propagation. Clean separation: detector = semantic anchor, tracker = temporal interpolator.

---

### Phase 6 — Noise-Robust SAM2 Decoder
**What:** SAM2 mask decoder fine-tuned with deliberately perturbed bounding boxes: 15% tight (±10%), 50% medium (±50%), 25% large (±100%), 10% extreme (full image). Encoder + prompt encoder frozen. Loss: Dice + BCE.
**Result: val IoU = 0.919** (30 epochs).
**Why it matters:** Bridges the gap between imperfect detector boxes (conf=0.19) and a clean mask initialisation for the tracker.

---

### Phase 7 — Full Pipeline Integrated
**What:** `infer_prostate.py` wires all three components — fine-tuned detector → noise-robust decoder → SAM2 video predictor — in a chunked loop with configurable reinit interval.

---

### Phase 8 — Throughput Optimization
Baseline ~16 fps on H100. `torch.compile mode="default"` brought this to ~22 fps. Frame-skipping (FRAME_STEP=3) benchmarked in `detect_segment_fast.py`.

**Key bottleneck:** SAM2 maskmem is autoregressive — hard sequential dependency, free VRAM cannot help. GPU is latency-bound.

**Highest-leverage remaining optimization:** Batch pre-encode all chunk frames through the ViT before calling `propagate_in_video`. The ViT has zero temporal dependency — all frames can be encoded in one pass and pre-populated into `inference_state["cached_features"]`. Current cache is a single-entry dict (cache miss on every frame during propagation).

**Remaining:** batch pre-encoding, `max-autotune`, Flash Attention 3 + FP8, multi-video parallelism. Reference: `facebookresearch/sam3` commit `9f22cb9`.
