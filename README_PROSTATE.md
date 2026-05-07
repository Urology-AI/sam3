# Prostate Segmentation — Custom Pipeline

This branch adds prostate gland segmentation and tracking for robotic surgery video (RARP)
on top of the base SAM3 codebase. See Meta's `README.md` for the upstream SAM3 documentation.

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
