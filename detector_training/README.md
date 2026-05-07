# SAM3 Detector Fine-tuning — Prostate Gland Detection

## Goal

Fine-tune the **SAM3 detector head** to automatically detect the prostate gland in surgical video frames, using only a fixed text query ("prostate gland") — no bounding-box prompt at inference time. This is the first stage of the full SAM3 pipeline: detector finds the box → tracker propagates the mask through the video.

---

## Background and Context

The SAM2-based tracker (`sam3_bbox_segment.py`) already works well when given a bounding box on frame 0. The bottleneck is that box has to be provided manually every time. The SAM3 model includes a **language-conditioned detector** (`Sam3Image`) that takes an image + text query and outputs predicted boxes. Fine-tuning this component on prostate data would enable zero-prompt video segmentation.

For reference on how the SAM2 tracker was fine-tuned for instrument segmentation, see:
```
neel_projects/autosam-instruments-GraSP-trained/sam2/train_grasp.py
```
That script fine-tuned the SAM2 image encoder + a custom mask decoder end-to-end using Dice + CE loss on the GraSP dataset. The prostate detector fine-tuning follows a similar philosophy but targets SAM3's transformer decoder instead.

---

## Architecture: What Is Being Trained

SAM3 (`Sam3Image`) has the following major components:

| Component | Role | Trained? |
|-----------|------|----------|
| `backbone.vision_backbone` | ViT image encoder | **FROZEN** |
| `backbone.language_backbone` | CLIP text encoder | **FROZEN** |
| `transformer` | Cross-attention decoder (vision × text) | YES |
| `dot_prod_scoring` | Box/text similarity scorer | YES |
| `geometry_encoder` | Geometry input encoder | YES |
| `segmentation_head` | Pixel-level mask decoder | YES |

The vision and language backbones are frozen because:
1. They are large (majority of the ~600M params) and we have only 343 training samples.
2. The backbones already generalize well; it is the decoder that needs to learn what "prostate gland" looks like geometrically in this surgical domain.

Text query is fixed to `"prostate gland"` and encoded **once** before training begins — the embedding is cached and reused every batch, so the language backbone never runs at training time.

---

## Data

**Source dataset:** `prostate_tracker_long_qpr_19`
```
neel_projects/autosam-instruments-GraSP-trained/sam2/projects/prostate_tracker_long_qpr_19/
  images/{stem}.jpg    # 1920×1080 frames
  masks/{stem}.png     # binary masks (0=BG, 255=prostate)
```

**Split:** 343 total → 291 train / 52 val (85/15, seed=42)

**Ground-truth boxes:** derived from tight bounding box of the foreground region in each mask — not hand-labelled, so they are as accurate as the mask annotations.

**COCO annotation files (already generated):**
```
detector_training/data/annotations_train.json   # 291 samples
detector_training/data/annotations_val.json     # 52 samples
detector_training/data/images/                  # symlinks to source JPEGs
```

Box format in JSON: `[x, y, w, h]` pixel coords (COCO standard).
The dataset class converts to `[cx, cy, w, h]` normalized to [0,1] at load time.

**Images are resized to 1008×1008** at load time (SAM3 ViT input size), with ImageNet normalization.

---

## Scripts

### `masks_to_coco.py`
Converts binary mask PNGs → COCO JSON. Already run; outputs are in `detector_training/data/`.

```bash
python3 masks_to_coco.py   # only needs to be re-run if the source dataset changes
```

### `train_detector.py`
Main fine-tuning script.

```bash
# Default run (50 epochs, batch=4, lr=1e-4)
python3 train_detector.py

# Custom
python3 train_detector.py --epochs 30 --lr 5e-5 --batch_size 8
```

**Key hyperparameters (defaults):**

| Param | Value | Notes |
|-------|-------|-------|
| `--epochs` | 50 | cosine LR schedule |
| `--lr` | 1e-4 | AdamW |
| `--weight_decay` | 1e-5 | AdamW |
| `--batch_size` | 4 | 291 samples → ~72 steps/epoch |
| `--w_l1` | 5.0 | L1 box loss weight |
| `--w_giou` | 2.0 | GIoU box loss weight |

**Loss:** Sigmoid focal loss (classification) + L1 + GIoU (regression), with Hungarian matching to assign GT box to best-matching query.

**Metric:** IoU of the highest-confidence predicted box vs GT box.

---

## Forward Pass (Bypassing BatchedDatapoint)

The training loop calls SAM3's internal sub-stages directly, avoiding the complex `BatchedDatapoint` pipeline used in full SAM3 training:

```
1. model.backbone.forward_image(imgs)       → vis_out (frozen, no_grad)
2. inject cached lang_feats into backbone_out
3. model._encode_prompt(backbone_out, find_input, geo_prompt)
4. model._run_encoder(backbone_out, find_input, prompt, prompt_mask)
5. model._run_decoder(memory, pos_embed, src_mask, ...)
   → pred_logits (B, Q, 1)
   → pred_boxes  (B, Q, 4) cx,cy,w,h normalized
```

`FindStage` is constructed with zero input boxes/points — the detector is operating in pure text-grounding mode.

---

## Outputs

| Path | Contents |
|------|----------|
| `detector_training/checkpoints/checkpoint_latest.pth` | Latest epoch |
| `detector_training/checkpoints/checkpoint_best.pth` | Best val IoU |
| `detector_training/checkpoints/checkpoint_epoch_N.pth` | Every 10 epochs |
| `detector_training/tensorboard/` | TensorBoard logs |
| `detector_training/runs/<timestamp>/config.json` | Run hyperparameters |

Checkpoints save **only trainable weights** (frozen backbone keys excluded), so they are ~small and can be loaded on top of a fresh SAM3 checkpoint.

---

## Status

- [x] Source data identified: `prostate_tracker_long_qpr_19` (343 samples)
- [x] `masks_to_coco.py` written and run → 291 train / 52 val COCO JSON files
- [x] `train_detector.py` written — full training loop with loss, metrics, TensorBoard, checkpointing
- [x] Training completed — **best val IoU 0.907** at epoch 47 (50 epochs, batch 32, H100)
- [x] `_run_decoder` signature verified against live SAM3 codebase
- [x] `detect_and_segment.py` written — detector + tracker inference pipeline with clipping, colour, alpha CLI options
- [x] First inference run: `case_213_clipped` at 3959–3969s, 299/299 frames masked
- [ ] Long-video tracking strategy — see section below

---

## Next Steps to Resume

1. **Smoke test the forward pass** (one batch, no full training):
   ```bash
   python3 -c "
   import torch, json, os
   from sam3.model_builder import build_sam3_image_model
   from sam3.model.data_misc import FindStage
   from sam3.model.geometry_encoders import Prompt
   SAM3_CKPT = '/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'
   model = build_sam3_image_model(SAM3_CKPT, load_from_HF=False, eval_mode=False, enable_segmentation=True).cuda()
   x = torch.randn(2, 3, 1008, 1008).cuda()
   vis = model.backbone.forward_image(x)
   print('vis keys:', list(vis.keys()))
   print('smoke test passed')
   "
   ```

2. **Run training** (ideally as an LSF job on a GPU node):
   ```bash
   bsub -P acc_video_rarp -q gpu -n 4 -W 24:00 -R 'rusage[mem=32000]' \
        -gpu 'num=1:j_exclusive=yes' \
        -o detector_training/train_%J.log \
        python3 train_detector.py --epochs 50
   ```

3. **Check TensorBoard** after a few epochs:
   ```bash
   tensorboard --logdir detector_training/tensorboard --port 6006
   ```

4. **Integrate fine-tuned detector** into the inference pipeline once val IoU reaches ~0.6+.

---

## SAM3 Checkpoint Reference

```
/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/
  3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
```

Key structure in the checkpoint:
- `detector.*` — 1156 keys total (includes `detector.backbone.*` = 759 keys)
- `tracker.*` — 309 keys

When loading the tracker predictor (`sam3_bbox_segment.py`), both `tracker.*` and `detector.backbone.*` must be loaded; the backbone keys are remapped from `detector.backbone.*` → `backbone.*`. See `sam3_bbox_segment.py` for the exact loading logic.

---

## Training Results

| Run | Epochs | Batch | LR | Best val IoU | Checkpoint |
|-----|--------|-------|----|-------------|------------|
| `20260428_0942` | 50 | 32 | 1e-4 | **0.907** (epoch 47) | `checkpoints/20260428_0942/checkpoint_best.pth` |

Training curve was smooth — IoU went from 0.0 at epoch 0 to 0.437 by epoch 3, stabilising above 0.89 by epoch 35. No overfitting observed over 50 epochs (train and val loss tracked together in MLflow / TensorBoard).

First inference on an unseen video (`case_213_clipped`, 3959–3969 s):
- Detection confidence: 0.19 (domain shift from training clips, but box was geometrically correct)
- 299/299 frames masked, ~18–22% frame coverage, ~15 s propagation time
- Outperforms the AutoSAM2 still-frame zero-shot baseline on temporal consistency

---

## Long-Video Tracking Strategy

### The core problem

The current inference pipeline (`detect_and_segment.py`) does the following:

1. Run the detector **once** on a single frame to get a bounding box.
2. Feed that box to the SAM3 tracker as the initial prompt.
3. Call `propagate_in_video` — the tracker runs to the end of the video without any further input from the detector.

The SAM3/SAM2 tracker maintains a **rolling memory bank** of past frames:
- ~6–7 "working memory" slots — the most recent masked frames.
- A small set of "long-term memory" frames — selected periodically when the tracker is confident.

On short clips this works well. On longer videos three failure modes accumulate:

| Failure mode | Cause | Observed symptom |
|--------------|-------|-----------------|
| **Drift** | Small per-frame errors compound | Mask slowly walks off the prostate |
| **Memory staleness** | Working memory only covers ~6 frames; appearance change (lighting, instrument) not represented | Tracker loses confidence silently, coverage collapses |
| **Occlusion recovery failure** | Instrument covers prostate for several seconds → memory fills with empty/occluded masks → no clean anchor to reattach to when prostate reappears | Hard failure after an occlusion |

The first sign of this appeared in the 10 s test clip: frame-250 coverage dropped from ~22% to ~11% with no instrument in frame, purely from accumulating small errors over ~8 s.

The key insight for a fix: the **detector is text-conditioned** (`"prostate gland"`) and makes an independent prediction on any single frame. It does not drift. It can serve as a periodic hard reset for the tracker.

---

### Option A — Chunked reinitialization (recommended first step)

Divide the video into fixed-size windows (e.g. 500 frames ≈ 17 s at 30 fps). At the **start of every window**, run the detector on that frame. If confidence exceeds a threshold, reinitialize the tracker from the new box and propagate through the window.

```
window 0           window 1           window 2
[det → track 500f] [det → track 500f] [det → track 500f] ...
```

**Pros:** zero added complexity inside the tracker, clean hard resets, uses the component we actually trained.  
**Cons:** abrupt mask boundary at window edges if the new box differs slightly from the tracked mask.

**Implementation sketch:**

```python
for chunk_start in range(0, total_frames, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
    box, conf = run_detector(ckpt, video, frame_idx=chunk_start, device=device)
    if conf < CONF_THRESHOLD:
        # detector not confident — extend previous window instead of resetting
        continue
    reset_tracker(predictor, inference_state)
    add_new_points_or_box(inference_state, frame_idx=chunk_start, box=box)
    for frame_idx, masks in propagate_in_video(inference_state,
                                               start=chunk_start,
                                               end=chunk_end):
        save_mask(frame_idx, masks)
```

---

### Option B — Confidence-triggered reinitialization (adaptive)

During propagation the tracker outputs **per-frame object scores**. Monitor these. When the score falls below a threshold for N consecutive frames, immediately run the detector on the current frame and reinitialize.

```
track ... track ... [score < τ for N frames] → detect → reset → track ...
```

**Pros:** only reinitializes when actually needed, no unnecessary resets on stable stretches; better for surgical videos where the prostate is visible for long continuous runs.  
**Cons:** choosing τ and N requires calibration; triggering mid-stream requires flushing the tracker state cleanly at an arbitrary frame.

This is the natural evolution of Option A and should be layered on top of it rather than replacing it.

---

### Option C — Detector as a parallel corrector

Run the detector every K frames in parallel with tracking. Compute the IoU between the detector's predicted box and the bounding box of the current tracked mask. If IoU drops below a threshold, treat this as a divergence event and reset the tracker to the detector's box.

```
detector (every K frames):   [box] ............ [box] ............ [box]
tracker (continuous):        ──────────────────────────────────────────
                                    ↑ compare IoU, reset if diverged
```

This is closest to how the full `Sam3ImageOnVideoMultiGPU` class is designed — the detector runs across frames in a distributed fashion while the tracker maintains temporal state. The difference is that our detector is fine-tuned, so it adds domain-specific correction rather than the generic open-vocabulary detection.

**Pros:** most robust; catches gradual drift before it becomes a hard failure.  
**Cons:** most implementation work; running the detector every K frames adds non-trivial compute (~27 s per frame at current speed, though this can be batched and pipelined).

---

### Option D — Overlap-blended sliding windows

Identical to Option A but windows overlap by M frames. In the overlap region, blend masks from the two adjacent windows using a linear ramp on the object score. This removes the abrupt boundary artifact of Option A.

```
window 0: frames   0–549
window 1: frames 500–1049   (50-frame overlap with window 0: frames 500–549)
                             ↑ blend both masks here by score weight
```

**Pros:** smooth output suitable for clinical review or downstream model training.  
**Cons:** requires storing two concurrent mask streams during the overlap; slightly more bookkeeping.

---

### Recommendation

| Phase | Strategy | When to use |
|-------|----------|-------------|
| **Now** | Option A (chunked, ~500 fr windows) | Any video > 1 min; simplest to implement |
| **Next** | Add Option B on top of A | Once chunked pipeline is stable; replaces fixed window size with adaptive resets |
| **Long term** | Option C | If high frame-rate accuracy is needed for downstream model training or clinical use |

Option D can be added to any of the above as a post-processing step purely for visual output quality.

The central design principle for all options: **the detector is the semantic anchor, the tracker is the temporal interpolator**. Never let the tracker run longer than the window over which its memory remains reliable, and always use the detector — not the tracker's own memory — as the source of truth when the two disagree.
