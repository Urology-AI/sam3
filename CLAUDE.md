# CLAUDE.md — SAM3 Prostate Segmentation Project

Working directory: `/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3/`

---

## What This Project Does

Adapts Meta's **SAM3** model (language-conditioned detector + SAM2-inherited tracker) to
automatically segment the prostate gland in robotic surgery video — no manual bounding-box
prompt at inference time.

The pipeline has two stages:
1. **Detector** (`Sam3Image`) — fine-tuned on prostate data; takes a frame + text query
   `"prostate gland"` → outputs a bounding box.
2. **Tracker** (`Sam3TrackerPredictor`) — given the box on one frame, propagates a pixel-level
   mask through the rest of the video using SAM2's memory-based propagation.

---

## Key Files

| File | Purpose |
|------|---------|
| `train_detector.py` | Fine-tunes the SAM3 detector head (transformer decoder, dot-product scorer, geometry encoder, segmentation head). Backbones frozen. |
| `detect_and_segment.py` | End-to-end inference: optional ffmpeg clip → detector → tracker → overlay video. |
| `sam3_bbox_segment.py` | Earlier manual-box version of the tracker pipeline (reference). |
| `masks_to_coco.py` | Converts binary mask PNGs → COCO JSON for detector training data. |
| `detector_training/README.md` | Full technical documentation: training setup, results, and long-video tracking strategy. |

---

## Training

**Dataset:** `prostate_tracker_long_qpr_19` — 343 paired image/mask frames from robotic prostatectomy.
Split 291 train / 52 val (85/15, seed=42).

**Annotations:** derived from masks, stored as COCO JSON in `detector_training/data/`.

**What is trained vs frozen:**

| Component | Trained? |
|-----------|----------|
| `backbone.vision_backbone` (ViT) | FROZEN |
| `backbone.language_backbone` (CLIP) | FROZEN |
| `transformer` (cross-attention decoder) | YES |
| `dot_prod_scoring` | YES |
| `geometry_encoder` | YES |
| `segmentation_head` | YES |

Trainable params: **32.7M / 840.5M total**.

Text query `"prostate gland"` is encoded once before training and cached — the language
backbone never runs during training.

**Loss:** sigmoid focal (classification) + L1 + GIoU (regression), Hungarian matching.

**Run command:**
```bash
cd /sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
python3 train_detector.py --epochs 50 --batch_size 32
```

**Completed run:**

| Run ID | Epochs | Batch | Best val IoU | Checkpoint |
|--------|--------|-------|-------------|------------|
| `20260428_0942` | 50 | 32 | **0.907** (epoch 47) | `detector_training/checkpoints/20260428_0942/checkpoint_best.pth` |

Logs: `detector_training/train_run.log`, TensorBoard: `detector_training/tensorboard/`, MLflow: `detector_training/mlruns/`.

---

## Inference

```bash
python3 detect_and_segment.py \
  --video /sc/arion/projects/video_rarp/neel_projects/intuitive_videos/case_213_clipped.mp4 \
  --detector_ckpt detector_training/checkpoints/20260428_0942/checkpoint_best.pth \
  --clip_start 3959 --clip_end 3969 \
  --output_dir short_clips \
  --mask_color "0,255,0" \
  --mask_alpha 0.20
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Source video path |
| `--detector_ckpt` | required | Fine-tuned checkpoint (.pth) |
| `--clip_start` / `--clip_end` | None | Seconds — runs ffmpeg clip before inference |
| `--detect_frame` | 0 | Frame index (within clip) to run detector on |
| `--output_dir` | auto | Where overlay.mp4 and masks/ are written |
| `--mask_color` | `0,255,0` | R,G,B overlay colour |
| `--mask_alpha` | 0.35 | Overlay opacity |
| `--conf_threshold` | 0.1 | Warn if detection confidence is below this |

The clip is written to `output_dir/<stem>_<start>-<end>.mp4` first; all subsequent
steps operate on the clip, not the full source video.

**First result:** `case_213_clipped` at 3959–3969 s — 299/299 frames masked, conf=0.19,
~18–22% frame coverage. Output saved to `short_clips/overlay.mp4`.

---

## Known Bugs Fixed This Session

1. **BPE tokenizer path** — `pkg_resources.resource_filename` returned the path inside an
   egg/zip, yielding ZIP bytes instead of gzip. Fixed by adding `BPE_PATH` constant pointing
   directly to `sam3/assets/bpe_simple_vocab_16e6.txt.gz` and passing it explicitly to
   `build_sam3_image_model(bpe_path=BPE_PATH, ...)`.

2. **`build_sam3_image_model` arg order** — `bpe_path` is the *first* positional argument, not
   `checkpoint_path`. Always pass both as keyword arguments to avoid silent misassignment.

---

## SAM3 API — Forward Pass Pattern

When calling SAM3 internals directly (bypassing `BatchedDatapoint`):

```python
# 1. Vision features (frozen — run under no_grad)
vis_out = model.backbone.forward_image(imgs)    # keys: backbone_fpn, vision_pos_enc, ...

# 2. Inject cached text features
backbone_out = {**vis_out, "language_features": lang_feats.expand(-1, B, -1),
                            "language_mask":     lang_mask.expand(B, -1)}

# 3. Construct zero-prompt FindStage + Prompt
find_input = FindStage(img_ids=..., text_ids=zeros(B), input_boxes=zeros(B,0,4), ...)
geo_prompt = Prompt(box_embeddings=zeros(0,B,4), box_mask=zeros(B,0,bool))

# 4. Encode → encode → decode
prompt, prompt_mask, backbone_out = model._encode_prompt(backbone_out, find_input, geo_prompt)
backbone_out, encoder_out, _      = model._run_encoder(backbone_out, find_input, prompt, prompt_mask)
out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
out, _ = model._run_decoder(memory=out["encoder_hidden_states"],
                             pos_embed=encoder_out["pos_embed"],
                             src_mask=encoder_out["padding_mask"],
                             out=out, prompt=prompt, prompt_mask=prompt_mask,
                             encoder_out=encoder_out)

# 5. Outputs
pred_logits = out["pred_logits"]   # (B, 200, 1)  — sigmoid → confidence per query
pred_boxes  = out["pred_boxes"]    # (B, 200, 4)  — cx,cy,w,h normalised [0,1]
```

`_run_decoder` signature: `(self, pos_embed, memory, src_mask, out, prompt, prompt_mask, encoder_out)` — always use keyword args.

---

## SAM3 Checkpoint

```
/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/
  3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
```

Key namespaces:
- `detector.backbone.*` → 759 keys (ViT + neck); remapped to `backbone.*` when loading the tracker predictor
- `detector.*` (non-backbone) → 397 keys (transformer, scorer, heads)
- `tracker.*` → 309 keys (SAM2 mask decoder + memory)

---

## Long-Video Tracking — Problem and Roadmap

The current pipeline runs the detector **once** on a single frame and propagates indefinitely.
SAM2's rolling memory (~6–7 working + a few long-term frames) degrades on long videos via:
- **Drift** — small per-frame errors compound
- **Memory staleness** — appearance change not reflected in memory
- **Occlusion recovery failure** — memory fills with empty masks during occlusion

**Planned fix (phased):**

| Phase | Strategy |
|-------|----------|
| 1 (next) | Option A: chunked reinit — detect at start of every ~500-frame window, reset tracker |
| 2 | Option B: confidence-triggered reinit — monitor tracker object scores, reset when score collapses |
| 3 | Option C: detector as parallel corrector — run detector every K frames, reset if box/mask IoU diverges |

Full design rationale is in `detector_training/README.md` → *Long-Video Tracking Strategy*.

**Design principle:** the detector is the semantic anchor (text-conditioned, no drift);
the tracker is the temporal interpolator (appearance-based, efficient). Never let the
tracker run longer than its memory remains reliable.

---

## Throughput Bottleneck Analysis & Next Speedup

**Why GPU memory is ~3 GB used but throughput is still limited:**
SAM2's memory attention is autoregressive — frame N's output is written into the memory bank
before frame N+1 can run. This is a hard sequential dependency; free GPU memory cannot help here.
The GPU is latency-bound (small tensors, kernel launch overhead), not memory-bound.

**Image encoder cache — already partially done, but minimal:**
`SAM2VideoPredictor._get_image_feature` (`sam2_video_predictor.py:717`) has a cache:
```python
inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
```
This is a **single-entry dict** — it replaces itself on every new frame. The comment admits it:
*"Cache the most recent frame's feature (for repeated interactions with a frame; we can use
an LRU cache for more frames in the future)."*
During `propagate_in_video` every frame is a cache miss and the encoder runs one at a time.

**Next speedup — pre-encode the chunk in batches:**
The ViT image encoder has **zero temporal dependency** — all frames in a chunk can be encoded
in parallel. Before calling `propagate_in_video`, batch all chunk frames (e.g. batch=8) through
`predictor.forward_image`, and pre-populate `state["cached_features"]` with all results.
`_get_image_feature` will then always hit the cache and the encoder never runs during propagation.

Memory cost: 50 frames (150-frame chunk at step=3) of FPN features ≈ 500 MB–1 GB — fine on H100.

**SAM 3.1 speed improvements (from `sam3/perflib/`):**
| Improvement | File | Notes |
|---|---|---|
| Flash Attention 3 + FP8 | `perflib/fa3.py` | Q/K/V cast to `float8_e4m3fn`, FA3 kernel, output back to bf16 |
| `torch.compile` `max-autotune` | `sam3_tracker_base._compile_all_components` | Compiles `maskmem_backbone`, `transformer.encoder`, `sam_mask_decoder`; `cache_size_limit=64` |
| Triton NMS + connected components | `perflib/triton/` | Autotuned Triton kernels replacing PyTorch/CPU post-processing |
| Async frame loading (TorchCodec) | `model/io_utils.py` | Background thread decoder overlaps I/O with GPU compute |
| `compile_wrapper` pattern | `perflib/compile.py` | `.contiguous()` on inputs + `.clone()` on outputs enables `fullgraph=True` |

**What we have vs what remains:**
- Applied: `mode="default"` compile (partial), frame-skipping FRAME_STEP=3, chunked reinit
- Remaining: pre-encode batching, `max-autotune`, FA3+FP8, multi-video parallelism

---

## Environment Notes

- Platform: Arion HPC (Linux, LSF scheduler)
- GPU available interactively: NVIDIA H100 80GB HBM3
- No `nvidia-smi` in PATH; use `torch.cuda.get_device_properties(0)` instead
- `/tmp` resolves to login node — always use `SCRATCH_TMP = "/sc/arion/projects/video_rarp/neel_projects/tmp_sam3_seg"` for temp files
- MLflow FutureWarning about filesystem backend is harmless — ignore
- `pkg_resources` deprecation warning is harmless — ignore

---

## Related Projects

| Path | Description |
|------|-------------|
| `neel_projects/autosam-instruments-GraSP-trained/sam2/prostate_tracking/tracker_train.py` | AutoSAM2 prostate segmentation (still-frame, no detector) — reference for training loop patterns |
| `neel_projects/autosam-instruments-GraSP-trained/sam2/projects/prostate_tracker_long_qpr_19/` | Source dataset: `images/` + `masks/` |
| `neel_projects/intuitive_videos/` | Source surgical videos for inference |
