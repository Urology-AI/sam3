# Surgical Anatomy Segmentation — Custom Pipeline

This branch adds automatic segmentation and tracking of surgical anatomy (prostate gland,
vas deferens) in robotic surgery video (RARP) on top of the base SAM3 codebase.
See `README_SAM3.md` for the upstream SAM3 documentation.

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
| `infer_prostate.py` | Full pipeline — detector → decoder → tracker, chunked |
| `infer_prostate_bidir.py` | Same but propagates bidirectionally from an anchor frame |
| `infer_vas.sh` | Single-clip VAS deferens inference (wraps `infer_prostate_bidir.py`) |
| `infer_vas_batch.sh` | Batch VAS inference over a list of clips from `untitled.txt` |
| `train_detector.py` | Fine-tune SAM3 detector head on any structure |
| `train_sam2_decoder.py` | Fine-tune SAM2 mask decoder with noisy-box augmentation |
| `detect_segment_fast.py` | Fast prostate pipeline: torch.compile + FRAME_STEP=3 + batch pre-encoding |
| `masks_to_coco.py` | Convert binary mask PNGs to COCO JSON for detector training |
| `aua_tracking/` | Browser annotation tools + SAM2 tracking scripts for AUA presentation |

---

## Checkpoints

| Model | Target | Val IoU | Path |
|-------|--------|---------|------|
| SAM3 fine-tuned detector | Prostate | 0.907 | `detector_training/checkpoints/20260428_0942/checkpoint_best.pth` |
| SAM2 noise-robust decoder | Prostate | 0.919 | `sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth` |
| SAM3 fine-tuned detector | VAS deferens | — | `detector_training/checkpoints/20260522_1403/checkpoint_best.pth` |
| SAM2 noise-robust decoder | VAS deferens | — | `sam2_decoder_training/checkpoints/20260522_1448/checkpoint_best.pth` |

> Checkpoint files (`.pth`) are gitignored. Store on the HPC filesystem or a separate artifact store.

> **Text token:** Both detectors are queried with `"prostate gland"` at inference — including
> the VAS model. The CLIP language backbone is frozen during training, so the text embedding
> is a fixed class label, not a semantic description. The detector learns to associate that
> fixed vector with whichever structure appears in the training data. Passing `"vas deferens"`
> at inference would produce a different CLIP vector and break the trained association.

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

| Configuration | Throughput |
|---|---|
| Baseline (SAM2-large) | ~16 fps |
| + `torch.compile` `mode="default"` | ~22 fps |
| + FRAME_STEP=3 + batch pre-encoding | **1.18× real-time** (19.06s for 22.5s @ 59.9 fps) |
| Meta upstream target (FA3 + max-autotune) | 32 fps |

Batch pre-encoding (`pre_encode_chunk` in `detect_segment_fast.py`) removes the ViT encoder
from the propagation critical path. All chunk frames are encoded in batches of 8 before
`propagate_in_video` starts — the maskmem loop hits the cache on every frame and never calls
`forward_image`. Masks verified correct.

**Remaining speedups:** `max-autotune`, Flash Attention 3 + FP8, multi-video parallelism.

---

## Streaming / Real-Time Roadmap

Batch pre-encoding does not transfer to streaming — frame N+1 hasn't arrived when frame N
is being processed, so there is nothing to batch ahead of time. The ViT cost returns per frame.

**Irreducible constraint:** maskmem is autoregressive. Frame N's memory token must be written
before frame N+1 can start. This is a hard sequential dependency regardless of parallelism.

**Options ranked by impact:**

| Option | Mechanism | Status |
|--------|-----------|--------|
| **CUDA stream pipelining** | Encode frame N+1 on GPU stream 1 while maskmem runs frame N on stream 2. H100 has enough SMs for both concurrently. `cached_features` is the handoff — encode writes, maskmem reads. Hides ViT cost almost entirely. | Not implemented |
| **Async detector reinit** | SAM3 detector runs in a background thread every ~2.5s. Maskmem loop never stalls waiting for the new box. | Not implemented |
| **Frame skipping** | FRAME_STEP=5/6 — immediate throughput gain, masks go staler. | Config change |
| **max-autotune + FA3+FP8** | ~40% free throughput from Meta's upstream commit. | Not implemented |
| **SAM2-base for propagation** | Smaller maskmem model, faster per step, modest quality impact. | Not implemented |

**Target streaming architecture:**
```
Thread 1 (CPU)         : decode frame → preprocess
Thread 2 (GPU stream 1): ViT encode → write to cached_features
Thread 3 (GPU stream 2): maskmem read cached_features → propagate → emit mask
Thread 4 (CPU, async)  : SAM3 detector reinit every 150 frames (non-blocking)
```

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
Baseline ~16 fps on H100. `torch.compile mode="default"` brought this to ~22 fps. Frame-skipping (FRAME_STEP=3) + batch pre-encoding of the ViT brought this to **1.18× real-time** (19.06s for 22.5s @ 59.9 fps, H100).

**Key bottleneck:** SAM2 maskmem is autoregressive — hard sequential dependency, free VRAM cannot help. GPU is latency-bound.

**Batch pre-encoding (`pre_encode_chunk` in `detect_segment_fast.py`):** The ViT image encoder has zero temporal dependency — all chunk frames are encoded in batches of 8 before `propagate_in_video` starts. The maskmem loop hits `inference_state["cached_features"]` on every frame and never calls `forward_image`. Previously the cache was a single-entry dict (cache miss on every frame during propagation).

**Linear mask interpolation (`detect_segment_fast.py`, rendering step):** With FRAME_STEP=3 only every 3rd frame has a computed mask. Rather than holding the last mask as a static overlay for skipped frames, the renderer linearly interpolates between consecutive keyframe masks. Per-pixel float alpha gives smooth transitions instead of hard cuts every 3 frames. Pre-saved `.npy` keyframe masks are reusable — alpha and interpolation can be changed without rerunning the GPU pipeline (`rerender_overlay.py`).

**Remaining:** `max-autotune`, Flash Attention 3 + FP8, multi-video parallelism. Reference: `facebookresearch/sam3` commit `9f22cb9`.

---

---

## Nerve-Sparing Phase Localisation

> **Note:** This is a standalone research project. It is maintained in this repository for practical reasons — the HPC environment and SAM2 installation are already set up here, and the pipeline uses the SAM2 frozen encoder as its feature extractor.

### Clinical Motivation

The nerve-sparing phase of a robotic prostatectomy (RARP) is roughly 20–30 minutes within a 2–3 hour surgery. Analysing patient outcomes from the full video is impractical — the goal is to automatically extract just that window.

The nerve-sparing phase is bounded by two surgical events:
- **Start:** VAS deferens cutting (the bilateral division of the vas marks the entry into the nerve-sparing dissection). An earlier alternative start is the catheter pull following the anterior bladder-neck incision — both timestamps are available in `intuitive_videos/annotate_fine.csv`.
- **End:** Endobagging (the prostate is placed into a laparoscopic bag for extraction). This marks the completion of the dissection.

### Confidence-Sweep Baseline (failed)

As a first attempt, the trained VAS deferens detector was run across every sampled frame of the full surgery to see if its box confidence peaked during VAS cutting. The resulting trace (`vas_confidence_sweep.png`) showed no clean spike. The detector fires on tubular structures throughout the surgery and is not phase-gated — it cannot distinguish "VAS being cut now" from "VAS is incidentally visible".

![VAS confidence sweep](vas_confidence_sweep.png)

### Embedding-Based Approach

Since the SAM2 image encoder is entirely frozen and demonstrably produces features sufficient for pixel-level VAS segmentation, those same features must encode enough signal to classify whether a frame belongs to the VAS-cutting phase. The encoder has never been updated — it generalises across all phases of the surgery.

**Pipeline (event-agnostic — same scripts handle every annotated phase):**

```
extract_endobag_features.py --event <name>   — sample frames → SAM2 forward_image →
                                                avg+max pool FPN coarse scales → .npz
train_endobag_classifier.py                  — LOCO-CV with logistic regression on the saved features
localize_endobag.py --event <name> --all     — full-video inference → rolling-mean smoothing
                                                → pick highest-confidence segment per case
```

Both scripts accept `--event <name>` where `<name>` is any value from the `event` column of `annotate_fine.csv`. Default is `endobag` for backward compatibility. The output directories are derived from the event name (`<event>_features/`, `<event>_localization/`) so different phases live in separate folders.

The seven currently annotated phases:

| `--event` value | Clinical meaning |
|---|---|
| `endobag`         | Specimen placed in extraction bag (end of nerve-sparing window) |
| `vas_cut_1`       | First (left/right) vas deferens transection |
| `vas_cut_2`       | Second vas deferens transection |
| `catheter_pull`   | Catheter pulled after anterior bladder-neck incision (alternative start marker) |
| `apical_cut`      | Apical dissection — prostate freed from urethra |
| `posterior_cut`   | Posterior dissection — prostate freed from rectum |
| `seminal_peeling` | Seminal vesicle dissection |

Feature extraction: the finest FPN scale is discarded (local texture, not useful for phase detection). For the two remaining coarser scales, average and max pooling are concatenated — average captures mean activation, max captures whether a feature is present anywhere in the frame. Final feature vector: **640-d** = `[fpn[1]-avg(256), fpn[1]-max(256), fpn[2]-avg(64), fpn[2]-max(64)]`. The deepest FPN level is 64 channels because SAM2 internally projects it down to `mem_dim` after the FpnNeck — same layout for every Hiera variant.

Annotations for VAS cutting only: `intuitive_videos/untitled.txt`.
Annotations for all phases (the ones above): `intuitive_videos/annotate_fine.csv`.

### Generalised phase localisation (`run_phase_localization.sh`)

End-to-end orchestrator for any of the phases above — runs feature extraction then LOCO-CV localisation, with the production `--fast` defaults (Hiera-Small, bf16, compile, 4 DataLoader workers, GPU classifier, batch 32):

```bash
# Run a single phase end-to-end
bash run_phase_localization.sh vas_cut_1
bash run_phase_localization.sh apical_cut
bash run_phase_localization.sh catheter_pull

# Override the backbone / disable --fast for a baseline run
SAM2_VARIANT=large FAST=0 bash run_phase_localization.sh apical_cut

# Skip extraction if <event>_features/ is already populated
SKIP_EXTRACT=1 bash run_phase_localization.sh endobag

# Pass extra flags through to localize_endobag.py
bash run_phase_localization.sh catheter_pull --threshold 0.45 --smooth_window 5
```

Outputs land in `<event>_features/` and `<event>_localization/`. The orchestrator is a thin wrapper around the two Python scripts — anything they accept can be passed through.

**Caveats when sweeping phases:**
- LOCO-CV variance grows quickly when a phase has fewer annotated cases than endobag's 7.
- Short events (e.g. `catheter_pull` typically lasts a few seconds) need a smaller `--smooth_window` (default 20s would average the signal away).
- For phases that occur multiple times in close temporal proximity (e.g. `vas_cut_1` and `vas_cut_2`), the "highest mean probability" segment picker may select the wrong instance. Raise `--threshold` to demand stronger confidence.

### Backbone and feature-slice ablation

Run via `bash run_endobag_size_ablation.sh`. Per-frame LOCO-CV (7 cases):

| Variant | Slice | feat_dim | AUC-ROC | F1 | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| Large | all  | 640 | 0.975 ± 0.023 | 0.685 | 0.898 |
| Large | fpn1 | 512 | 0.950 ± 0.070 | 0.710 | 0.896 |
| Large | fpn2 | 128 | 0.907 ± 0.104 | 0.653 | 0.882 |
| **Small** | **all**  | **640** | **0.964 ± 0.050** | **0.757** | **0.926** |
| Small | fpn1 | 512 | 0.930 ± 0.082 | 0.665 | 0.887 |
| Small | fpn2 | 128 | 0.949 ± 0.046 | 0.748 | 0.905 |

Findings: (i) dropping either FPN scale hurts both backbones — keep all 640-d; (ii) Hiera-Small matches Hiera-Large on per-frame AUC (within fold noise) and is the production choice because it wins on full-video localisation and runs **3.57× faster** at the encoder (measured — see throughput benchmark below).

### Encoder throughput benchmark

Pure-GPU `forward_image` throughput at batch 32, 1024×1024 input, H100 80GB. Script: `python3 benchmark.py --skip_pipeline`. Synthetic input pre-allocated on the device — no video decode, no preprocessing, isolates encoder work from I/O.

| Variant | Config             | fps    | ms/frame | Peak GB | Speedup |
|---------|--------------------|-------:|---------:|--------:|--------:|
| Large   | fp32 baseline      |  46.46 |   21.52  |  14.88  |   1.00× |
| Large   | + bf16 autocast    |  85.92 |   11.64  |   9.82  |   1.85× |
| Large   | + bf16 + compile   | 131.96 |    7.58  |   9.30  |   2.84× |
| **Small** | **fp32 baseline**  | 151.55 |    6.60  |   9.71  |   1.00× |
| **Small** | **+ bf16 autocast**| 276.37 |    3.62  |   6.76  |   1.82× |
| **Small** | **+ bf16 + compile**| **471.29** | **2.12** | **6.15** | **3.11×** |

**Key results:**
- bf16 autocast alone is ~1.85× over fp32, with no code changes beyond a one-line `torch.autocast` wrapper. The current extraction scripts run fp32 — wrapping `forward_image` is free throughput.
- `torch.compile` (`mode="default"`) on top of bf16 gives another ~1.5–1.7×, total ~3× over the fp32 baseline. Cost: ~30s of graph capture at startup.
- **Hiera-S vs Hiera-L, best config vs best config: 3.57×.** The earlier "~5×" estimate (based on FLOPs ratio) was too optimistic — H100 is latency-bound at this batch size, so wall-clock ratio is closer to ~3.5× than the raw FLOPs would suggest.
- Throughput here is the encoder alone. End-to-end extraction throughput including `cv2.set/read/resize` will be lower since video I/O does not benefit from bf16 or compile.

### End-to-end pipeline optimisation (`localize_endobag.py --fast`)

The encoder benchmark above measures the GPU in isolation. The real localisation pipeline is bounded by `min(decode, encoder)` *and* by serial CPU work between batches. To exercise the full path, `localize_endobag.py` accepts a `--fast` flag that bundles four optimisations together (each is also individually toggleable):

| Flag                | What it does |
|---------------------|--------------|
| `--use_bf16`        | wraps `forward_image` in `torch.autocast("cuda", dtype=torch.bfloat16)` |
| `--use_compile`     | `torch.compile(forward_image, mode="default")` + 5-iter warmup |
| `--num_workers 4`   | DataLoader with an `IterableDataset` that splits the frame-index list into 4 contiguous chunks; each worker does one initial seek then sequential `cv2.grab()`/`cv2.read()` — matches the baseline's access pattern but in parallel |
| `--gpu_classifier`  | runs the logistic regression on GPU (`feats @ w + b → sigmoid`) so features never leave the device and sklearn never blocks the main thread between batches |

Plus the default `--batch_size` was raised from 8 to 32 (amortises per-batch CPU/sync overhead).

**Measured end-to-end (Hiera-S, case 213 hold-out, 0.5 fps sampling, H100):**

| Config | End-to-end fps | Wall-clock | Speedup |
|---|---:|---:|---:|
| Baseline (inline loop, fp32, sklearn on CPU, batch 8) | 21.22 | 202s | 1.00× |
| **`--fast` (bf16 + compile + nw=4 + GPU clf + batch 32)** | **38.46** | **112s** | **1.81×** |

Selected segment was identical between the two runs (case 213: 4096–4110s, +3s start error).

**Why only 1.8× when the components were ~3× each?**
Amdahl's law on the pipeline composition. The encoder benchmark assumed features stay on GPU forever; the decode benchmark assumed an instant consumer. In practice each batch passes through CPU work between GPU launches (sklearn `predict_proba`, numpy z-score, IPC of the [B, 3, 1024, 1024] tensor from worker to main, tqdm/list overhead). Even after `--gpu_classifier` removes the sklearn step and the `.cpu()` sync, the worker→main IPC and per-batch fixed costs remain. Headroom estimate: encoder is at 38/471 ≈ 8% of its peak, so the next ~2× would come from bigger batches and/or moving decode to NVDEC (TorchCodec) — see `install_torchcodec_isolated.sh`.

### Results

So far only `endobag` and `vas_cut` have been swept across all annotated cases; the same pipeline (`run_phase_localization.sh <event>`) extends to every other phase listed above. Results below are with the production `--fast` defaults (Hiera-Small, bf16, compile, 4 workers, GPU classifier, batch 32).

**VAS cutting localisation:** within 1–2 minutes of the true event across held-out cases.

**Endobagging localisation (Hiera-S, 7 cases, LOCO-CV):**

| Case | GT start | Selected | Start error | Within 3 min |
|------|----------|----------|-------------|--------------|
| 213 | 4093s (68.2m) | 4096s | +3s | ✓ |
| 214 | 3159s (52.6m) | 3304s | +145s | ✓ |
| 219 | 2632s (43.9m) | 2660s | +28s | ✓ |
| 220 | 3382s (56.4m) | 3386s | +4s | ✓ |
| 222 | 4742s (79.0m) | 4870s | +128s | ✓ |
| 245 | 5501s (91.7m) | 4724s | −777s | ✗ |
| 246 | 5096s (84.9m) | 5100s | +4s | ✓ |

**6/7 (86%) within 3 minutes.** Median error 28s, mean 156s. For comparison, Hiera-L on the same protocol was 5/7 with median 126s, mean 813s — the smaller backbone gives both higher hit rate and ~5× tighter localisation. The remaining failure (case 245) is a structurally similar earlier scene scoring higher than the true endobag; since endobagging is by definition the last sustained event in the surgery, switching `pick_best_segment` to "latest segment above threshold" is the next mitigation.

### Future direction: hard-negative mining from post-event field asymmetry

Several phases share a useful structural property: the surgical field looks **dramatically different** after the event ends than before it began. This asymmetry is a stronger signal than the event itself and can be exploited to sharpen the prediction boundary — specifically the *end* boundary, which is what matters clinically.

**Why ENDs matter more than STARTs for these phases.** The start of most dissection-type events is gradual: the surgeon may approach the structure, retract, peek, and adjust for a minute or more before committing to the cut. Any frame in that ~1-minute preamble could plausibly be labelled "event starting" — it is intrinsically ambiguous. The *end* is unambiguous: the structure is either transected, freed, or extracted. Optimising for sharp end-boundary prediction is the right objective; start-time error is dominated by inherent annotation noise.

**Per-phase post-event visual signatures (strong hard negatives):**

| Phase | After the event ends |
|---|---|
| `endobag`        | Prostate no longer visible; pelvic cavity is empty of the specimen |
| `apical_cut`     | Prostate cut loose and mobile; apical anatomy fundamentally changed |
| `catheter_pull`  | Catheter cleanly visible and static, no longer being drawn through tissue |
| `posterior_cut`  | Posterior plane fully developed; prostate floats free of rectum |
| `vas_cut_*`      | Cut end of VAS visible; characteristic stump pattern |

For `catheter_pull` specifically the temporal pattern is also asymmetric in a useful way: *during* the pull the field is constantly changing (catheter length in view varies frame to frame) while *after* the pull the catheter is fully visible and static. Both pre-pull (no catheter) and post-pull (static catheter) are negatives, but they look different from each other and from the dynamic pull itself.

**Mining strategy (to be implemented):**
The current `build_sample_plan` in `extract_endobag_features.py` draws negatives uniformly across all non-event frames. For events with strong post-event signatures, this underweights the most discriminating negatives. The proposed change:

1. For each annotated event, define a post-event window of length comparable to the event itself (e.g. `end_sec` to `end_sec + 60s`).
2. Oversample negatives from this window (e.g. 50% of all negatives drawn from post-event windows, 50% drawn uniformly from elsewhere).
3. Expect the classifier to learn a sharper decision boundary specifically around the event *end*, which directly improves `pick_best_segment` accuracy for the end timestamp.

This complements the "pick latest segment above threshold" mitigation noted above for endobag: that heuristic exploits a temporal prior in post-processing, while hard-negative mining tightens the probability trace itself.

### Multi-class phase localiser (v2 pipeline)

Rather than running a separate binary classifier per event, the v2 pipeline sweeps the SAM2 encoder once per video and jointly localises all 5 surgical milestones with a single 11-class softmax + Viterbi state machine.

**11 emission classes** (6 events + 5 interphases, chronological):
`pre_catheter_pull → catheter_pull → post_catheter_pull → posterior_cut → post_posterior_cut → vas_cut → post_vas_cut → apical_cut → post_apical_cut → endobag → post_endobag`

`vas_cut_1` and `vas_cut_2` CSV rows are both mapped to the single `vas_cut` class — left vs right is not visually distinguishable from FPN features, and keeping them separate would cause the softmax to fight itself. The gap between the two cuts gets the `post_vas_cut` interphase label.

**Viterbi state machine:** 11 states (1:1 to classes), stay + advance-by-1 transitions, plus 2 skip edges: `catheter_pull → posterior_cut` (1→3) and `apical_cut → endobag` (7→9). No vas skip edge — vas_cut is a single sustained state.

**v1 classifier (logreg):** `extract_multiclass_features.py` → `train_multiclass_classifier.py` → `localize_multiclass.py`. Sklearn `LogisticRegression` with `class_weight="balanced"`, per-frame `sample_weight` (hard-negative 5×, safety-gap zeroed), LOCO-CV. Orchestrator: `run_multiclass_localization.sh`.

**v2 classifier (PyTorch BiGRU):** `extract_multiclass_features_v2.py` → `train_multiclass_v2.py` → `validate_multiclass_v2.py`. Replaces logreg with `AttnPoolBiGRU` (attention pool over FPN spatial map → 320-d → BiGRU(128 hidden) → 11-class softmax) to handle OOD videos. Adds augmentation at extraction time: HUD-panel paste, letterbox/pillarbox bars, gaussian blur, unsharp mask, colour jitter, JPEG quality jitter. Orchestrator: `run_v2_pipeline.sh`.

---

### OOD Generalisation: Sony/gg1 Videos

The multiclass localiser (SAM2 features + LR + Viterbi) performs well on the 15 in-distribution intuitive_videos cases but **collapses completely** on Sony DVD-recorder videos (`gg1_videos_daniel/SUBJ_*`).

**Failure mode:** Emissions are flat across all frames — `post_endobag` dominates everywhere (~0.5 mean probability), target event classes stay <0.05 inside GT windows. Viterbi collapses all 5 events into zero-width windows within the first ~845s of a 158-minute recording.

**Root cause — high-dimensional feature-space gap:**

- Domain classifier (linear, 640-d SAM2 features): train-vs-OOD AUC = **1.0** — for clean features, augmented features, and every FPN scale independently. Mean per-dim shift ~1σ.
- Augmentation (`extract_multiclass_features_aug.py`) undershoots: |aug shift| = 2.09 vs |true domain shift| = 7.05, cosine similarity = 0.52 → right direction but only ~30% of the required magnitude.
- Gap spans at least **7 linear directions** — projecting them all out still leaves domain AUC 0.95. Per-case z-score normalisation (CORAL-lite) does not recover event signal — the gap is nonlinear and high-dimensional, not a per-channel affine shift.

**Four fixes tested — all failed:**

| Fix | Result |
|---|---|
| Crop to surgical bbox (remove HUD and black bars) | Removed spurious off-window spikes but created no in-window event signal |
| + Histogram-match Sony → intuitive (global per-channel LUT) | Made traces *flatter* (max p: 0.075 → 0.0001). Photometric normalisation exhausted. |
| Per-case feature z-score normalisation (classifier-side) | Still fails — gap is not a per-dim affine shift |
| DINOv2 ViT-B/14 (invariance-trained backbone, `domain_auc_gate.py`) | Tissue-only AUC = **0.989**, surgical-bbox AUC = **1.000** — an invariance-trained backbone does not bridge the gap |

An in-distribution control (case 213 catheter_pull, in training set) showed probe ratio ~3000× (p_in 0.58 vs p_out 0.0002) — confirming the pipeline machinery is sound and the Sony flatline is a genuine domain gap, not a code bug.

**Verdict:** The cross-recorder gap is intrinsic and high-dimensional in every encoder tried (SAM2 raw, per-case-normalised SAM2, DINOv2). No pixel normalisation, classifier fix, or backbone swap works zero-shot. The logistic regression's linearity was never the bottleneck — it made the gap *legible*. The v2 PyTorch BiGRU classifier (`train_multiclass_v2.py`) addresses the classifier capacity side, but the feature-space shift is the binding constraint.

**Remaining levers — both require domain-aware adaptation:**
1. **(Recommended)** Annotate 1–2 Sony cases → mixed-domain training. Highest confidence fix.
2. Unsupervised domain alignment (adversarial / CORAL) using the 18 unlabelled Sony recordings, but still needs at least some labelled Sony cases to validate.

Probe outputs: `multiclass_smoking_gun/probe/`. Aug localisation results: `multiclass_localization_aug/`. Key scripts: `probe_crop_spike.py`, `domain_auc_gate.py`, `test_dv5_hypothesis.py`, `test_crop_hypothesis.py`.

---

### Scripts

| Script | Purpose |
|--------|---------|
| `extract_endobag_features.py --event <name>` | SAM2 feature extraction for any annotated phase. Default `--event endobag`. Output → `<event>_features/` |
| `train_endobag_classifier.py` | LOCO-CV classifier evaluation on saved features (read `--features_dir` to switch phases) |
| `localize_endobag.py --event <name> --all` | Full-video LOCO-CV localisation for any phase. Default `--event endobag`. Output → `<event>_localization/` |
| `run_phase_localization.sh <event>` | Orchestrator: extract → localise for one phase end-to-end, `--fast` defaults baked in |
| `extract_vas_features.py` / `localize_vas.py` | Legacy VAS-only scripts (kept for the `untitled.txt` annotation format) |
| `run_endobag_size_ablation.sh` | Backbone × FPN-slice ablation: Hiera-S extraction + LOCO-CV across `{large, small} × {all, fpn1, fpn2}` |
| `benchmark.py` | Encoder forward + end-to-end pipeline benchmarks (cv2 serial, cv2+DataLoader, TorchCodec if installed) |
| `install_torchcodec_isolated.sh` | Optional: `--target` pip install of TorchCodec into `.torchcodec_env/` so the container's torch is untouched |

---

### Phase 9 — VAS Deferens Detector + AUA Tracking Toolkit

**VAS deferens detector:** Trained a second SAM3 detector on vas deferens annotations using the same architecture and training procedure as the prostate detector. The text query at inference is `"prostate gland"` — identical to the prostate model. The CLIP language backbone is frozen, so the text embedding is just a fixed class label; the model learns to associate it with whatever structure appears in the annotations. Swapping the query string at inference time would break the trained association.

Inference: `infer_prostate_bidir.py` propagates bidirectionally from a detected anchor frame (forward + backward without `reverse=True`, using `ReverseVideoChunkLoader` for the backward pass). `infer_vas.sh` wraps this for a single clip; `infer_vas_batch.sh` iterates over the clip list in `intuitive_videos/untitled.txt`.

**AUA tracking toolkit (`aua_tracking/`):** Browser-based annotation tools and SAM2 propagation scripts consolidated for the AUA presentation. Covers three workflows: brush-mask tracking (nerve bundles), single-instance box tracking (VAS/seminals/retrotrigonal), and simultaneous multi-instance box tracking (both VAS at once, both seminals at once). See `aua_tracking/README.md` for full documentation.
