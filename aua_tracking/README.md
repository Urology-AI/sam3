# AUA Tracking — Surgical Structure Segmentation

Tools used to track anatomical structures (nerve bundles, vas deferens, seminal vesicles, retrotrigonal layers, etc.) in 3D side-by-side robotic surgery video for the AUA presentation.

General workflow: **annotate in the browser → export JSON → run SAM2 propagation → overlay video**.

There are three distinct workflows depending on what you are tracking:

| Workflow | Annotate with | Run with | Use when |
|----------|--------------|----------|----------|
| Brush mask — SBS 3D video (recommended) | `annotate_brush.html` | `track_nerves_ffmpeg_v2.py` | Nerve bundles in 3D side-by-side video |
| Brush mask — plain 2D video | `annotate_brush_2d.html` | `track_nerves_2d.py` | Nerve bundles in 2D (non-SBS) video |
| Box — multiple structure types, one instance each | `annotate_multi.html` | `track_multi_structure.py` | VAS, seminals, retrotrigonal layer — one of each visible at a time |
| Box — multiple instances of the same structure | `annotate_instances.html` | `track_instances.py` | Both VAS simultaneously, both seminals simultaneously, etc. |

---

## All files

| File | Role |
|------|------|
| `annotate_brush.html` | Paint brush masks on keyframes; exports JSON with base64 mask PNGs. |
| `annotate_brush_2d.html` | Same as above, just relabelled for 2D (non-SBS) source video. Drawing and export are identical; only the export-filename suffix differs (`_mask_tracks_2d.json`). |
| `track_nerves.py` | Propagates brush-mask annotations. OpenCV frame reading (MP4). Has fps-ratio bugs — superseded by `track_nerves_ffmpeg_v2.py`. |
| `track_nerves_ffmpeg.py` | First ffmpeg variant. Still has the renderer fps-ratio misalignment described below — superseded by `_v2`. Kept for reference. |
| `track_nerves_ffmpeg_old.py` | Original ffmpeg variant. Same renderer bug; mask-saving stride bug also present. Kept for reference / diff against `_v2`. |
| `track_nerves_ffmpeg_v2.py` | **Recommended for SBS 3D video.** Fixes mask sub-chunk sizing, mask-saving stride, AND renderer–frame alignment when `fps_actual != fps_ann` (see below). |
| `track_nerves_2d.py` | **Recommended for plain 2D video.** Same fixes as `_v2`, with all SBS / eye-cropping logic removed. |
| `track_nerves_multi.py` | Like `track_nerves.py` but accepts multiple JSON files, loads SAM2 once. |
| `annotate_multi.html` | Draw one bounding box per structure type per keyframe; exports JSON. |
| `track_multi_structure.py` | Propagates box annotations; one instance per track, one SAM2 object per segment. |
| `annotate_instances.html` | Draw multiple bounding boxes per frame, one per visible instance; exports JSON with `instance_idx` per box. |
| `track_instances.py` | Propagates multi-instance box annotations; all instances tracked simultaneously in one SAM2 session with separate object IDs. |
| `annotate_boxes.html` | Simple single-object box annotator; draw one box per keyframe, exports flat JSON list. |
| `track_from_boxes.py` | Plain SAM2-large box tracking. No fine-tuned models. Uses `annotate_boxes.html` JSON; reinitialises from the annotated box at each keyframe. |
| `track_from_boxes_finetuned.py` | Same as above but passes each box through the fine-tuned SAM2 decoder first to get a clean seed mask before propagation. |
| `heatmap_fast.py` | AUA-specific one-off: full SAM3 detector → fine-tuned decoder → SAM2 propagation, but rendered as a radial heatmap (red→green→blue from centre) with a fade-in/fade-out schedule. Hardcoded to `aua_videos/media7_0_13.mp4`. |

---

## Workflow 1 — Brush mask tracking (nerve bundles)

### Annotation: `annotate_brush.html`

Open directly in Chrome/Firefox (no server needed).

1. Load a video file from disk.
2. Create a named track for each structure (e.g. "left nerve bundle", "right nerve bundle").
3. Scrub to a keyframe and paint a brush mask over the structure.
4. Add more keyframes within a track if the structure moves substantially — the tracker reinitialises from each one.
5. Set the track start/end frame range.
6. Export → saves a `.json` with frame indices, brush mask PNGs (base64), colors, and labels.

### Propagation: `track_nerves_ffmpeg_v2.py` (SBS 3D) / `track_nerves_2d.py` (plain 2D)

```bash
# SBS 3D — annotate on the chosen eye, propagate that eye, mirror the overlay to the other eye
python3 track_nerves_ffmpeg_v2.py \
  --video   aua_videos/case_X.mp4 \
  --tracks  aua_boxes/case_X_mask_tracks.json \
  --sbs_eye left \
  --mask_alpha 0.10 \
  --render_from_first_track

# Plain 2D — full frame in, full frame out
python3 track_nerves_2d.py \
  --video   aua_videos/case_2d.mp4 \
  --tracks  aua_boxes/case_2d_mask_tracks_2d.json \
  --mask_alpha 0.10 \
  --render_from_first_track
```

Both default to pure green at α=0.10 — passing `--mask_color` overrides this. The older `track_nerves.py` / `track_nerves_ffmpeg.py` / `_old.py` variants are kept for reference but have the fps-ratio bugs described at the bottom of this file.

| Flag | Default | Description |
|------|---------|-------------|
| `--sbs_eye` | `left` | Which half of a side-by-side 3D video was annotated. `none` for plain 2D. |
| `--frame_step` | `1` | Propagate every Nth frame. `2` or `3` for speed at minor quality cost. |
| `--mask_alpha` | `0.10` | Overlay opacity (0–1). |
| `--render_from_first_track` | off | Clip output to the annotated frame range instead of rendering the whole video. |
| `--clip_to_tracks` | off | Render from frame 0 to last annotation + `--clip_tail_s` seconds. |
| `--mask_color` | per-track | Override all track colors with a hex value, e.g. `#ff4444`. |
| `--track_ids` | all | Comma-separated track IDs to run, e.g. `0,2`. |
| `--no_compile` | off | Disable `torch.compile` (slower but easier to debug). |

Output: `masks/track_NN/*.npy` + `overlay_sbs.mp4` (SBS) or `overlay.mp4` (2D).

Each named track runs as a separate SAM2 session. Tracks with masks on multiple keyframes are split into segments, each seeded from its keyframe mask — this corrects drift without re-running from scratch. Segments longer than 600 frames are sub-chunked automatically, carrying the last predicted mask forward.

**Which script to use:**
- `.mp4` → `track_nerves.py` (simpler, OpenCV)
- `.mov` → `track_nerves_ffmpeg.py` (see [OpenCV timestamp bug](#why-two-scripts-the-opencv-timestamp-bug) below)

### Variant: `track_nerves_multi.py` — multiple JSON files, one model load

Use when you annotated different structures in separate sessions and have two or more JSON files for the same video.

```bash
python3 track_nerves_multi.py \
  --video   aua_videos/case_X.mp4 \
  --tracks  aua_boxes/case_left_nerve.json aua_boxes/case_right_nerve.json \
  --sbs_eye left \
  --render_from_first_track
```

SAM2 is loaded and compiled once, then each JSON is processed in sequence. Each JSON gets its own output subdirectory named after the JSON stem. The render window is clipped to the intersection of all annotation ranges so outputs are temporally aligned.

Note: uses OpenCV (not ffmpeg), so the MOV timestamp bug applies.

---

## Workflow 2 — Box tracking, one instance per structure (`annotate_multi.html` + `track_multi_structure.py`)

Designed for tracking distinct structure types where only one instance of each is visible at a time — e.g. one VAS, one seminal vesicle, the retrotrigonal layer.

### Annotation: `annotate_multi.html`

Open directly in the browser. For each structure, draw a bounding box on one or more keyframes and label it. The JSON stores `tracks[].boxes[]: {frame, box: [x1,y1,x2,y2]}`.

### Propagation: `track_multi_structure.py`

```bash
python3 track_multi_structure.py \
  --video      aua_videos/HHY1_3D_750s_870s.mp4 \
  --tracks     aua_boxes/HHY1_multi_tracks.json \
  --sbs_eye    left \
  --frame_step 1 \
  --mask_alpha 0.30
```

Each track runs as a separate SAM2 session with `obj_id=1` (single object). The annotated box initialises propagation; if multiple boxes are drawn for a track, the track is split into segments at each box and re-initialised from the new box. Sub-chunk carry-over (600-frame limit) applies the same way as workflow 1.

Output: `masks/track_NN/*.npy` + `overlay_sbs.mp4` or `overlay.mp4`.

---

## Workflow 3 — Box tracking, multiple simultaneous instances (`annotate_instances.html` + `track_instances.py`)

Designed for cases where multiple instances of the same structure are visible at the same time — both vas deferens, both seminal vesicles, etc.

### Annotation: `annotate_instances.html`

Open directly in the browser. On each keyframe, draw one bounding box per visible instance of the structure — the tool assigns each an `instance_idx`. The JSON stores `tracks[].boxes[]: {frame, time_s, instance_idx, box: [x1,y1,x2,y2]}`, with multiple boxes per frame allowed.

### Propagation: `track_instances.py`

```bash
python3 track_instances.py \
  --video      aua_videos/case.mov \
  --tracks     aua_boxes/case_instances.json \
  --sbs_eye    left \
  --render_from_first_track
```

The key difference from workflow 2: all instances at a seed frame are added to the **same SAM2 session** with distinct object IDs (`instance_idx + 1`). `propagate_in_video` tracks all of them in one pass — SAM2's memory bank maintains separate object pointers for each. This means the instances are aware of each other during propagation, reducing ID swaps.

Output directories are per-instance: `masks/track_NN_obj_1/`, `masks/track_NN_obj_2/`, etc.

On sub-chunk boundaries, the carry-over switches from boxes to the predicted masks from the previous chunk — one carry mask per object ID.

---

## Workflow 4 — Simple box tracking (`annotate_boxes.html` + `track_from_boxes.py` / `track_from_boxes_finetuned.py`)

The earliest and simplest tracking workflow. One box per keyframe, one structure, no multiple instances.

### Annotation: `annotate_boxes.html`

Open in the browser. Scrub to a frame, draw a box, move to the next keyframe and repeat. Exports a flat JSON list of `{frame, time_s, box: [x1,y1,x2,y2]}` entries (no tracks, no labels — just the box list).

### `track_from_boxes.py` — plain SAM2, no fine-tuning

```bash
python3 track_from_boxes.py \
  --video      aua_videos/HHY1_3D_85s_140s.mp4 \
  --boxes      aua_boxes/HHY1_3D_85s_140s_boxes.json \
  --sbs_eye    left \
  --frame_step 1 \
  --mask_alpha 0.35
```

Uses stock SAM2-large with `add_new_points_or_box`. Reinitialises at each annotated box frame. The rendered overlay shows the reinit box for 1 second at each keyframe. No fine-tuned models required.

### `track_from_boxes_finetuned.py` — fine-tuned decoder seed

Drop-in replacement. Before each SAM2 session it runs the annotated box through the **fine-tuned SAM2 mask decoder** (`sam2_decoder_training/checkpoints/20260504_1506/checkpoint_best.pth`) to produce a cleaner seed mask, then seeds SAM2 with `add_new_mask` instead of `add_new_points_or_box`. The idea: the fine-tuned decoder knows prostate appearance, so the initial mask it produces from a box is tighter than SAM2's default prompt encoder would give, resulting in better propagation.

---

## Workflow 5 — Prostate heatmap (`heatmap_fast.py`)

A one-off visualisation script made for the AUA presentation. Runs the full prostate pipeline (SAM3 detector → fine-tuned decoder → SAM2 propagation, identical to `detect_segment_fast.py`) on a hardcoded clip (`aua_videos/media7_0_13.mp4`), but renders a **radial heatmap** instead of a solid colour overlay:

- Colour: red at the heatmap centre (a random foreground pixel from frame 0), fading through green to blue at the mask edge
- Alpha: ~0.24 (intentionally light)
- Schedule: heatmap is off for the first third of the video, fades in during the second third, fades out and off for the final third

To use on a different clip, edit `VIDEO_CLIP` and `OUTPUT_DIR` at the top of the file.

---

## Path note for scripts in this folder

Scripts that build paths to checkpoints or assets relative to `SAM3_DIR` use `os.path.dirname(os.path.dirname(...))` to resolve to `sam3/` (the parent of `aua_tracking/`). This affects `track_from_boxes_finetuned.py` and `heatmap_fast.py`. Scripts that only use `SAM3_DIR` for `sys.path.insert` (all others) are unaffected by the folder depth.

---

## New scripts: Ureter, Obturator, and other structures

`track_nerves_ffmpeg_v2.py`, `track_nerves_2d.py`, `track_nerves_ffmpeg_old.py`, `benchmark_sam2_frames.py`, and `annotate_brush_2d.html` were created to extend the annotation and tracking workflow beyond nerve bundles — specifically to handle structures such as the **ureter**, **obturator nerve**, and similar anatomy encountered during nerve-sparing RARP.

`track_nerves_ffmpeg_old.py` is the original ffmpeg variant kept for reference / diff. `track_nerves_ffmpeg_v2.py` and `track_nerves_2d.py` are the recommended scripts for SBS 3D and plain 2D video respectively — they carry two bug fixes relative to `track_nerves_ffmpeg.py`:

**FIX 1 — Sub-chunk sizing in actual frames, not annotation frames.**
The old script split by `MAX_CACHED_FRAMES` annotation frames. For a 2× fps ratio this loaded ~2× as many actual frames as expected, filling GPU memory and causing OOM. This version uses `MAX_ACTUAL_FRAMES` (actual video frames) and derives the annotation-frame chunk size as:
```
ann_chunk_size = round(MAX_ACTUAL_FRAMES * fps_ann / fps_actual)
```
so every sub-chunk encodes at most `MAX_ACTUAL_FRAMES` actual frames regardless of the fps ratio.

**FIX 2 — `global_idx` scaled by fps ratio.**
The old script saved mask `k` at annotation frame `sub_start + k`. With `fps_actual > fps_ann` the loader returns more actual frames than annotation frames in the sub-chunk, so later sub-chunks overwrote earlier ones with masks from different video times, causing visible drift at each sub-chunk boundary. The fix:
```
global_idx = sub_start + round(local_idx * frame_step * fps_ann / fps_actual)
```
maps actual-frame indices back to annotation-frame indices correctly. When `fps_actual == fps_ann` the ratio is 1.0 and behaviour is identical to the old script.

`benchmark_sam2_frames.py` measures SAM2 encoder throughput on a set of frames — used to profile the annotation pipeline and verify that sub-chunk sizing changes did not regress encoding speed.

---

## The fps-ratio rendering bug (fixed in `_v2` / `_2d`)

This bug bit us on videos where the source fps differs from the annotation fps — e.g. a 59.94 fps video annotated at 30 fps (`fps_ratio ≈ 2`). The old scripts (`track_nerves_ffmpeg.py`, `track_nerves_ffmpeg_old.py`) produced misaligned overlays — sometimes silently for the first track and visibly garbage for later tracks.

### What was going wrong

Two independent issues, in two different places:

**1. Mask-saving stride (fixed in `_v2`).**
The propagation loader pulls actual frames from ffmpeg; with `fps_actual = 59.94` and `fps_ann = 30`, a 250-ann-frame sub-chunk loads ~500 actual frames. The old code saved each propagated mask at `global_idx = sub_start + local_idx`, treating an *actual*-frame index as if it were an *annotation*-frame index. The masks therefore covered twice the annotation range they should have, and at sub-chunk boundaries later sub-chunks overwrote earlier ones with masks from different video times.

Fix: `global_idx = sub_start + round(local_idx * fps_ann / fps_actual)` — actual indices mapped back to annotation indices before saving.

**2. Renderer–frame misalignment (the bug this whole writeup is about).**
The render loop iterated over **annotation-frame indices**:

```python
for fidx in range(render_start, render_end):
    frame = next(_render_gen, None)        # consumes one ACTUAL frame from ffmpeg
    mask = mask_cache.get_mask_at(fidx)    # indexed by ann frame
```

But each iteration consumes one **actual** frame from the ffmpeg pipe (which yields at source fps). So after K iterations the underlying video had advanced by `K/fps_actual` seconds (≈ `K/2` annotation frames at fps_ratio=2), but the mask lookup used `fidx = render_start + K` (K annotation frames). The mask "ran" at twice the speed of the video content underneath it.

In `_old` this cancelled with bug #1 — both indexing schemes were off by the same factor — but **only** when a track's `sub_start` equalled `render_start`. That's why track 0 looked fine (its `sub_start == render_start`) while track 1 (starting later in the video) produced garbage. `_v2` fixed bug #1 in mask saving, which exposed bug #2 in rendering for *all* tracks.

Fix: iterate over actual frames and compute a fractional annotation-frame coordinate per frame:

```python
n_render_actual = int(round((render_end - render_start) * fps / fps_ann))
for out_idx in range(n_render_actual):
    frame = next(_render_gen, None)
    ann_fidx = render_start + out_idx * fps_ann / fps   # float
    mask = mask_cache.get_mask_at(ann_fidx)             # bisect handles floats
```

`LazyMaskCache.get_mask_at` already does linear interpolation between adjacent keyframes and accepts any numeric index, so no other change was needed.

**Symptoms in the wild:** the overlay appears to drift, "rush ahead" of the underlying anatomy, or paint masks during black/gap frames where nothing should be highlighted. If you see this on `_v2` output: check `fps_ann` in the JSON matches what you annotated at, and re-run with `_v2` or `_2d`.

---

## Why two scripts? The OpenCV timestamp bug

`track_nerves.py` and `track_multi_structure.py` use `cv2.VideoCapture` with `CAP_PROP_POS_FRAMES` for seeking. On some `.mov` files (specifically `Mochita_3D.mov`) this produced a **2× timestamp mismatch** — OpenCV's frame-count seeking landed on roughly double the intended frame, so a mask drawn on frame N was applied to frame ~2N. The cause is that some MOV containers store a frame count that does not match the actual presentation timestamps used by decoders.

`track_nerves_ffmpeg.py` fixes this by converting every frame index to a presentation timestamp (`frame_idx / fps_ann`) and passing it to ffmpeg via `-ss`. ffmpeg honours the container's PTS metadata and always seeks accurately. The rendering pass also switches from `cv2.VideoCapture` to an ffmpeg pipe for the same reason — if propagation and rendering seek differently, masks land on mismatched frames.

**Rule of thumb:** `.mp4` → OpenCV is fine. `.mov` → use the ffmpeg variant.
