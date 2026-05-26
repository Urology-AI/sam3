# AUA Tracking — Surgical Structure Segmentation

Tools used to track anatomical structures (nerve bundles, vas deferens, seminal vesicles, retrotrigonal layers, etc.) in 3D side-by-side robotic surgery video for the AUA presentation.

General workflow: **annotate in the browser → export JSON → run SAM2 propagation → overlay video**.

There are three distinct workflows depending on what you are tracking:

| Workflow | Annotate with | Run with | Use when |
|----------|--------------|----------|----------|
| Brush mask — single or multiple structures | `annotate_brush.html` | `track_nerves.py` / `track_nerves_ffmpeg.py` / `track_nerves_multi.py` | Nerve bundles; anything where you want to paint a freeform mask |
| Box — multiple structure types, one instance each | `annotate_multi.html` | `track_multi_structure.py` | VAS, seminals, retrotrigonal layer — one of each visible at a time |
| Box — multiple instances of the same structure | `annotate_instances.html` | `track_instances.py` | Both VAS simultaneously, both seminals simultaneously, etc. |

---

## All files

| File | Role |
|------|------|
| `annotate_brush.html` | Paint brush masks on keyframes; exports JSON with base64 mask PNGs. |
| `track_nerves.py` | Propagates brush-mask annotations. OpenCV frame reading (MP4). |
| `track_nerves_ffmpeg.py` | Same as above but uses ffmpeg seeking. **Use for MOV files.** |
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

### Propagation: `track_nerves.py` / `track_nerves_ffmpeg.py`

```bash
python3 track_nerves_ffmpeg.py \          # or track_nerves.py for MP4
  --video   aua_videos/case_X.mov \
  --tracks  aua_boxes/case_X_mask_tracks.json \
  --sbs_eye left \
  --frame_step 1 \
  --mask_alpha 0.30 \
  --render_from_first_track
```

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

## Why two scripts? The OpenCV timestamp bug

`track_nerves.py` and `track_multi_structure.py` use `cv2.VideoCapture` with `CAP_PROP_POS_FRAMES` for seeking. On some `.mov` files (specifically `Mochita_3D.mov`) this produced a **2× timestamp mismatch** — OpenCV's frame-count seeking landed on roughly double the intended frame, so a mask drawn on frame N was applied to frame ~2N. The cause is that some MOV containers store a frame count that does not match the actual presentation timestamps used by decoders.

`track_nerves_ffmpeg.py` fixes this by converting every frame index to a presentation timestamp (`frame_idx / fps_ann`) and passing it to ffmpeg via `-ss`. ffmpeg honours the container's PTS metadata and always seeks accurately. The rendering pass also switches from `cv2.VideoCapture` to an ffmpeg pipe for the same reason — if propagation and rendering seek differently, masks land on mismatched frames.

**Rule of thumb:** `.mp4` → OpenCV is fine. `.mov` → use the ffmpeg variant.
