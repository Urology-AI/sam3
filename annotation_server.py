"""
SAM3 + Grounding DINO Annotation Server
========================================
Combined model loader with lazy caching.
Both models are loaded once on first request and kept in GPU memory
for the lifetime of the server process.

Usage (inside Singularity container):
    python annotation_server.py --port 8890 --experiments-dir ./experiments

Then tunnel:
    ssh -J gahaln01@minerva.hpc.mssm.edu -L 8892:127.0.0.1:8892 gahaln01@lg03e03
    Open http://localhost:8890 in your browser.
"""

import argparse
import base64
import io
import json
import os
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, request, send_from_directory, send_file, session, redirect, url_for

# ---------------------------------------------------------------------------
# Model cache — singleton pattern
# ---------------------------------------------------------------------------

class ModelCache:
    """
    Lazy-loading, persistent cache for DINO and SAM3 models.
    Models are loaded on first use and stay in GPU memory.
    """

    def __init__(self):
        self._dino_model = None
        self._dino_processor = None
        self._sam3_model = None
        self._sam3_processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelCache] Using device: {self._device}")

    @property
    def device(self):
        return self._device

    # ---- DINO ----

    def get_dino(self):
        """Return (model, processor) for Grounding DINO, loading on first call."""
        if self._dino_model is None:
            print("[ModelCache] Loading Grounding DINO (first call)...")
            t0 = time.time()
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            model_id = "IDEA-Research/grounding-dino-tiny"
            self._dino_processor = AutoProcessor.from_pretrained(model_id)
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self._device)
            print(f"[ModelCache] DINO loaded in {time.time() - t0:.1f}s")
        return self._dino_model, self._dino_processor

    # ---- SAM3 ----

    def get_sam3(self):
        """Return (model, processor) for SAM3, loading on first call."""
        if self._sam3_model is None:
            print("[ModelCache] Loading SAM3 (first call)...")
            t0 = time.time()
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self._sam3_model = build_sam3_image_model()
            self._sam3_processor = Sam3Processor(self._sam3_model)
            print(f"[ModelCache] SAM3 loaded in {time.time() - t0:.1f}s")
        return self._sam3_model, self._sam3_processor

    def warmup(self):
        """Pre-load both models at server start (optional)."""
        print("[ModelCache] Warming up — loading both models...")
        self.get_dino()
        self.get_sam3()
        print("[ModelCache] Warmup complete.")


# Global singleton
cache = ModelCache()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", uuid.uuid4().hex)

# Password — set via env var or CLI arg, defaults to "annotate"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "annotate")

# Base path for file browsing — users cannot browse above this
BROWSE_ROOT = Path(os.environ.get("BROWSE_ROOT", "/sc/arion/projects/video_rarp"))

# Config via env var or CLI args
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-load models at import time
cache.warmup()
print(f"[Startup] Experiments dir: {EXPERIMENTS_DIR.resolve()}")
print(f"[Startup] Browse root: {BROWSE_ROOT}")
print(f"[Startup] Models pre-loaded on {cache.device}")


def login_required(f):
    """Decorator to require authentication on routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            # For API calls, return 401
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_experiment_dirs(experiment_name: str) -> Path:
    """Create and return the experiment root directory with sub-folders."""
    exp_dir = EXPERIMENTS_DIR / experiment_name
    (exp_dir / "frames").mkdir(parents=True, exist_ok=True)
    (exp_dir / "masks").mkdir(parents=True, exist_ok=True)
    (exp_dir / "annotations").mkdir(parents=True, exist_ok=True)
    return exp_dir


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def mask_to_base64_png(mask_np: np.ndarray) -> str:
    """Convert a boolean/binary mask to a base64-encoded PNG."""
    mask_uint8 = (mask_np.astype(np.uint8)) * 255
    img = Image.fromarray(mask_uint8, mode="L")
    return pil_to_base64(img, fmt="PNG")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET"])
def login_page():
    return send_from_directory("static", "login.html")


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    if data.get("password") == APP_PASSWORD:
        session["authenticated"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Wrong password"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Static frontend (protected)
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
@login_required
def static_files(filename):
    return send_from_directory("static", filename)


# ---------------------------------------------------------------------------
# File browser
# ---------------------------------------------------------------------------

@app.route("/api/browse", methods=["POST"])
@login_required
def browse_filesystem():
    """
    Browse directories and files under BROWSE_ROOT.
    Expects JSON: { path: "relative/path" } or { path: "" } for root.
    Returns: { dirs: [...], files: [...], current_path: "...", parent_path: "..." }
    """
    data = request.get_json()
    rel_path = data.get("path", "").strip().strip("/")

    # Resolve and enforce jail to BROWSE_ROOT
    browse_path = (BROWSE_ROOT / rel_path).resolve()
    if not str(browse_path).startswith(str(BROWSE_ROOT.resolve())):
        return jsonify({"error": "Access denied"}), 403

    if not browse_path.is_dir():
        return jsonify({"error": "Not a directory"}), 404

    # List contents
    dirs = []
    files = []
    try:
        for entry in sorted(browse_path.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.is_file():
                suffix = entry.suffix.lower()
                size_mb = entry.stat().st_size / (1024 * 1024)
                files.append({
                    "name": entry.name,
                    "size_mb": round(size_mb, 2),
                    "type": suffix,
                })
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    # Compute parent path (relative to BROWSE_ROOT)
    parent_path = ""
    if browse_path != BROWSE_ROOT.resolve():
        parent_path = str(browse_path.parent.relative_to(BROWSE_ROOT.resolve()))
        if parent_path == ".":
            parent_path = ""

    # Current path relative to BROWSE_ROOT
    current_rel = str(browse_path.relative_to(BROWSE_ROOT.resolve()))
    if current_rel == ".":
        current_rel = ""

    return jsonify({
        "dirs": dirs,
        "files": files,
        "current_path": current_rel,
        "full_path": str(browse_path),
        "parent_path": parent_path,
        "is_root": browse_path == BROWSE_ROOT.resolve(),
    })


# ---------------------------------------------------------------------------
# Experiment management
# ---------------------------------------------------------------------------

@app.route("/api/experiments", methods=["GET"])
@login_required
def list_experiments():
    """List all existing experiments."""
    if not EXPERIMENTS_DIR.exists():
        return jsonify({"experiments": []})
    exps = sorted([
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir()
    ])
    return jsonify({"experiments": exps})


@app.route("/api/experiments", methods=["POST"])
@login_required
def create_experiment():
    """Create a new experiment (or reuse existing)."""
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Experiment name is required"}), 400
    # Sanitize: replace spaces with underscores, lowercase
    name = name.replace(" ", "_").lower()
    exp_dir = ensure_experiment_dirs(name)
    return jsonify({
        "name": name,
        "path": str(exp_dir),
        "message": "Experiment ready",
    })


# ---------------------------------------------------------------------------
# Video browsing and serving
# ---------------------------------------------------------------------------

@app.route("/api/videos", methods=["POST"])
@login_required
def list_videos():
    """List video files in a given directory on the server."""
    data = request.get_json()
    video_dir = data.get("path", "")
    if not os.path.isdir(video_dir):
        return jsonify({"error": "Directory not found"}), 404
    files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))
    ])
    return jsonify({"videos": files, "path": video_dir})


@app.route("/api/video_file")
@login_required
def serve_video():
    """Stream a video file from the server filesystem."""
    filepath = request.args.get("path", "")
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath)


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

@app.route("/api/capture", methods=["POST"])
@login_required
def capture_frame():
    """
    Save a captured frame.
    Expects JSON: { experiment, frame_b64, video_name, timestamp }
    """
    data = request.get_json()
    experiment = data["experiment"]
    frame_b64 = data["frame_b64"]
    video_name = data.get("video_name", "unknown")
    timestamp = data.get("timestamp", 0)

    exp_dir = ensure_experiment_dirs(experiment)
    frame_id = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    frame_path = exp_dir / "frames" / f"{frame_id}.png"

    img = base64_to_pil(frame_b64)
    img.save(str(frame_path))

    return jsonify({
        "frame_id": frame_id,
        "frame_path": str(frame_path),
        "width": img.width,
        "height": img.height,
    })


# ---------------------------------------------------------------------------
# DINO inference
# ---------------------------------------------------------------------------

@app.route("/api/dino", methods=["POST"])
@login_required
def run_dino():
    """
    Run Grounding DINO on a frame.
    Expects JSON: { frame_path, prompt, threshold? }
    Returns: list of { box: [x1,y1,x2,y2], score, label }
    """
    data = request.get_json()
    frame_path = data["frame_path"]
    prompt = data["prompt"]
    threshold = data.get("threshold", 0.3)

    model, processor = cache.get_dino()

    image = Image.open(frame_path).convert("RGB")

    # Format prompt for DINO (period-separated labels)
    labels = [l.strip() for l in prompt.split(",")]
    text = ". ".join(labels) + "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    candidates = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        candidates.append({
            "box": [round(x, 2) for x in box.tolist()],
            "score": round(score.item(), 4),
            "label": label,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    return jsonify({"candidates": candidates, "count": len(candidates)})


# ---------------------------------------------------------------------------
# SAM3 inference
# ---------------------------------------------------------------------------

@app.route("/api/sam3", methods=["POST"])
@login_required
def run_sam3():
    """
    Run SAM3 on a frame with flexible prompt combinations.

    Accepts JSON with any combination of:
      - text_prompt: "..."
      - box: [x1, y1, x2, y2]  (pixel coords)
      - points: [{ x, y, label }]  (label: 1=positive, 0=negative, pixel coords)

    Returns: list of { mask_b64, score }
    """
    data = request.get_json()
    frame_path = data["frame_path"]
    box = data.get("box")              # [x1, y1, x2, y2] or None
    text_prompt = data.get("text_prompt")  # string or None
    points = data.get("points")        # [{ x, y, label }] or None

    model, processor = cache.get_sam3()
    image = Image.open(frame_path).convert("RGB")
    img_w, img_h = image.size

    has_text = bool(text_prompt)
    has_box = bool(box)
    has_points = bool(points) and len(points) > 0

    if not has_text and not has_box and not has_points:
        return jsonify({"error": "Provide at least one of: text_prompt, box, or points"}), 400

    # ── Text-only path (custom sam3 API) ──
    if has_text and not has_box and not has_points:
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks = output["masks"]
        out_boxes = output.get("boxes", [])
        scores = output["scores"]

    # ── Box prompt (with optional text) via custom sam3 API ──
    elif has_box and not has_points:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        box_cxcywh = [cx, cy, bw, bh]

        inference_state = processor.set_image(image)
        if has_text:
            # Set text first, then refine with box
            processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        output = processor.add_geometric_prompt(
            box=box_cxcywh,
            label=True,
            state=inference_state,
        )
        masks = output["masks"]
        out_boxes = output.get("boxes", [])
        scores = output["scores"]

    # ── Points (with optional box and text) via add_geometric_prompt ──
    elif has_points:
        inference_state = processor.set_image(image)

        if has_text:
            processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        if has_box:
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            processor.add_geometric_prompt(
                box=[cx, cy, bw, bh],
                label=True,
                state=inference_state,
            )

        # Add each point as a tiny box prompt
        # Positive points (label=1) → True, negative (label=0) → False
        output = None
        for pt in points:
            px = pt["x"] / img_w
            py = pt["y"] / img_h
            # Use a small box around the point (1% of image dimension)
            size = 0.01
            pt_box = [px, py, size, size]
            is_positive = pt.get("label", 1) == 1
            output = processor.add_geometric_prompt(
                box=pt_box,
                label=is_positive,
                state=inference_state,
            )

        if output is None:
            return jsonify({"masks": [], "count": 0})

        masks = output["masks"]
        out_boxes = output.get("boxes", [])
        scores = output["scores"]

    else:
        return jsonify({"error": "Invalid prompt combination"}), 400

    # ── Normalize to numpy and build response ──
    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu()
    if torch.is_tensor(out_boxes):
        out_boxes = out_boxes.detach().cpu()

    results = []
    for i in range(len(scores)):
        mask_i = masks[i]
        if torch.is_tensor(mask_i):
            mask_i = mask_i.squeeze().numpy()
        else:
            mask_i = np.array(mask_i).squeeze()

        if mask_i.dtype != np.bool_:
            mask_i = mask_i > 0.5

        score_i = float(scores[i]) if torch.is_tensor(scores) else float(scores[i])
        box_i = None
        if len(out_boxes) > i:
            b = out_boxes[i]
            if torch.is_tensor(b):
                b = b.numpy()
            box_i = [round(float(x), 2) for x in (b if isinstance(b, list) else b.tolist())]

        results.append({
            "mask_b64": mask_to_base64_png(mask_i),
            "score": round(score_i, 4),
            "box": box_i,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"masks": results, "count": len(results)})


# ---------------------------------------------------------------------------
# Mask eraser (server-side pixel removal)
# ---------------------------------------------------------------------------

@app.route("/api/erase_mask", methods=["POST"])
@login_required
def erase_mask():
    """
    Erase pixels from a mask.
    Expects JSON: {
        mask_b64: base64 grayscale PNG,
        strokes: [{ x, y, radius }]  — pixel coords to erase
    }
    Returns: { mask_b64: updated mask }
    """
    data = request.get_json()
    mask_b64 = data["mask_b64"]
    strokes = data.get("strokes", [])

    mask_img = base64_to_pil(mask_b64).convert("L")
    mask_np = np.array(mask_img)

    h, w = mask_np.shape
    for s in strokes:
        cx, cy, r = int(s["x"]), int(s["y"]), int(s["radius"])
        # Create circular erase region
        yy, xx = np.ogrid[:h, :w]
        dist = (xx - cx) ** 2 + (yy - cy) ** 2
        mask_np[dist <= r * r] = 0

    # Convert back
    result_b64 = mask_to_base64_png(mask_np > 128)
    return jsonify({"mask_b64": result_b64})


# ---------------------------------------------------------------------------
# SAM3 direct text prompt (shortcut — skips DINO)
# ---------------------------------------------------------------------------

@app.route("/api/sam3_text", methods=["POST"])
@login_required
def run_sam3_text():
    """
    Convenience endpoint: text prompt → SAM3 directly.
    Expects JSON: { frame_path, prompt }
    """
    data = request.get_json()
    data["text_prompt"] = data.pop("prompt", data.get("text_prompt"))
    # Reuse the sam3 handler logic
    with app.test_request_context(
        "/api/sam3", method="POST",
        json=data, content_type="application/json"
    ):
        return run_sam3()


# ---------------------------------------------------------------------------
# Save annotation
# ---------------------------------------------------------------------------

@app.route("/api/save", methods=["POST"])
@login_required
def save_annotation():
    """
    Save an approved annotation.
    Expects JSON: {
        experiment, frame_id, frame_path,
        box, mask_b64, prompt, label,
        video_name, timestamp, score
    }
    Saves: binary mask PNG + metadata JSON into the experiment folder.
    """
    data = request.get_json()
    experiment = data["experiment"]
    frame_id = data.get("frame_id", f"unknown_{uuid.uuid4().hex[:6]}")

    exp_dir = ensure_experiment_dirs(experiment)
    annotation_id = f"ann_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Save binary mask
    mask_b64 = data["mask_b64"]
    mask_path = exp_dir / "masks" / f"{annotation_id}_mask.png"
    mask_img = base64_to_pil(mask_b64)
    mask_img.save(str(mask_path))

    # Save metadata
    meta = {
        "annotation_id": annotation_id,
        "frame_id": frame_id,
        "frame_path": data.get("frame_path", ""),
        "box": data.get("box"),
        "prompt": data.get("prompt", ""),
        "label": data.get("label", ""),
        "video_name": data.get("video_name", ""),
        "timestamp": data.get("timestamp", 0),
        "score": data.get("score", 0),
        "mask_path": str(mask_path),
        "created_at": datetime.now().isoformat(),
    }
    meta_path = exp_dir / "annotations" / f"{annotation_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return jsonify({
        "annotation_id": annotation_id,
        "mask_path": str(mask_path),
        "meta_path": str(meta_path),
        "message": "Annotation saved",
    })


# ---------------------------------------------------------------------------
# List annotations for an experiment
# ---------------------------------------------------------------------------

@app.route("/api/annotations/<experiment>", methods=["GET"])
@login_required
def list_annotations(experiment):
    """List all saved annotations for an experiment."""
    exp_dir = EXPERIMENTS_DIR / experiment / "annotations"
    if not exp_dir.exists():
        return jsonify({"annotations": []})
    annotations = []
    for f in sorted(exp_dir.glob("*.json")):
        with open(f) as fh:
            annotations.append(json.load(fh))
    return jsonify({"annotations": annotations, "count": len(annotations)})

@app.route("/api/categories/<experiment>", methods=["GET"])
@login_required
def get_categories(experiment):
    """Return the category list for an experiment."""
    cat_path = EXPERIMENTS_DIR / experiment / "categories.json"
    if not cat_path.exists():
        return jsonify({"categories": []})
    with open(cat_path) as f:
        data = json.load(f)
    return jsonify(data)
 
 
@app.route("/api/categories/<experiment>", methods=["POST"])
@login_required
def save_categories(experiment):
    """
    Save (overwrite) the category list for an experiment.
    Expects JSON: { categories: [ { id, name, color }, ... ] }
    """
    data = request.get_json()
    cats = data.get("categories", [])
 
    # Basic validation
    for cat in cats:
        if not isinstance(cat.get("id"), str) or not isinstance(cat.get("name"), str):
            return jsonify({"error": "Invalid category format"}), 400
 
    exp_dir = ensure_experiment_dirs(experiment)
    cat_path = exp_dir / "categories.json"
    with open(cat_path, "w") as f:
        json.dump({"categories": cats}, f, indent=2)
 
    return jsonify({"ok": True, "count": len(cats)})
 

# ---------------------------------------------------------------------------
# Serve saved frames/masks by path
# ---------------------------------------------------------------------------

@app.route("/api/file", methods=["GET"])
@login_required
def serve_file():
    """Serve a file from the experiments directory. ?path=relative/path"""
    rel_path = request.args.get("path", "")
    full_path = EXPERIMENTS_DIR / rel_path
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(full_path))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(cache.device),
        "dino_loaded": cache._dino_model is not None,
        "sam3_loaded": cache._sam3_model is not None,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 + DINO Annotation Server")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind address. Use 0.0.0.0 to allow external access.")
    parser.add_argument("--experiments-dir", type=str, default="./experiments")
    parser.add_argument("--password", type=str, default=None,
                        help="Login password (default: 'annotate')")
    parser.add_argument("--browse-root", type=str, default=None,
                        help="Base path for file browser (default: /sc/arion/projects/video_rarp)")
    args = parser.parse_args()

    global EXPERIMENTS_DIR, APP_PASSWORD, BROWSE_ROOT
    EXPERIMENTS_DIR = Path(args.experiments_dir)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.password:
        APP_PASSWORD = args.password
    if args.browse_root:
        BROWSE_ROOT = Path(args.browse_root)

    print(f"\n{'='*60}")
    print(f"  Annotation Server")
    print(f"  URL:  http://{args.host}:{args.port}")
    print(f"  Password: {APP_PASSWORD}")
    print(f"  Browse root: {BROWSE_ROOT}")
    print(f"  Experiments: {EXPERIMENTS_DIR.resolve()}")
    print(f"  Device: {cache.device}")
    print(f"  Models: pre-loaded")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()