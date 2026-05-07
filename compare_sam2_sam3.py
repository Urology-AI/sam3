#!/usr/bin/env python3
"""
compare_sam2_sam3.py

Side-by-side comparison video:
  Left  → SAM2 per-frame (AutoSamSeg, no temporal tracking)
  Right → SAM3 chunked reinit tracking (pre-computed masks reused)

Both overlays at alpha=0.15.
SAM3 right panel shows the reinit detection bounding box for 1 second
after each chunk initialisation.

Input:
  - Pre-clipped video from the SAM3 run output dir
  - Pre-computed SAM3 .npy masks (from detect_and_segment_chunked.py)
Output:
  - comparison.mp4 (1920x540, left=SAM2, right=SAM3)
"""

import os, sys, time, subprocess
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

SAM3_CKPT    = ("/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
                "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")
BPE_PATH     = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
SAM2_BASE    = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_base_plus.pt")
MODEL_CFG    = "configs/sam2.1/sam2.1_hiera_b+.yaml"
AUTOSAM_CKPT = os.path.join(SAM2_DIR, "output_experiment/"
               "prostate_tracker_20260427_1049_343imgs/checkpoints/model_best_dice.pth")
DET_CKPT     = os.path.join(SAM3_DIR,
               "detector_training/checkpoints/20260428_0942/checkpoint_best.pth")

VIDEO_CLIP = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/"
              "M_04222026081653_U013419042221253_2_002_0002-01_1290-1380.mp4")
MASKS_DIR  = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "M_04222026081653_U013419042221253_2_002_0002-01_sam3_chunked/masks")
OUTPUT_DIR = ("/sc/arion/projects/video_rarp/neel_projects/chahat_videos/TITLE 002/"
              "comparison_sam2_vs_sam3")

REINIT_INTERVAL   = 150       # must match the chunked run that produced the masks
MASK_ALPHA        = 0.15
MASK_COLOR_BGR    = (0, 255, 0)   # green
BOX_COLOR_BGR     = (0, 0, 255)   # red
TEXT_QUERY        = "prostate gland"
DETECTOR_IMG_SIZE = 1008
SAM2_BATCH_SIZE   = 8
OUT_W_EACH        = 960           # per-panel output width
OUT_H_EACH        = 540           # per-panel output height

_DET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_DET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── SAM2 (AutoSamSeg) ─────────────────────────────────────────────────────────

def build_sam2_model(device):
    from autosam_utils import AutoSamSeg, MaskDecoder, TwoWayTransformer
    from sam2.build_sam import build_sam2

    sam2  = build_sam2(MODEL_CFG, SAM2_BASE, device=device)
    enc   = sam2.image_encoder.to(device)
    model = AutoSamSeg(
        image_encoder=enc,
        seg_decoder=MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=2,
        ),
    ).to(device)
    ckpt  = torch.load(AUTOSAM_CKPT, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    print(f"  SAM2 model loaded  epoch={ckpt.get('epoch', '?')}")
    return model


def preprocess_bgr(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return IMG_TRANSFORM(pil)


def sam2_infer_batch(model, frames_bgr, device):
    """Returns list of uint8 (1024,1024) masks: 0=BG, 1=prostate."""
    tensors = torch.stack([preprocess_bgr(f) for f in frames_bgr]).float().to(device)
    with torch.no_grad():
        logits, _ = model(tensors)
        b      = tensors.shape[0]
        logits = logits.view(b, -1, 1024, 1024)
        prob   = F.softmax(logits, dim=1)
        preds  = torch.argmax(prob, dim=1).cpu().numpy().astype(np.uint8)
    return list(preds)


# ── SAM3 detector (box coords only) ──────────────────────────────────────────

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
    print(f"  SAM3 detector loaded  epoch={raw.get('epoch', '?')}")
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
        gp  = Prompt(
            box_embeddings = torch.zeros(0, B, 4, device=device),
            box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
        )
        prompt, pm, bb = det._encode_prompt(bb, fi, gp)
        bb, eo, _      = det._run_encoder(bb, fi, prompt, pm)
        out = {"encoder_hidden_states": eo["encoder_hidden_states"]}
        out, _ = det._run_decoder(
            memory      = out["encoder_hidden_states"],
            pos_embed   = eo["pos_embed"],
            src_mask    = eo["padding_mask"],
            out         = out,
            prompt      = prompt,
            prompt_mask = pm,
            encoder_out = eo,
        )
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best_q = scores.argmax().item()
    conf   = scores[best_q].item()
    cx, cy, w, h = out["pred_boxes"][0][best_q].cpu().tolist()
    box_abs = np.array(
        [(cx - w/2) * w_vid, (cy - h/2) * h_vid,
         (cx + w/2) * w_vid, (cy + h/2) * h_vid],
        dtype=np.float32,
    )
    return box_abs, conf


# ── Rendering helpers ─────────────────────────────────────────────────────────

def apply_mask(frame, mask_hw, color_bgr, alpha):
    if not mask_hw.any():
        return frame
    overlay                      = frame.copy()
    overlay[mask_hw.astype(bool)] = color_bgr
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def put_label(frame, text, pos=(16, 38)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")

    # Video metadata
    cap      = cv2.VideoCapture(VIDEO_CLIP)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    box_dur = round(fps)    # frames the box stays on screen = 1 second
    print(f"Clip: {n_frames} frames  {w_vid}x{h_vid} @ {fps:.1f} fps")
    print(f"Box duration: {box_dur} frames (1 s)")
    print(f"Output: {OUT_W_EACH*2}x{OUT_H_EACH} (960+960 side-by-side)")

    # ── [1] Run SAM3 detector on every reinit frame to recover box coords ────
    print("\n[1] SAM3 detector → box coords at reinit frames...")
    t0  = time.time()
    det, lang_feats, lang_mask = load_detector(device)
    reinit_boxes = {}   # frame_idx -> (box_abs_xyxy float32, conf float)
    cap = cv2.VideoCapture(VIDEO_CLIP)
    for fidx in range(0, n_frames, REINIT_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if not ret:
            continue
        box, conf = detect_box(det, lang_feats, lang_mask, f, w_vid, h_vid, device)
        reinit_boxes[fidx] = (box, conf)
        print(f"  frame {fidx:5d}: conf={conf:.3f}  box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
    cap.release()
    del det, lang_feats, lang_mask
    torch.cuda.empty_cache()
    print(f"  Done in {time.time()-t0:.1f}s  ({len(reinit_boxes)} boxes)")

    # ── [2] Load SAM2 model ──────────────────────────────────────────────────
    print("\n[2] Loading SAM2 (AutoSam) model...")
    sam2_model = build_sam2_model(device)

    # ── [3] Render (batch-process SAM2, stream-read frames) ─────────────────
    print(f"\n[3] Rendering comparison video...")
    out_raw  = os.path.join(OUTPUT_DIR, "_comparison_raw.mp4")
    out_path = os.path.join(OUTPUT_DIR, "comparison.mp4")
    writer   = cv2.VideoWriter(
        out_raw,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUT_W_EACH * 2, OUT_H_EACH),
    )

    cap       = cv2.VideoCapture(VIDEO_CLIP)
    fidx      = 0
    batch_buf = []      # list of BGR frames
    t0        = time.time()

    def flush(batch):
        nonlocal fidx
        if not batch:
            return
        sam2_masks = sam2_infer_batch(sam2_model, batch, device)
        for frame_bgr, sam2_mask in zip(batch, sam2_masks):
            # ── Left panel: SAM2 ──────────────────────────────────────────
            sam2_mask_r = cv2.resize(sam2_mask, (w_vid, h_vid),
                                     interpolation=cv2.INTER_NEAREST)
            left = apply_mask(frame_bgr.copy(), sam2_mask_r, MASK_COLOR_BGR, MASK_ALPHA)
            put_label(left, "SAM2  (AutoSam per-frame)")

            # ── Right panel: SAM3 ─────────────────────────────────────────
            right    = frame_bgr.copy()
            npy_path = os.path.join(MASKS_DIR, f"{fidx:06d}.npy")
            if os.path.exists(npy_path):
                m = np.load(npy_path)
                if m.shape[:2] != (h_vid, w_vid):
                    m = cv2.resize(m, (w_vid, h_vid), interpolation=cv2.INTER_NEAREST)
                right = apply_mask(right, m, MASK_COLOR_BGR, MASK_ALPHA)

            # Bounding box visible for 1 s after each reinit
            for r_start, (box_abs, conf) in reinit_boxes.items():
                if r_start <= fidx < r_start + box_dur:
                    x1, y1, x2, y2 = box_abs.astype(int)
                    cv2.rectangle(right, (x1, y1), (x2, y2), BOX_COLOR_BGR, 3)
                    label = f"reinit  conf={conf:.2f}"
                    cv2.putText(right, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(right, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                BOX_COLOR_BGR, 2, cv2.LINE_AA)
                    break
            put_label(right, "SAM3  (Chunked Reinit)")

            # ── Scale and combine ────────────────────────────────────────
            l_s      = cv2.resize(left,  (OUT_W_EACH, OUT_H_EACH))
            r_s      = cv2.resize(right, (OUT_W_EACH, OUT_H_EACH))
            combined = np.concatenate([l_s, r_s], axis=1)
            # Divider line
            cv2.line(combined, (OUT_W_EACH, 0), (OUT_W_EACH, OUT_H_EACH),
                     (180, 180, 180), 2)
            writer.write(combined)

            fidx += 1
            if fidx % 300 == 0:
                elapsed = time.time() - t0
                print(f"  frame {fidx}/{n_frames}  "
                      f"({fidx / max(elapsed, 1e-3):.0f} fps)", flush=True)

        batch.clear()

    while True:
        ret, f = cap.read()
        if not ret:
            break
        batch_buf.append(f)
        if len(batch_buf) >= SAM2_BATCH_SIZE:
            flush(batch_buf)
    flush(batch_buf)     # tail

    cap.release()
    writer.release()
    del sam2_model
    torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"  Render done: {fidx} frames in {elapsed:.1f}s  "
          f"({fidx/max(elapsed,1e-3):.0f} fps)")

    # ── [4] Re-mux to H.264 ─────────────────────────────────────────────────
    print("\n[4] Re-muxing to H.264...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", out_raw,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg warning:", result.stderr[:400])
    os.remove(out_raw)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Output: {out_path}")
    print(f"  Frames: {fidx}")
    print(f"  Size:   {OUT_W_EACH*2}x{OUT_H_EACH}  (960+960)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
