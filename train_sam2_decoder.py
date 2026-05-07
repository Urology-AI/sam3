#!/usr/bin/env python3
"""
train_sam2_decoder.py
=====================
Fine-tune the SAM2 mask decoder to be robust to noisy bounding boxes
for prostate segmentation.

Architecture
------------
  FROZEN : SAM2 image encoder (ViT-L)
  FROZEN : SAM2 prompt encoder
  TRAINED: SAM2 mask decoder (TwoWayTransformer + upscaling + output heads)

The decoder is trained with heavily perturbed GT boxes — from tight ±10%
all the way to the full image — so that at inference the imperfect boxes
produced by the SAM3 detector still yield clean masks.

Box noise regimes (sampled per-sample each epoch):
  15% → tight    scale ~ U(0.00, 0.10)
  50% → medium   scale ~ U(0.10, 0.50)
  25% → large    scale ~ U(0.50, 1.00)
  10% → extreme  box = entire image

Loss: Dice + BCE on full-resolution (1024×1024) mask logits.

Usage
-----
  cd /sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3
  python3 train_sam2_decoder.py [--epochs 30] [--batch_size 4] [--lr 1e-4]
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM3_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = "/sc/arion/projects/video_rarp/neel_projects/autosam-instruments-GraSP-trained/sam2"
sys.path.insert(0, SAM3_DIR)
sys.path.insert(0, SAM2_DIR)

DATASET_DIR = (
    "/sc/arion/projects/video_rarp/neel_projects/"
    "autosam-instruments-GraSP-trained/sam2/projects/prostate_tracker_long_qpr_19"
)
SAM2_CKPT   = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG    = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUT_DIR  = os.path.join(SAM3_DIR, "sam2_decoder_training")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
TB_DIR   = os.path.join(OUT_DIR, "tensorboard")

SAM2_IMG_SIZE = 1024   # SAM2 large input resolution

# ── Dice + BCE loss ────────────────────────────────────────────────────────────

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """pred_logits: (B,1,H,W) raw logits; target: (B,1,H,W) float {0,1}."""
    pred  = torch.sigmoid(pred_logits)
    inter = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def seg_loss(pred_logits, target):
    return (
        dice_loss(pred_logits, target)
        + F.binary_cross_entropy_with_logits(pred_logits, target)
    )


# ── Box utilities ─────────────────────────────────────────────────────────────

def mask_to_box(mask_np: np.ndarray, img_size: int) -> np.ndarray:
    """Return tight xyxy box in [0, img_size] coords from a uint8 mask (original size)."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return np.array([0, 0, img_size, img_size], dtype=np.float32)
    orig_h, orig_w = mask_np.shape[:2]
    sx = img_size / orig_w
    sy = img_size / orig_h
    return np.array([
        xs.min() * sx, ys.min() * sy,
        xs.max() * sx, ys.max() * sy,
    ], dtype=np.float32)


def perturb_box(box: np.ndarray, img_size: int = SAM2_IMG_SIZE) -> np.ndarray:
    """Apply regime-sampled noise to an xyxy box, staying within [0, img_size]."""
    r = random.random()
    if r < 0.15:
        scale = random.uniform(0.00, 0.10)
    elif r < 0.65:
        scale = random.uniform(0.10, 0.50)
    elif r < 0.90:
        scale = random.uniform(0.50, 1.00)
    else:
        return np.array([0.0, 0.0, img_size, img_size], dtype=np.float32)

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 += random.uniform(-scale * bw, scale * bw)
    y1 += random.uniform(-scale * bh, scale * bh)
    x2 += random.uniform(-scale * bw, scale * bw)
    y2 += random.uniform(-scale * bh, scale * bh)
    x1 = float(np.clip(x1, 0, img_size))
    y1 = float(np.clip(y1, 0, img_size))
    x2 = float(np.clip(x2, 0, img_size))
    y2 = float(np.clip(y2, 0, img_size))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < 2: x2 = min(x1 + 2, float(img_size))
    if y2 - y1 < 2: y2 = min(y1 + 2, float(img_size))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ProstateBoxDataset(Dataset):
    def __init__(self, stems: list[str], augment: bool = True):
        self.stems   = stems
        self.augment = augment
        img_dir  = Path(DATASET_DIR) / "images"
        mask_dir = Path(DATASET_DIR) / "masks"
        self.img_paths  = [img_dir  / f"{s}.jpg" for s in stems]
        self.mask_paths = [mask_dir / f"{s}.png" for s in stems]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(str(self.img_paths[idx])),  cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        # Tight GT box in original coords, before resize
        gt_box = mask_to_box(mask, SAM2_IMG_SIZE)

        # Resize both to 1024×1024
        img  = cv2.resize(img,  (SAM2_IMG_SIZE, SAM2_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SAM2_IMG_SIZE, SAM2_IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Normalise image
        img = img.astype(np.float32) / 255.0
        img = (img - _IMG_MEAN) / _IMG_STD
        img_t = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)

        # Binary mask float tensor
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # (1, H, W)

        # Perturb box (train) or use tight box (val)
        box = perturb_box(gt_box) if self.augment else gt_box
        box_t = torch.from_numpy(box)   # (4,) xyxy in [0, 1024]

        return img_t, mask_t, box_t


# ── SAM2 forward (image encode once, decode per box) ─────────────────────────

_BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]   # SAM2 large, 1024 input


def encode_images(model, imgs: torch.Tensor):
    """
    Frozen image encoding. Returns (image_embed, high_res_feats).
      image_embed    : (B, 256, 64, 64)
      high_res_feats : [(B, C, 256, 256), (B, C, 128, 128)]
    """
    backbone_out = model.forward_image(imgs)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    B = imgs.shape[0]
    feats = [
        feat.permute(1, 2, 0).view(B, -1, *fs)
        for feat, fs in zip(vision_feats[::-1], _BB_FEAT_SIZES[::-1])
    ][::-1]
    return feats[-1], feats[:-1]


def decode_masks(model, image_embed, high_res_feats, boxes: torch.Tensor):
    """
    boxes: (B, 4) xyxy absolute pixel coords in [0, 1024].
    Returns high_res_masks (B, 1, 1024, 1024) as logits.
    """
    B = boxes.shape[0]
    device = boxes.device

    # Encode box as point-pair (labels 2=top-left, 3=bottom-right)
    box_coords  = boxes.reshape(B, 2, 2)                              # (B, 2, 2)
    box_labels  = torch.tensor([[2, 3]], dtype=torch.int, device=device).expand(B, -1)

    with torch.no_grad():
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(box_coords, box_labels),
            boxes=None,
            masks=None,
        )

    low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    # low_res_masks: (B, 1, 256, 256) → upsample to 1024
    high_res_masks = F.interpolate(
        low_res_masks, size=(SAM2_IMG_SIZE, SAM2_IMG_SIZE),
        mode="bilinear", align_corners=False,
    )
    return high_res_masks, iou_preds


# ── IoU metric ────────────────────────────────────────────────────────────────

def batch_iou(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred_logits.sigmoid() > 0.5).float()
    inter = (pred * target).sum(dim=(-3, -2, -1))
    union = (pred + target).clamp(max=1).sum(dim=(-3, -2, -1))
    return (inter / (union + 1e-6)).mean().item()


# ── Training / validation loops ───────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool):
    model.sam_mask_decoder.train(train)
    total_loss = total_iou = n = 0

    for imgs, masks, boxes in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        boxes = boxes.to(device)

        with torch.no_grad():
            image_embed, high_res_feats = encode_images(model, imgs)

        pred_masks, _ = decode_masks(model, image_embed, high_res_feats, boxes)
        loss = seg_loss(pred_masks, masks)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            iou = batch_iou(pred_masks, masks)
        total_loss += loss.item() * imgs.shape[0]
        total_iou  += iou        * imgs.shape[0]
        n          += imgs.shape[0]

    return total_loss / n, total_iou / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    run_id   = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints", run_id)
    run_dir  = os.path.join(RUNS_DIR, run_id)
    for d in [ckpt_dir, run_dir, TB_DIR, os.path.join(OUT_DIR, "mlruns")]:
        os.makedirs(d, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data split ────────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stems = sorted(p.stem for p in (Path(DATASET_DIR) / "images").glob("*.jpg"))
    random.shuffle(stems)
    n_val    = max(1, int(len(stems) * args.val_split))
    val_stems, train_stems = stems[:n_val], stems[n_val:]
    print(f"Dataset: {len(train_stems)} train / {len(val_stems)} val")

    train_ds = ProstateBoxDataset(train_stems, augment=True)
    val_ds   = ProstateBoxDataset(val_stems,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ── Load SAM2 ─────────────────────────────────────────────────────────────
    print("\nLoading SAM2 large...")
    from sam2.build_sam import build_sam2
    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    model.eval()

    # Freeze everything except the mask decoder
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.sam_mask_decoder.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── MLflow + TensorBoard ──────────────────────────────────────────────────
    tb = SummaryWriter(log_dir=os.path.join(TB_DIR, run_id))
    mlflow.set_tracking_uri(os.path.join(OUT_DIR, "mlruns"))
    mlflow.set_experiment("sam2_decoder_noisy_box")
    mlflow.start_run(run_name=run_id)
    mlflow.log_params({
        **vars(args),
        "n_train":          len(train_ds),
        "n_val":            len(val_ds),
        "n_trainable_M":    round(n_trainable / 1e6, 1),
        "n_total_M":        round(n_total     / 1e6, 1),
        "sam2_ckpt":        os.path.basename(SAM2_CKPT),
        "noise_regimes":    "tight0.15/medium0.50/large0.25/extreme0.10",
    })
    print(f"MLflow run: {run_id}")

    log_path = os.path.join(OUT_DIR, "train_run.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n{'='*60}")
    log(f"Run {run_id}  |  {args.epochs} epochs  batch={args.batch_size}  lr={args.lr}")
    log(f"{'='*60}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_iou = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_iou = run_epoch(model, train_dl, optimizer, device, train=True)
        scheduler.step()

        with torch.no_grad():
            val_loss, val_iou = run_epoch(model, val_dl, optimizer, device, train=False)

        elapsed = time.time() - t0
        log(f"Epoch {epoch} done in {elapsed:.0f}s | "
            f"loss={train_loss:.4f}  iou={train_iou:.3f}")
        log(f"  Val — loss={val_loss:.4f}  iou={val_iou:.3f}")

        lr_now = optimizer.param_groups[0]["lr"]
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("train/iou",  train_iou,  epoch)
        tb.add_scalar("val/loss",   val_loss,   epoch)
        tb.add_scalar("val/iou",    val_iou,    epoch)
        tb.add_scalar("lr",         lr_now,     epoch)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_iou":  train_iou,
            "val_loss":   val_loss,
            "val_iou":    val_iou,
            "lr":         lr_now,
        }, step=epoch)

        # Always save latest
        ckpt = {
            "epoch":      epoch,
            "val_iou":    val_iou,
            "state_dict": model.sam_mask_decoder.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_latest.pth"))

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best.pth"))
            log(f"  -> Saved best checkpoint (val_iou={val_iou:.3f})")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    log(f"\nTraining complete. Best val IoU: {best_val_iou:.3f}")
    log(f"Best checkpoint: {ckpt_dir}/checkpoint_best.pth")
    log(f"Run logged to  : {log_path}")

    mlflow.log_metrics({"best_val_iou": best_val_iou})
    mlflow.log_artifact(config_path)
    mlflow.end_run()

    # ── runs.log summary ──────────────────────────────────────────────────────
    runs_log = os.path.join(OUT_DIR, "runs.log")
    with open(runs_log, "a") as f:
        f.write(f"{run_id}  epochs={args.epochs}  batch={args.batch_size}  "
                f"lr={args.lr}  best_val_iou={best_val_iou:.3f}  "
                f"ckpt={ckpt_dir}/checkpoint_best.pth\n")

    tb.close()
    log_file.close()


if __name__ == "__main__":
    main()
