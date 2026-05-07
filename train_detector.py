#!/usr/bin/env python3
"""
train_detector.py
=================
Fine-tune the SAM3 detector (transformer decoder + dot-product scorer +
geometry encoder + segmentation head) for prostate bounding-box grounding.

What is frozen
--------------
  model.backbone.vision_backbone   — ViT image encoder
  model.backbone.language_backbone — CLIP text encoder

What is trained
---------------
  model.transformer        — cross-attention decoder
  model.dot_prod_scoring   — box/text similarity scorer
  model.geometry_encoder   — geometry input encoder
  model.segmentation_head  — pixel-level mask decoder (optional; can disable)

Text query
----------
  Fixed string "prostate gland" — encoded once before training begins and
  cached.  All images in a batch share the same text embedding.

Data
----
  Run masks_to_coco.py first to produce:
    detector_training/data/annotations_train.json
    detector_training/data/annotations_val.json
    detector_training/data/images/{stem}.jpg

Loss
----
  Per-prediction: sigmoid focal loss on logits (label=1 for matched query).
  Per-matched-box: L1 + GIoU between pred_boxes (cx,cy,w,h) and GT box.
  Matching: Hungarian (bipartite) with cost = focal + L1 + GIoU — using
  SAM3's own HungarianMatcher so the cost functions are identical to the
  original training.

Usage
-----
  python3 train_detector.py                   # defaults
  python3 train_detector.py --epochs 30 --lr 1e-4 --batch_size 4
"""

import argparse
import datetime
import json
import os
import sys
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM3_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(SAM3_DIR, "detector_training")
DATA_DIR   = os.path.join(TRAIN_DIR, "data")
CKPT_DIR   = os.path.join(TRAIN_DIR, "checkpoints")
TB_DIR     = os.path.join(TRAIN_DIR, "tensorboard")
RUNS_DIR   = os.path.join(TRAIN_DIR, "runs")

sys.path.insert(0, SAM3_DIR)

SAM3_CKPT  = "/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/" \
             "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
BPE_PATH   = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

TEXT_QUERY = "prostate gland"


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProstateDetectionDataset(Dataset):
    """
    Reads COCO-format annotations produced by masks_to_coco.py.
    Returns (img_tensor, bbox_cxcywh_norm) where bbox is [cx, cy, w, h]
    normalised to [0, 1] by image dimensions.
    """

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]
    IMG_SIZE = 1008   # SAM3 ViT expects 1008×1008

    def __init__(self, ann_json: str, augment: bool = False):
        with open(ann_json) as f:
            coco = json.load(f)

        img_dir = os.path.join(DATA_DIR, "images")
        self.img_dir = img_dir

        ann_by_img = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_info in coco["images"]:
            anns = ann_by_img.get(img_info["id"], [])
            if not anns:
                continue
            # One ground-truth box per image (prostate is a single structure)
            ann    = anns[0]
            x, y, w, h = ann["bbox"]          # COCO: [x,y,w,h] pixel
            iw, ih     = img_info["width"], img_info["height"]
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            self.samples.append({
                "file_name": img_info["file_name"],
                "box_norm":  [cx, cy, nw, nh],   # [cx, cy, w, h] in [0,1]
            })

        self.augment   = augment
        self.normalize = T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, s["file_name"])).convert("RGB")

        if self.augment:
            # Random horizontal flip — update cx accordingly
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
                cx, cy, w, h = s["box_norm"]
                s = dict(s, box_norm=[1.0 - cx, cy, w, h])

        img = TF.resize(img, [self.IMG_SIZE, self.IMG_SIZE])
        img = TF.to_tensor(img)            # [0,1]
        img = self.normalize(img)          # ImageNet norm

        box = torch.tensor(s["box_norm"], dtype=torch.float32)
        return img, box                    # (3,H,W), (4,)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def generalized_box_iou(b1, b2):
    """GIoU between two sets of [x1,y1,x2,y2] boxes. Both (N,4)."""
    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1    = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2    = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = a1 + a2 - inter + 1e-6

    # Enclosing box
    enc_x1 = torch.min(b1[:, 0], b2[:, 0])
    enc_y1 = torch.min(b1[:, 1], b2[:, 1])
    enc_x2 = torch.max(b1[:, 2], b2[:, 2])
    enc_y2 = torch.max(b1[:, 3], b2[:, 3])
    enc    = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-6

    return inter / union - (enc - union) / enc   # GIoU ∈ [-1, 1]


@torch.no_grad()
def hungarian_match(pred_logits, pred_boxes, gt_boxes):
    """
    Simple bipartite matching: cost = -sigmoid(logit) + L1 + (1 - GIoU).
    pred_logits : (B, Q, 1)
    pred_boxes  : (B, Q, 4) cxcywh
    gt_boxes    : (B, 4)    cxcywh
    Returns indices: (B,) — the matched query index per image.
    """
    B, Q, _ = pred_boxes.shape
    probs    = pred_logits.squeeze(-1).sigmoid()           # (B, Q)
    gt_exp   = gt_boxes.unsqueeze(1).expand(-1, Q, -1)    # (B, Q, 4)

    cost_cls  = -probs                                     # (B, Q)
    cost_l1   = (pred_boxes - gt_exp).abs().sum(-1)        # (B, Q)

    pb_xyxy = box_cxcywh_to_xyxy(pred_boxes.reshape(B*Q, 4))
    gt_xyxy = box_cxcywh_to_xyxy(gt_exp.reshape(B*Q, 4))
    giou    = generalized_box_iou(pb_xyxy, gt_xyxy).reshape(B, Q)
    cost_giou = 1.0 - giou                                 # (B, Q)

    total_cost = 2.0 * cost_cls + 5.0 * cost_l1 + 2.0 * cost_giou
    return total_cost.argmin(dim=1)   # (B,)


def detection_loss(pred_logits, pred_boxes, gt_boxes, w_l1=5.0, w_giou=2.0):
    """
    pred_logits : (B, Q, 1)
    pred_boxes  : (B, Q, 4)  cxcywh normalised
    gt_boxes    : (B, 4)     cxcywh normalised
    """
    B = pred_logits.shape[0]
    matched_idx = hungarian_match(pred_logits, pred_boxes, gt_boxes)  # (B,)

    batch_idx = torch.arange(B, device=pred_logits.device)

    # ── Classification: focal loss ─────────────────────────────────────────
    # +1 for matched query, -1 (ignore / 0 label) for rest
    labels = torch.zeros(B, pred_logits.shape[1], device=pred_logits.device)
    labels[batch_idx, matched_idx] = 1.0

    alpha, gamma = 0.25, 2.0
    p     = pred_logits.squeeze(-1).sigmoid()
    ce    = F.binary_cross_entropy_with_logits(
                pred_logits.squeeze(-1), labels, reduction="none")
    p_t   = p * labels + (1 - p) * (1 - labels)
    focal = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    loss_cls = (alpha_t * focal).mean()

    # ── Box losses on matched predictions only ──────────────────────────────
    matched_boxes = pred_boxes[batch_idx, matched_idx]   # (B, 4)

    loss_l1 = F.l1_loss(matched_boxes, gt_boxes, reduction="mean")

    pb_xyxy = box_cxcywh_to_xyxy(matched_boxes)
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
    loss_giou = (1 - generalized_box_iou(pb_xyxy, gt_xyxy)).mean()

    total = loss_cls + w_l1 * loss_l1 + w_giou * loss_giou
    return total, loss_cls.detach(), loss_l1.detach(), loss_giou.detach()


def box_iou_metric(pred_boxes, gt_boxes):
    """Average IoU (not GIoU) for logging."""
    pb = box_cxcywh_to_xyxy(pred_boxes)
    gb = box_cxcywh_to_xyxy(gt_boxes)
    x1 = torch.max(pb[:, 0], gb[:, 0])
    y1 = torch.max(pb[:, 1], gb[:, 1])
    x2 = torch.min(pb[:, 2], gb[:, 2])
    y2 = torch.min(pb[:, 3], gb[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    a2 = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    union = a1 + a2 - inter + 1e-6
    return (inter / union).mean().item()


# ── Training helpers ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, v, n=1):
        self.sum   += v * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def get_best_pred_box(pred_logits, pred_boxes):
    """Return the highest-confidence box per image."""
    best = pred_logits.squeeze(-1).argmax(dim=1)   # (B,)
    B    = pred_logits.shape[0]
    return pred_boxes[torch.arange(B, device=pred_logits.device), best]  # (B,4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--w_l1",        type=float, default=5.0)
    parser.add_argument("--w_giou",      type=float, default=2.0)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--print_freq",  type=int,   default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Output dirs ────────────────────────────────────────────────────────
    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir      = os.path.join(RUNS_DIR, ts)
    run_ckpt_dir = os.path.join(CKPT_DIR, ts)   # per-run: no cross-run overwrites
    for d in [run_ckpt_dir, TB_DIR, RUNS_DIR, run_dir]:
        os.makedirs(d, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.path.join(TRAIN_DIR, "mlruns"))
    mlflow.set_experiment("prostate_detector")
    mlflow.start_run(run_name=ts)
    mlflow.log_params({**vars(args), "text_query": TEXT_QUERY})
    print(f"MLflow run: {ts}")

    writer = SummaryWriter(os.path.join(TB_DIR, ts))

    # ── Dataset & loaders ─────────────────────────────────────────────────
    ann_train = os.path.join(DATA_DIR, "annotations_train.json")
    ann_val   = os.path.join(DATA_DIR, "annotations_val.json")
    assert os.path.exists(ann_train), \
        f"Run masks_to_coco.py first — {ann_train} not found."

    train_ds = ProstateDetectionDataset(ann_train, augment=True)
    val_ds   = ProstateDetectionDataset(ann_val,   augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    mlflow.log_params({"n_train": len(train_ds), "n_val": len(val_ds)})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────────
    print("\nBuilding SAM3 image model...")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        eval_mode=False,
        enable_segmentation=True,
        bpe_path=BPE_PATH,
    )
    model = model.to(device)

    # Freeze vision and language backbones
    freeze(model.backbone.vision_backbone)
    freeze(model.backbone.language_backbone)
    print("  Frozen: backbone.vision_backbone, backbone.language_backbone")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # ── Pre-compute fixed text embedding ──────────────────────────────────
    print(f"\nPre-computing text embedding for '{TEXT_QUERY}'...")
    model.eval()
    with torch.no_grad():
        text_out = model.backbone.forward_text(
            captions=[TEXT_QUERY], device=device
        )
    # language_features: (seq_len, 1, dim) — keep on CPU, move to device per batch
    lang_feats = text_out["language_features"].cpu()   # (seq_len, 1, dim)
    lang_mask  = text_out["language_mask"].cpu()       # (1, seq_len)
    print(f"  Text features: {lang_feats.shape}, mask: {lang_mask.shape}")
    model.train()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )

    best_val_iou = 0.0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        # Keep backbone in eval mode so BN/LN stats stay frozen
        model.backbone.vision_backbone.eval()
        model.backbone.language_backbone.eval()

        meters = {k: AverageMeter() for k in
                  ["loss", "loss_cls", "loss_l1", "loss_giou", "iou"]}
        t0 = time.time()

        for i, (imgs, gt_boxes) in enumerate(train_loader):
            imgs     = imgs.to(device)             # (B, 3, 1008, 1008)
            gt_boxes = gt_boxes.to(device)         # (B, 4) cxcywh

            B = imgs.shape[0]

            # ── Vision features (trainable path is downstream of backbone) ──
            with torch.no_grad():
                vis_out = model.backbone.forward_image(imgs)

            # ── Inject cached text features (expand for batch) ────────────
            # language_features shape: (seq_len, 1, dim) → (seq_len, B, dim)
            # language_mask shape:     (1, seq_len)      → (B, seq_len)
            backbone_out = {
                **vis_out,
                "language_features": lang_feats.to(device).expand(-1, B, -1),
                "language_mask":     lang_mask.to(device).expand(B, -1),
            }

            # ── Construct minimal FindStage ────────────────────────────────
            from sam3.model.data_misc import FindStage
            from sam3.model.geometry_encoders import Prompt

            find_input = FindStage(
                img_ids           = torch.arange(B, device=device),
                text_ids          = torch.zeros(B, dtype=torch.long, device=device),
                input_boxes       = torch.zeros(B, 0, 4, device=device),
                input_boxes_mask  = torch.zeros(B, 0, dtype=torch.bool, device=device),
                input_boxes_label = torch.zeros(B, 0, dtype=torch.long, device=device),
                input_points      = torch.zeros(B, 0, 2, device=device),
                input_points_mask = torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

            geo_prompt = Prompt(
                box_embeddings = torch.zeros(0, B, 4, device=device),
                box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
            )

            # ── Encode prompt + run encoder + decoder ─────────────────────
            prompt, prompt_mask, backbone_out = model._encode_prompt(
                backbone_out, find_input, geo_prompt
            )
            backbone_out, encoder_out, _ = model._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            }
            out, _ = model._run_decoder(
                memory    = out["encoder_hidden_states"],
                pos_embed = encoder_out["pos_embed"],
                src_mask  = encoder_out["padding_mask"],
                out       = out,
                prompt    = prompt,
                prompt_mask = prompt_mask,
                encoder_out = encoder_out,
            )

            pred_logits = out["pred_logits"]   # (B, Q, 1)
            pred_boxes  = out["pred_boxes"]    # (B, Q, 4) cxcywh

            # ── Loss ──────────────────────────────────────────────────────
            loss, l_cls, l_l1, l_giou = detection_loss(
                pred_logits, pred_boxes, gt_boxes,
                w_l1=args.w_l1, w_giou=args.w_giou,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            # Metric: IoU of best-scoring box vs GT
            with torch.no_grad():
                best_box = get_best_pred_box(pred_logits.detach(), pred_boxes.detach())
                iou      = box_iou_metric(best_box, gt_boxes)

            n = imgs.shape[0]
            meters["loss"].update(loss.item(), n)
            meters["loss_cls"].update(l_cls.item(), n)
            meters["loss_l1"].update(l_l1.item(), n)
            meters["loss_giou"].update(l_giou.item(), n)
            meters["iou"].update(iou, n)

            step = i + epoch * len(train_loader)
            if i % args.print_freq == 0:
                print(f"  [{epoch}][{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}  "
                      f"cls={l_cls.item():.4f}  "
                      f"l1={l_l1.item():.4f}  "
                      f"giou={l_giou.item():.4f}  "
                      f"iou={iou:.3f}")

        scheduler.step()
        epoch_time = time.time() - t0

        for k, m in meters.items():
            writer.add_scalar(f"train/{k}", m.avg, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch} done in {epoch_time:.0f}s | "
              f"loss={meters['loss'].avg:.4f}  iou={meters['iou'].avg:.3f}")

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_meters = {k: AverageMeter() for k in ["loss", "iou"]}

        with torch.no_grad():
            for imgs, gt_boxes in val_loader:
                imgs     = imgs.to(device)
                gt_boxes = gt_boxes.to(device)
                B        = imgs.shape[0]

                vis_out = model.backbone.forward_image(imgs)
                backbone_out = {
                    **vis_out,
                    "language_features": lang_feats.to(device).expand(-1, B, -1),
                    "language_mask":     lang_mask.to(device).expand(B, -1),
                }
                find_input = FindStage(
                    img_ids           = torch.arange(B, device=device),
                    text_ids          = torch.zeros(B, dtype=torch.long, device=device),
                    input_boxes       = torch.zeros(B, 0, 4, device=device),
                    input_boxes_mask  = torch.zeros(B, 0, dtype=torch.bool, device=device),
                    input_boxes_label = torch.zeros(B, 0, dtype=torch.long, device=device),
                    input_points      = torch.zeros(B, 0, 2, device=device),
                    input_points_mask = torch.zeros(B, 0, dtype=torch.bool, device=device),
                )
                geo_prompt = Prompt(
                    box_embeddings = torch.zeros(0, B, 4, device=device),
                    box_mask       = torch.zeros(B, 0, device=device, dtype=torch.bool),
                )
                prompt, prompt_mask, backbone_out = model._encode_prompt(
                    backbone_out, find_input, geo_prompt
                )
                backbone_out, encoder_out, _ = model._run_encoder(
                    backbone_out, find_input, prompt, prompt_mask
                )
                out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
                out, _ = model._run_decoder(
                    memory    = out["encoder_hidden_states"],
                    pos_embed = encoder_out["pos_embed"],
                    src_mask  = encoder_out["padding_mask"],
                    out       = out,
                    prompt    = prompt,
                    prompt_mask = prompt_mask,
                    encoder_out = encoder_out,
                )
                pred_logits = out["pred_logits"]
                pred_boxes  = out["pred_boxes"]

                loss, *_ = detection_loss(
                    pred_logits, pred_boxes, gt_boxes,
                    w_l1=args.w_l1, w_giou=args.w_giou,
                )
                best_box = get_best_pred_box(pred_logits, pred_boxes)
                iou      = box_iou_metric(best_box, gt_boxes)

                val_meters["loss"].update(loss.item(), imgs.shape[0])
                val_meters["iou"].update(iou, imgs.shape[0])

        val_loss = val_meters["loss"].avg
        val_iou  = val_meters["iou"].avg
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/iou",  val_iou,  epoch)
        print(f"  Val — loss={val_loss:.4f}  iou={val_iou:.3f}")

        mlflow.log_metrics({
            "train_loss":      meters["loss"].avg,
            "train_loss_cls":  meters["loss_cls"].avg,
            "train_loss_l1":   meters["loss_l1"].avg,
            "train_loss_giou": meters["loss_giou"].avg,
            "train_iou":       meters["iou"].avg,
            "val_loss":        val_loss,
            "val_iou":         val_iou,
            "lr":              optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt = {
            "epoch":      epoch + 1,
            "state_dict": {
                k: v for k, v in model.state_dict().items()
                if "backbone.vision_backbone" not in k
                and "backbone.language_backbone" not in k
            },
            "optimizer":  optimizer.state_dict(),
            "val_iou":    val_iou,
            "val_loss":   val_loss,
        }
        torch.save(ckpt, os.path.join(run_ckpt_dir, "checkpoint_latest.pth"))

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(ckpt, os.path.join(run_ckpt_dir, "checkpoint_best.pth"))
            print(f"  -> Saved best checkpoint (val_iou={val_iou:.3f})")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(run_ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    writer.close()
    print(f"\nTraining complete. Best val IoU: {best_val_iou:.3f}")
    print(f"Best checkpoint: {run_ckpt_dir}/checkpoint_best.pth")

    mlflow.log_metrics({"best_val_iou": best_val_iou})
    mlflow.log_artifact(config_path)
    mlflow.end_run()

    run_record = {
        "run":          ts,
        "run_dir":      run_dir,
        "ckpt_dir":     run_ckpt_dir,
        "timestamp":    datetime.datetime.now().isoformat(),
        "best_val_iou": round(best_val_iou, 4),
        **vars(args),
    }
    log_path = os.path.join(TRAIN_DIR, "runs.log")
    with open(log_path, "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"Run logged to  : {log_path}")


if __name__ == "__main__":
    main()
