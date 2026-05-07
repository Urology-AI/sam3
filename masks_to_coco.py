#!/usr/bin/env python3
"""
masks_to_coco.py
================
Convert paired image/mask dataset to COCO-format JSON for SAM3 detector training.

Source layout (prostate_tracker_long_qpr_19):
    images/{stem}.jpg
    masks/{stem}.png   (binary: 0 = background, 255 = foreground)

Output:
    detector_training/data/annotations_train.json
    detector_training/data/annotations_val.json
    detector_training/data/images/       (symlinks to source images)

COCO bbox format: [x, y, w, h] in pixel coordinates (top-left origin).
The training script converts these to normalized [cx, cy, w, h] at load time.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

SAM3_DIR  = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR  = os.path.join(SAM3_DIR, "..", "..", "autosam-instruments-GraSP-trained", "sam2")
DATA_OUT  = os.path.join(SAM3_DIR, "detector_training", "data")


def mask_to_bbox(mask_path: str):
    """
    Return tight [x, y, w, h] bounding box (pixel coords) of the foreground
    region in a binary mask, or None if the mask is empty.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = (mask > 128).astype(np.uint8)
    if fg.sum() == 0:
        return None, mask.shape
    ys, xs = np.where(fg)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1], mask.shape   # [x,y,w,h]


def build_coco(stems, img_dir, mask_dir, category_id=1):
    images      = []
    annotations = []
    ann_id      = 1
    skipped     = 0

    for img_id, stem in enumerate(stems, 1):
        img_path  = os.path.join(img_dir,  stem + ".jpg")
        mask_path = os.path.join(mask_dir, stem + ".png")

        img = Image.open(img_path)
        w, h = img.size

        bbox, _ = mask_to_bbox(mask_path)
        if bbox is None:
            skipped += 1
            continue

        images.append({
            "id":        img_id,
            "file_name": stem + ".jpg",
            "width":     w,
            "height":    h,
        })

        annotations.append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": category_id,
            "bbox":        bbox,          # [x, y, w, h]
            "area":        bbox[2] * bbox[3],
            "iscrowd":     0,
        })
        ann_id += 1

    if skipped:
        print(f"  Skipped {skipped} empty masks.")

    return {
        "images":      images,
        "annotations": annotations,
        "categories":  [{"id": category_id, "name": "prostate gland"}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir",
        default=os.path.join(SAM2_DIR, "projects", "prostate_tracker_long_qpr_19"),
    )
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    img_dir  = os.path.join(args.project_dir, "images")
    mask_dir = os.path.join(args.project_dir, "masks")

    # Collect paired stems
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg")
        and os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".png"))
    )
    print(f"Found {len(stems)} paired samples in {args.project_dir}")

    # Deterministic split
    rng     = np.random.RandomState(args.seed)
    idx     = rng.permutation(len(stems))
    split   = int(len(stems) * args.train_ratio)
    train_s = [stems[i] for i in idx[:split]]
    val_s   = [stems[i] for i in idx[split:]]
    print(f"Split: {len(train_s)} train / {len(val_s)} val")

    # Output dirs
    os.makedirs(DATA_OUT, exist_ok=True)
    link_dir = os.path.join(DATA_OUT, "images")
    os.makedirs(link_dir, exist_ok=True)

    # Symlink images so the training script can find them by stem
    linked = 0
    for stem in stems:
        src = os.path.abspath(os.path.join(img_dir, stem + ".jpg"))
        dst = os.path.join(link_dir, stem + ".jpg")
        if not os.path.exists(dst):
            os.symlink(src, dst)
            linked += 1
    print(f"Linked {linked} new image symlinks → {link_dir}")

    # Write COCO JSONs
    for split_name, split_stems in [("train", train_s), ("val", val_s)]:
        coco = build_coco(split_stems, img_dir, mask_dir)
        out  = os.path.join(DATA_OUT, f"annotations_{split_name}.json")
        with open(out, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Wrote {len(coco['images'])} {split_name} samples → {out}")


if __name__ == "__main__":
    main()
