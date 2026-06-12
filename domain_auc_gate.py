#!/usr/bin/env python3
"""
domain_auc_gate.py
==================
Label-free go/no-go for a backbone swap. Asks one question: can a linear probe
tell the intuitive domain apart from the Sony domain in this backbone's feature
space? AUC=1.0 -> domains are disjoint, transfer is hopeless. AUC~0.5 ->
domains overlap, transfer is possible.

No phase labels are used. We only need the domain each frame came from.

Two crop tightnesses are evaluated so we can separate the confounds:
  bbox    : the surgical-view box you pass (still contains HUD furniture)
  tissue  : central fraction of that box (no HUD strips / borders -> pure tissue)

If bbox AUC is ~1.0 but tissue AUC drops, the separation was furniture/framing
(fixable by cropping). If both stay ~1.0, it is the tissue rendering itself
(codec/ISP) -> the hard case.

Descriptors reported (both, since "any reasonable readout separates" is the
bad-news condition):
  cls    : DINO CLS token (768)
  pooled : [CLS, avgpool patches, maxpool patches] (2304)

Run (GPU auto-detected; falls back to CPU):
  python3 domain_auc_gate.py
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

INTU_DIR = "/sc/arion/projects/video_rarp/neel_projects/intuitive_videos"
SONY_VIDEO = ("/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/"
              "SUBJ_1b7d93c2_Y2025_DOY143/"
              "M_10152025084201_U013419101506001_2_002_0001-01.MP4")

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def crop_bbox(fr, box):
    h, w = fr.shape[:2]
    x0, y0, x1, y1 = box
    return fr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def center_frac(fr, frac):
    h, w = fr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return fr[y0:y0 + ch, x0:x0 + cw]


def prep(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)
    return (t - MEAN) / STD


def sample_indices(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return np.linspace(total * 0.05, total * 0.95, n).astype(int)


def grab_frames(path, indices, box, tissue_frac):
    """Returns two lists of cropped frames: (bbox-crop, tissue-crop)."""
    cap = cv2.VideoCapture(path)
    bbox_imgs, tissue_imgs = [], []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, fr = cap.read()
        if not ret or fr is None:
            continue
        c = crop_bbox(fr, box)
        bbox_imgs.append(c)
        tissue_imgs.append(center_frac(c, tissue_frac))
    cap.release()
    return bbox_imgs, tissue_imgs


@torch.no_grad()
def encode(model, imgs, size, device, batch, use_bf16):
    cls_all, pooled_all = [], []
    for i in range(0, len(imgs), batch):
        t = torch.stack([prep(im, size) for im in imgs[i:i + batch]]).to(device)
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_features(t)
        else:
            out = model.forward_features(t)
        cls = out["x_norm_clstoken"].float()
        pat = out["x_norm_patchtokens"].float()
        pooled = torch.cat([cls, pat.mean(1), pat.amax(1)], dim=1)
        cls_all.append(cls.cpu().numpy())
        pooled_all.append(pooled.cpu().numpy())
    return np.concatenate(cls_all), np.concatenate(pooled_all)


def domain_auc(Xi, Xs, seed=0):
    """Cross-validated linear domain-AUC + mean-shift magnitude (in pooled std)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([Xi, Xs])
    y = np.r_[np.zeros(len(Xi)), np.ones(len(Xs))]
    Xz = StandardScaler().fit_transform(X)
    auc = cross_val_score(LogisticRegression(max_iter=2000), Xz, y,
                          cv=5, scoring="roc_auc").mean()
    sd = 0.5 * (Xi.std(0) + Xs.std(0)) + 1e-6
    zshift = np.abs(Xi.mean(0) - Xs.mean(0)) / sd
    return auc, zshift.mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sony_video",  default=SONY_VIDEO)
    p.add_argument("--sony_box",    default="27,-1,1318,1068")
    p.add_argument("--intu_box",    default="6,9,1274,1017")
    p.add_argument("--intu_cases",  default="213,222,246,268")
    p.add_argument("--n_per_intu",  type=int, default=100)
    p.add_argument("--n_sony",      type=int, default=400)
    p.add_argument("--tissue_frac", type=float, default=0.6)
    p.add_argument("--img_size",    type=int, default=518)
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--model",       default="dinov2_vitb14")
    return p.parse_args()


def main():
    args = parse_args()
    sony_box = [int(v) for v in args.sony_box.split(",")]
    intu_box = [int(v) for v in args.intu_box.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    print(f"device: {device}  bf16: {use_bf16}")

    torch.hub.set_dir("/hpc/users/gahaln01/.cache/torch/hub")
    model = torch.hub.load("facebookresearch/dinov2", args.model,
                           pretrained=True, source="github").to(device).eval()
    print(f"{args.model} loaded  embed_dim={model.embed_dim}")

    # ── gather frames ────────────────────────────────────────────────────────
    print("\nsampling intuitive frames ...")
    intu_bbox, intu_tissue = [], []
    for cid in args.intu_cases.split(","):
        vp = os.path.join(INTU_DIR, f"case_{cid.strip()}_clipped.mp4")
        if not os.path.exists(vp):
            print(f"  [skip] {vp} not found"); continue
        idx = sample_indices(vp, args.n_per_intu)
        b, t = grab_frames(vp, idx, intu_box, args.tissue_frac)
        intu_bbox += b; intu_tissue += t
        print(f"  case {cid}: {len(b)} frames")
    print(f"sampling sony frames from {os.path.basename(args.sony_video)} ...")
    sidx = sample_indices(args.sony_video, args.n_sony)
    sony_bbox, sony_tissue = grab_frames(args.sony_video, sidx, sony_box,
                                         args.tissue_frac)
    print(f"  sony: {len(sony_bbox)} frames")

    # ── encode + AUC, per crop ───────────────────────────────────────────────
    print(f"\n{'crop':<8} {'descriptor':<8} {'domain-AUC':>11} {'mean |z-shift|':>15}")
    print("-" * 46)
    results = {}
    for crop_name, intu_imgs, sony_imgs in (
            ("bbox",   intu_bbox,   sony_bbox),
            ("tissue", intu_tissue, sony_tissue)):
        Ci, Pi = encode(model, intu_imgs, args.img_size, device, args.batch, use_bf16)
        Cs, Ps = encode(model, sony_imgs, args.img_size, device, args.batch, use_bf16)
        for desc_name, Xi, Xs in (("cls", Ci, Cs), ("pooled", Pi, Ps)):
            auc, z = domain_auc(Xi, Xs)
            results[(crop_name, desc_name)] = auc
            print(f"{crop_name:<8} {desc_name:<8} {auc:>11.4f} {z:>15.3f}")

    print("\ninterpretation:")
    print("  AUC ~1.0 -> domains disjoint (transfer fails);  ~0.5 -> overlap (transfer possible)")
    b = results[("bbox", "pooled")]; t = results[("tissue", "pooled")]
    if b > 0.95 and t < 0.85:
        print(f"  bbox={b:.2f} high but tissue={t:.2f} lower -> separation is HUD/framing, "
              "fixable by tighter crop.")
    elif t > 0.95:
        print(f"  tissue={t:.2f} still high -> tissue rendering itself separates; "
              "DINO does NOT bridge the gap on pure anatomy.")
    else:
        print(f"  bbox={b:.2f} tissue={t:.2f} -> partial overlap; worth the full extraction + real probe.")


if __name__ == "__main__":
    main()
