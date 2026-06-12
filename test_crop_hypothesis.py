"""
Diagnostic: confirm letterbox-bars OOD hypothesis on chunk 2 around 8:40-8:50,
where ground truth says posterior bladder neck dissection is happening.

For each of 90 frames sampled at 1 fps from 8:00-9:30:
  (a) Extract features via the CURRENT pipeline (no crop, 1920x1080 -> 1024^2).
  (b) Extract features via a CROPPED pipeline (auto-detect non-black column
      range, crop to that, then resize to 1024^2).
Score both with the saved multiclass_clf_small.joblib classifier and plot
per-class probabilities side by side. Highlight the 8:40-8:50 window.

If (b) shows a posterior_cut spike at 8:40-8:50 and (a) does not, the
letterbox-OOD hypothesis is confirmed.
"""

import os, sys, time
import cv2
import numpy as np
import torch
import joblib

SAM3_DIR = "/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3"
sys.path.insert(0, SAM3_DIR)

from localize_multiclass import (
    IMG_SIZE, IMG_MEAN, IMG_STD,
    NUM_CLASSES, CLASS_NAMES, EVENT_STATE_NAMES,
    load_sam2,
)

VIDEO = "/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_0e44bf22_Y2026_DOY114/M_11062025075420_0U13422110622320_2_001_0002-01.MP4"
CKPT  = os.path.join(SAM3_DIR, "multiclass_clf_small.joblib")
WINDOW_START_S = 480.0   # 8:00
WINDOW_END_S   = 570.0   # 9:30
HIGHLIGHT      = (520.0, 530.0)  # 8:40-8:50  (posterior bladder neck dissection)
SAMPLE_FPS     = 1.0
OUT_PNG        = "/tmp/crop_diagnostic.png"


def detect_content_bbox(video_path, probe_time_s=525.0, threshold=5):
    """Return (x_min, x_max) — leftmost and rightmost columns with non-black
    content in a probe frame. Vertical extent unchanged (the HUD overlay
    at the bottom is content the classifier has seen during training)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(probe_time_s * fps))
    ret, ref = cap.read()
    cap.release()
    assert ret, "failed to read probe frame"
    col_means = ref.mean(axis=(0, 2))                 # (W,)
    nz = np.where(col_means > threshold)[0]
    return int(nz[0]), int(nz[-1] + 1), ref.shape


def grab_window_frames(video_path, start_s, end_s, sample_fps):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round(fps / sample_fps)))
    start_idx = int(start_s * fps)
    end_idx   = int(end_s * fps)
    targets = list(range(start_idx, end_idx + 1, stride))
    cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
    cur = targets[0]
    frames, times = [], []
    for t in targets:
        while cur < t:
            cap.grab(); cur += 1
        ret, f = cap.read(); cur += 1
        if not ret: break
        frames.append(f)
        times.append(t / fps)
    cap.release()
    return frames, np.array(times)


def to_tensor(frame_bgr, crop_bbox):
    if crop_bbox is not None:
        x0, x1 = crop_bbox
        frame_bgr = frame_bgr[:, x0:x1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


@torch.no_grad()
def featurize(model, frames, crop_bbox, device, batch_size=16):
    feats_all = []
    for i in range(0, len(frames), batch_size):
        batch = torch.stack([to_tensor(f, crop_bbox) for f in frames[i:i+batch_size]])
        imgs = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_image(imgs)
        fpn = out["backbone_fpn"]
        pooled = []
        for ff in fpn[1:]:
            pooled.append(ff.mean(dim=[2, 3]))
            pooled.append(ff.amax(dim=[2, 3]))
        feats_all.append(torch.cat(pooled, dim=1).float().cpu().numpy())
    return np.concatenate(feats_all, axis=0)


def predict(feats, clf, mean, std):
    z = (feats - mean) / std
    proba = clf.predict_proba(z)
    full = np.zeros((len(proba), NUM_CLASSES), dtype=np.float32)
    for i, c in enumerate(clf.classes_):
        full[:, int(c)] = proba[:, i]
    return full


def main():
    device = torch.device("cuda")
    print("Detecting content bbox on probe frame (t=525s)...")
    x0, x1, shape = detect_content_bbox(VIDEO)
    H, W = shape[0], shape[1]
    print(f"  frame shape    : {W}x{H}")
    print(f"  content x range: [{x0}, {x1}]  (width={x1-x0}, "
          f"{100*(x1-x0)/W:.1f}% of source)")
    print(f"  black pillars  : left={x0}px, right={W-x1}px")

    print(f"\nLoading saved classifier ({CKPT})...")
    d = joblib.load(CKPT)
    clf, mean, std = d["clf"], d["mean"], d["std"]
    print(f"  classes seen during training: {clf.classes_.tolist()}")

    print(f"\nLoading SAM2 backbone (small) ...")
    model = load_sam2("small", device)

    print(f"\nGrabbing window {WINDOW_START_S:.0f}s-{WINDOW_END_S:.0f}s at "
          f"{SAMPLE_FPS} fps ...")
    t0 = time.perf_counter()
    frames, times_s = grab_window_frames(VIDEO, WINDOW_START_S, WINDOW_END_S,
                                          SAMPLE_FPS)
    print(f"  {len(frames)} frames in {time.perf_counter()-t0:.1f}s")

    print(f"\nFeaturising NO-CROP ...")
    t0 = time.perf_counter()
    feats_no = featurize(model, frames, crop_bbox=None, device=device)
    print(f"  {time.perf_counter()-t0:.1f}s  feat dim = {feats_no.shape[1]}")

    print(f"\nFeaturising WITH-CROP (x={x0}..{x1}) ...")
    t0 = time.perf_counter()
    feats_cr = featurize(model, frames, crop_bbox=(x0, x1), device=device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"\nFeature-distribution diagnostics (avg-pool channel shift):")
    z_no = (feats_no - mean) / std
    z_cr = (feats_cr - mean) / std
    avg_dim = 256 + 64
    print(f"  z-score mean over all 640 dims:  NO_CROP={z_no.mean():+.3f}  "
          f"WITH_CROP={z_cr.mean():+.3f}")
    print(f"  z-score mean over avg-pool 320d: NO_CROP={z_no[:, :256].mean()+z_no[:, 512:576].mean():+.3f}  "
          f"WITH_CROP={z_cr[:, :256].mean()+z_cr[:, 512:576].mean():+.3f}")
    print(f"  z-score mean over max-pool 320d: NO_CROP={z_no[:, 256:512].mean()+z_no[:, 576:].mean():+.3f}  "
          f"WITH_CROP={z_cr[:, 256:512].mean()+z_cr[:, 576:].mean():+.3f}")

    proba_no = predict(feats_no, clf, mean, std)
    proba_cr = predict(feats_cr, clf, mean, std)

    print(f"\nPer-frame top-3 class probabilities (★ = inside posterior-cut GT window):")
    print(f"  {'t(s)':>6}  NO-CROP top-3                          WITH-CROP top-3")
    for i, t in enumerate(times_s):
        in_w = "★" if HIGHLIGHT[0] <= t <= HIGHLIGHT[1] else " "
        top_no = np.argsort(proba_no[i])[::-1][:3]
        top_cr = np.argsort(proba_cr[i])[::-1][:3]
        s_no = "  ".join(f"{CLASS_NAMES[c][:14]:<14}={proba_no[i,c]:.2f}" for c in top_no)
        s_cr = "  ".join(f"{CLASS_NAMES[c][:14]:<14}={proba_cr[i,c]:.2f}" for c in top_cr)
        print(f"  {t:>5.0f}s {in_w} | {s_no}    |    {s_cr}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    plot_classes = [(1, "catheter_pull"),  (3, "posterior_cut"),
                    (5, "vas_cut"),        (7, "apical_cut"),
                    (9, "endobag"),        (10, "post_endobag")]
    cmap = plt.get_cmap("tab10")
    for ax, p, title in [(axes[0], proba_no, "NO CROP (current pipeline) — emission probabilities"),
                          (axes[1], proba_cr, "WITH letterbox crop — emission probabilities")]:
        for i, (cls, lbl) in enumerate(plot_classes):
            lw = 2.0 if cls == 3 else 1.2
            ax.plot(times_s, p[:, cls], color=cmap(i), linewidth=lw, label=lbl)
        ax.axvspan(HIGHLIGHT[0], HIGHLIGHT[1], color="yellow", alpha=0.30,
                   label="GT posterior_cut (8:40-8:50)")
        ax.set_title(title)
        ax.set_ylabel("P(class)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time in chunk 2 (s)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"\nPlot → {OUT_PNG}")


if __name__ == "__main__":
    main()
