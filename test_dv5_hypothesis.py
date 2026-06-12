"""
Test the robot-mismatch hypothesis: training data is dv5; the failing case is
dv Xi (different scope). Run the same crop-vs-no-crop diagnostic on a dv5
video where catheter pull happens at 15:07-15:13.

Scope-content detector: largest connected component of (gray > 10) AND
(local 16x16 std > 15). Works for both center-pillar (dv Xi) and
top-left-quadrant (dv5) layouts because UI panels have low texture.
"""
import os, sys, time
import cv2
import numpy as np
import torch
import joblib

SAM3_DIR = "/sc/arion/projects/video_rarp/neel_projects/segmentation_prostate/sam3"
sys.path.insert(0, SAM3_DIR)

from localize_multiclass import (
    IMG_SIZE, IMG_MEAN, IMG_STD, NUM_CLASSES, CLASS_NAMES, load_sam2,
)

VIDEO = "/sc/arion/projects/video_rarp/neel_projects/gg1_videos_daniel/SUBJ_1b7d93c2_Y2025_DOY143/M_10152025084201_U013419101506001_2_002_0001-01.MP4"
CKPT  = os.path.join(SAM3_DIR, "multiclass_clf_small.joblib")
WINDOW_START_S = 870.0    # 14:30
WINDOW_END_S   = 940.0    # 15:40
HIGHLIGHT      = (907.0, 913.0)   # 15:07-15:13  (catheter pull GT)
SAMPLE_FPS     = 1.0
DIAG_DIR       = os.path.join(SAM3_DIR, "diagnostic_images", "dv5")
os.makedirs(DIAG_DIR, exist_ok=True)
OUT_PNG        = os.path.join(DIAG_DIR, "dv5_diagnostic.png")
PROBE_VIS      = os.path.join(DIAG_DIR, "dv5_bbox_vis.jpg")

# Hardcoded scope bbox by visual inspection: scope window is in the top-left
# of the dv5 layout (HUD panel on right, instrument strip on bottom).
HARDCODED_DV5_BBOX = (25, 16, 990, 990)   # (x0, y0, x1, y1)


def detect_scope_bbox(frame):
    """Largest connected component of textured non-black pixels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=(16, 16))
    blur_sq = cv2.boxFilter((gray.astype(np.float32)) ** 2,
                             ddepth=cv2.CV_32F, ksize=(16, 16))
    local_std = np.sqrt(np.maximum(0.0, blur_sq - blur ** 2))
    mask = ((gray > 10) & (local_std > 15)).astype(np.uint8) * 255
    # Morphological close to merge thin gaps inside scope content
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[biggest, cv2.CC_STAT_LEFT])
    y = int(stats[biggest, cv2.CC_STAT_TOP])
    w = int(stats[biggest, cv2.CC_STAT_WIDTH])
    h = int(stats[biggest, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


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
        frames.append(f); times.append(t / fps)
    cap.release()
    return frames, np.array(times)


def to_tensor(frame_bgr, bbox=None):
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        frame_bgr = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMG_MEAN) / IMG_STD


@torch.no_grad()
def featurize(model, frames, bbox, device, batch_size=16):
    feats_all = []
    for i in range(0, len(frames), batch_size):
        batch = torch.stack([to_tensor(f, bbox) for f in frames[i:i+batch_size]])
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

    print(f"Probing dv5 video at t=910s ...")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(910 * fps))
    ret, ref = cap.read()
    cap.release()
    assert ret
    H, W = ref.shape[:2]
    bbox = HARDCODED_DV5_BBOX
    x0, y0, x1, y1 = bbox
    print(f"  frame {W}x{H}  hardcoded scope_bbox=(x:[{x0},{x1}], y:[{y0},{y1}])  "
          f"size={x1-x0}x{y1-y0}  area={100*(x1-x0)*(y1-y0)/(W*H):.1f}% of source")

    vis = ref.copy()
    cv2.rectangle(vis, (x0, y0), (x1-1, y1-1), (0, 255, 255), 4)
    cv2.imwrite(PROBE_VIS, vis)
    print(f"  bbox vis → {PROBE_VIS}")

    crop = ref[y0:y1, x0:x1]
    crop_path = os.path.join(DIAG_DIR, "dv5_crop_raw.jpg")
    no_crop_path = os.path.join(DIAG_DIR, "dv5_encoder_input_NO_CROP.jpg")
    with_crop_path = os.path.join(DIAG_DIR, "dv5_encoder_input_WITH_CROP.jpg")
    cv2.imwrite(crop_path, crop)
    resized_no_crop = cv2.resize(ref, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(no_crop_path, resized_no_crop)
    resized_with_crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                    interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(with_crop_path, resized_with_crop)
    print(f"  crop raw           → {crop_path} "
          f"({crop.shape[1]}x{crop.shape[0]})")
    print(f"  encoder NO-CROP   → {no_crop_path} (1024x1024)")
    print(f"  encoder WITH-CROP → {with_crop_path} (1024x1024)")

    print(f"\nLoading classifier ({CKPT}) ...")
    d = joblib.load(CKPT)
    clf, mean, std = d["clf"], d["mean"], d["std"]

    print(f"\nLoading SAM2 backbone (small) ...")
    model = load_sam2("small", device)

    print(f"\nGrabbing window {WINDOW_START_S:.0f}s-{WINDOW_END_S:.0f}s "
          f"at {SAMPLE_FPS} fps ...")
    t0 = time.perf_counter()
    frames, times_s = grab_window_frames(VIDEO, WINDOW_START_S,
                                          WINDOW_END_S, SAMPLE_FPS)
    print(f"  {len(frames)} frames in {time.perf_counter()-t0:.1f}s")

    print(f"\nFeaturising NO-CROP ...")
    t0 = time.perf_counter()
    feats_no = featurize(model, frames, bbox=None, device=device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"\nFeaturising WITH-CROP ...")
    t0 = time.perf_counter()
    feats_cr = featurize(model, frames, bbox=bbox, device=device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    z_no = (feats_no - mean) / std
    z_cr = (feats_cr - mean) / std
    print(f"\nFeature-distribution diagnostics:")
    print(f"  avg-pool z mean: NO_CROP={z_no[:, :256].mean()+z_no[:, 512:576].mean():+.3f}  "
          f"WITH_CROP={z_cr[:, :256].mean()+z_cr[:, 512:576].mean():+.3f}")
    print(f"  max-pool z mean: NO_CROP={z_no[:, 256:512].mean()+z_no[:, 576:].mean():+.3f}  "
          f"WITH_CROP={z_cr[:, 256:512].mean()+z_cr[:, 576:].mean():+.3f}")

    p_no = predict(feats_no, clf, mean, std)
    p_cr = predict(feats_cr, clf, mean, std)

    print(f"\nPer-frame top-3 probabilities (★ = inside catheter_pull GT window 15:07-15:13):")
    print(f"  {'t(s)':>5}  NO-CROP                                    WITH-CROP")
    for i, t in enumerate(times_s):
        in_w = "★" if HIGHLIGHT[0] <= t <= HIGHLIGHT[1] else " "
        top_no = np.argsort(p_no[i])[::-1][:3]
        top_cr = np.argsort(p_cr[i])[::-1][:3]
        s_no = "  ".join(f"{CLASS_NAMES[c][:14]:<14}={p_no[i,c]:.2f}" for c in top_no)
        s_cr = "  ".join(f"{CLASS_NAMES[c][:14]:<14}={p_cr[i,c]:.2f}" for c in top_cr)
        print(f"  {t:>4.0f} {in_w} | {s_no}    |    {s_cr}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    plot_classes = [(1, "catheter_pull"),  (3, "posterior_cut"),
                    (5, "vas_cut"),        (7, "apical_cut"),
                    (9, "endobag"),        (10, "post_endobag")]
    cmap = plt.get_cmap("tab10")
    for ax, p, title in [(axes[0], p_no, "dv5 — NO CROP — emission probabilities"),
                          (axes[1], p_cr, "dv5 — WITH scope-bbox crop — emission probabilities")]:
        for i, (cls, lbl) in enumerate(plot_classes):
            lw = 2.0 if cls == 1 else 1.2
            ax.plot(times_s, p[:, cls], color=cmap(i), linewidth=lw, label=lbl)
        ax.axvspan(HIGHLIGHT[0], HIGHLIGHT[1], color="yellow", alpha=0.30,
                   label="GT catheter_pull (15:07-15:13)")
        ax.set_title(title)
        ax.set_ylabel("P(class)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time in chunk 1 (s)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    print(f"\nPlot → {OUT_PNG}")


if __name__ == "__main__":
    main()
