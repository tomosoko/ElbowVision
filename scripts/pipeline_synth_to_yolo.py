#!/usr/bin/env python3
"""
Integrated Pipeline: Extension CT → Synthesized Flexion DRR → YOLO Keypoint Detection
========================================================================================

Clinical scenario:
  Patient has CT in extension (180°) only.
  We synthesize the LAT X-ray at any target flexion angle,
  then run YOLO keypoint detection on the synthesized image.

Usage:
  cd /Users/kohei/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/pipeline_synth_to_yolo.py
  python scripts/pipeline_synth_to_yolo.py --target_angles 90 120 150
  python scripts/pipeline_synth_to_yolo.py --model runs/elbow_v6/weights/best.pt

Output:
  results/pipeline_synth_to_yolo/
  ├── synth_90deg.png       ← synthesized DRR at 90°
  ├── yolo_90deg.png        ← YOLO keypoint overlay
  ├── pipeline_90deg.png    ← side-by-side comparison
  └── pipeline_report.txt  ← keypoint coordinates + confidence
"""

import argparse
import os
import sys
import math

import cv2
import numpy as np
from scipy.ndimage import affine_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = "/Users/kohei/develop/research/ElbowVision"
sys.path.insert(0, os.path.join(ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume, auto_detect_landmarks, generate_drr,
    rotation_matrix_z, rotate_volume_and_landmarks,
)

# ── Paths ──────────────────────────────────────────────────────────────────
CT_DIR = os.path.join(ROOT, "data/raw_dicom/ct_volume",
                      "ﾃｽﾄ 008_0009900008_20260310_108Y_F_000")
DEFAULT_MODEL = os.path.join(ROOT, "elbow-api/models/yolo_pose_best.pt")
REAL_XRAY_LAT = os.path.join(ROOT, "data/real_xray/images/008_LAT.png")
OUT_DIR = os.path.join(ROOT, "results/pipeline_synth_to_yolo")
os.makedirs(OUT_DIR, exist_ok=True)

HU_MIN, HU_MAX = 50, 1000
TARGET_SIZE = 256   # Match YOLO training size
SID_MM = 1000.0
LATERALITY = "L"

KP_NAMES = [
    "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
    "forearm_shaft", "radial_head", "olecranon",
]
KP_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
]


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis
# ═══════════════════════════════════════════════════════════════════════════

def build_bone_split(vol, lm, joint_key="joint_center"):
    """Split volume into humerus and forearm parts along PD axis at joint center."""
    pd_size = vol.shape[0]
    jc = lm[joint_key]
    joint_pd = int(jc[0] * pd_size)

    bone_thresh = float(np.percentile(vol[vol > 0.01], 50))
    blend_half = max(3, int(pd_size * 0.04))

    hum_w = np.zeros(pd_size, dtype=np.float32)
    for i in range(pd_size):
        if i < joint_pd - blend_half:
            hum_w[i] = 1.0
        elif i > joint_pd + blend_half:
            hum_w[i] = 0.0
        else:
            hum_w[i] = 0.5 * (1.0 - (i - joint_pd) / blend_half)

    hum_w = hum_w[:, None, None]
    fore_w = 1.0 - hum_w

    vol_hum = vol * hum_w
    vol_fore = vol * fore_w
    return vol_hum, vol_fore, joint_pd, bone_thresh


def synthesize_lat_drr(vol_180, lm_180, voxel_mm, target_flexion_deg):
    """
    Synthesize LAT-view DRR from 180° (extension) CT at target flexion angle.
    Uses full-volume rotation (soft tissue included) with combined compositing.

    Best configuration from ct_to_xray_final.py:
      - Full volume (not bone-only)
      - Combined compositing (humerus + rotated forearm in 3D then DRR)
      - Gamma 1.1 post-processing
      - No CLAHE (degrades result)
    """
    pd_size, ap_size, ml_size = vol_180.shape
    vol_hum, vol_fore, joint_pd, _ = build_bone_split(vol_180, lm_180)

    # Rotation pivot: joint center in voxel space (PD, AP, ML)
    jc = lm_180["joint_center"]
    pivot = np.array([jc[0] * pd_size, jc[1] * ap_size, jc[2] * ml_size],
                     dtype=np.float64)

    # Angle delta from 180° to target
    angle_delta = math.radians(180.0 - target_flexion_deg)

    R = rotation_matrix_z(angle_delta)
    R_inv = R.T
    off = pivot - R_inv @ pivot

    vol_fore_rot = affine_transform(
        vol_fore, R_inv, offset=off, order=3, mode="constant", cval=0.0)

    # Combined volume DRR (avoids 2D compositing artifacts)
    vol_comb = vol_hum + vol_fore_rot
    drr = generate_drr(vol_comb, axis="LAT", sid_mm=SID_MM, voxel_mm=voxel_mm)

    # Gamma correction (1.1 = slight brightening, best from final.py)
    drr_f = (drr.astype(np.float32) / 255.0) ** (1.0 / 1.1)
    drr = np.clip(drr_f * 255.0, 0, 255).astype(np.uint8)

    return drr


# ═══════════════════════════════════════════════════════════════════════════
# YOLO inference
# ═══════════════════════════════════════════════════════════════════════════

def run_yolo_inference(model, img_bgr, target_size=256):
    """Run YOLO pose inference, return keypoints and confidence."""
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    results = model.predict(img_resized, conf=0.2, verbose=False)

    keypoints = []
    bbox_conf = 0.0

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        bbox_conf = float(box.conf[0])
        kps = results[0].keypoints.data[0].cpu().numpy()  # (6, 3): x, y, conf
        keypoints = kps
    return keypoints, bbox_conf, img_resized


def draw_keypoints(img, keypoints, scale=1.0):
    """Draw keypoints and skeleton on image."""
    out = img.copy()
    if len(keypoints) == 0:
        return out

    # Skeleton connections
    connections = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 5), (3, 4), (3, 5)]

    # Draw connections
    for i, j in connections:
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 > 0.2 and c2 > 0.2:
            pt1 = (int(x1 * scale), int(y1 * scale))
            pt2 = (int(x2 * scale), int(y2 * scale))
            cv2.line(out, pt1, pt2, (200, 200, 200), 1)

    # Draw keypoints
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf > 0.2:
            pt = (int(x * scale), int(y * scale))
            color = KP_COLORS[idx]
            cv2.circle(out, pt, 5, color, -1)
            cv2.circle(out, pt, 6, (255, 255, 255), 1)
            label = f"{KP_NAMES[idx][:4]} {conf:.2f}"
            cv2.putText(out, label, (pt[0] + 6, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(target_angles, model_path):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Pipeline: Extension CT → Synth DRR → YOLO")
    print("=" * 60)

    # Load 180° volume
    print("\nLoading 180° CT volume...")
    vol_180, spacing, lat, voxel_mm = load_ct_volume(
        CT_DIR, laterality=LATERALITY, series_num=4,
        hu_min=HU_MIN, hu_max=HU_MAX, target_size=TARGET_SIZE)
    lm_180 = auto_detect_landmarks(vol_180, laterality=lat)
    print(f"  Volume shape: {vol_180.shape}, voxel_mm={voxel_mm:.3f}")

    # Load YOLO model
    print(f"\nLoading YOLO model: {os.path.basename(model_path)}")
    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found: {model_path}")
        print(f"  Using first available model...")
        for candidate in [
            os.path.join(ROOT, "elbow-api/models/yolo_pose_best.pt"),
            os.path.join(ROOT, "runs/elbow_v4/weights/best.pt"),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break
    model = YOLO(model_path)
    print(f"  Loaded: {model_path}")

    # Load real X-ray for comparison
    real_lat = None
    if os.path.exists(REAL_XRAY_LAT):
        real_lat = cv2.imread(REAL_XRAY_LAT, cv2.IMREAD_GRAYSCALE)

    report_lines = []
    report_lines.append("Pipeline: Extension CT → Synthesized DRR → YOLO\n")
    report_lines.append(f"Model: {model_path}\n")
    report_lines.append("=" * 60 + "\n")

    for target_angle in target_angles:
        print(f"\n--- Target: {target_angle}° ---")

        # Step 1: Synthesize DRR
        print(f"  Synthesizing LAT DRR at {target_angle}°...")
        drr_gray = synthesize_lat_drr(vol_180, lm_180, voxel_mm, target_angle)
        drr_bgr = cv2.cvtColor(drr_gray, cv2.COLOR_GRAY2BGR)

        # Save synthesized DRR
        synth_path = os.path.join(OUT_DIR, f"synth_{target_angle}deg.png")
        cv2.imwrite(synth_path, drr_gray)

        # Step 2: YOLO inference on synthesized DRR
        print(f"  Running YOLO inference...")
        kps, bbox_conf, img_for_yolo = run_yolo_inference(model, drr_bgr)
        yolo_img = draw_keypoints(img_for_yolo, kps)

        yolo_path = os.path.join(OUT_DIR, f"yolo_{target_angle}deg.png")
        cv2.imwrite(yolo_path, yolo_img)

        # Step 3: Visualize pipeline result
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)

        titles = ["Synthesized DRR", "YOLO Detection"]
        imgs_show = [drr_gray, cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)]

        if real_lat is not None:
            titles = ["Synthesized DRR", "YOLO Detection", "Real LAT (ref)"]
            imgs_show.append(real_lat)

        for i, (title, im) in enumerate(zip(titles, imgs_show)):
            ax = fig.add_subplot(gs[0, i])
            cmap = "gray" if im.ndim == 2 else None
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Add pipeline info text
        ax_info = fig.add_subplot(gs[0, 3])
        ax_info.axis("off")
        info_text = f"Target: {target_angle}°\n\nBbox conf: {bbox_conf:.3f}\n\n"
        if len(kps) > 0:
            for i, (x, y, c) in enumerate(kps):
                info_text += f"{KP_NAMES[i][:8]}: {c:.2f}\n"
        else:
            info_text += "No detection\n"
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=8, va="top", family="monospace")

        fig.suptitle(f"Extension CT → {target_angle}° LAT → YOLO", fontsize=12)
        pipeline_path = os.path.join(OUT_DIR, f"pipeline_{target_angle}deg.png")
        fig.savefig(pipeline_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Report
        report_lines.append(f"\nAngle: {target_angle}°\n")
        report_lines.append(f"  Bbox conf: {bbox_conf:.3f}\n")
        if len(kps) > 0:
            for i, (x, y, c) in enumerate(kps):
                report_lines.append(f"  {KP_NAMES[i]:25s}: ({x:.1f}, {y:.1f}) conf={c:.3f}\n")
            n_detected = sum(1 for _, _, c in kps if c > 0.3)
            report_lines.append(f"  Detected (conf>0.3): {n_detected}/6\n")
        else:
            report_lines.append("  No detection\n")

        print(f"  Bbox conf: {bbox_conf:.3f}")
        if len(kps) > 0:
            for i, (x, y, c) in enumerate(kps):
                print(f"  {KP_NAMES[i]:25s}: conf={c:.3f}")

        print(f"  Saved: {pipeline_path}")

    # Save report
    report_path = os.path.join(OUT_DIR, "pipeline_report.txt")
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    print(f"\n{'=' * 60}")
    print(f"  Done. Results: {OUT_DIR}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synth CT-to-DRR → YOLO pipeline")
    parser.add_argument("--target_angles", nargs="+", type=int,
                        default=[90, 120, 150],
                        help="Target flexion angles in degrees (default: 90 120 150)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to YOLO model weights")
    args = parser.parse_args()

    run_pipeline(args.target_angles, args.model)
