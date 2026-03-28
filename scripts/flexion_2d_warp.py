#!/usr/bin/env python3
"""
Approach A: Flexion synthesis by 2D TPS warping of DRR images.

Generate a DRR from the 180deg (extended) volume, then warp it using
thin-plate spline interpolation so that landmarks move from their 180deg
projected positions to the 90deg projected positions.  Compare with the
real 90deg DRR.
"""

import sys, os, math
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))
from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    generate_drr,
    _project_kp_perspective,
)

# ── paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_DIR = os.path.join(
    BASE,
    "data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000",
)
OUT_DIR = os.path.join(BASE, "results/flexion_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)

LATERALITY = "L"
HU_MIN, HU_MAX = 50, 1000
TARGET_SIZE = 256
SERIES_180 = 4   # FC85 bone kernel, extended
SERIES_90  = 12  # FC85 bone kernel, 90deg flexion
SID_MM = 1000.0

KP_ORDER = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "forearm_shaft",
    "radial_head",
    "olecranon",
]


# ── helpers ──
def project_landmarks_2d(landmarks_norm, vol_shape, sid_mm, voxel_mm, axis="LAT"):
    """Project 3D normalised landmarks to 2D pixel coords (x, y) on an image
    of size (H, W) matching the DRR output dimensions."""
    NP, NA, NM = vol_shape
    SID_vox = sid_mm / voxel_mm

    if axis == "AP":
        H, W, D = NP, NM, NA
    else:  # LAT
        H, W, D = NP, NA, NM

    pts = {}
    for name in KP_ORDER:
        n_PD, n_AP, n_ML = landmarks_norm[name]
        if axis == "AP":
            D_s = max(SID_vox - NA, 1.0)
            px, py = _project_kp_perspective(n_PD, n_AP, n_ML, NA, D_s, SID_vox)
        else:
            D_s = max(SID_vox - NM, 1.0)
            px, py = _project_kp_perspective(n_PD, n_ML, n_AP, NM, D_s, SID_vox)
        # px, py are normalised [0,1] -> pixel coords
        pts[name] = (px * W, py * H)
    return pts, (H, W)


def tps_warp(src_img, src_pts, dst_pts):
    """Warp src_img so that src_pts move to dst_pts using scipy RBF TPS.

    We build two RBF interpolators:
      map_x: dst_coord -> src_x
      map_y: dst_coord -> src_y
    Then remap the source image.
    """
    from scipy.interpolate import RBFInterpolator

    h, w = src_img.shape[:2]

    # Add border anchor points (identity mapping) to stabilise the warp
    border = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
        (w // 4, 0), (3 * w // 4, 0), (w // 4, h - 1), (3 * w // 4, h - 1),
        (0, h // 4), (0, 3 * h // 4), (w - 1, h // 4), (w - 1, 3 * h // 4),
    ]
    src_all = np.array(list(src_pts) + border, dtype=np.float64)
    dst_all = np.array(list(dst_pts) + border, dtype=np.float64)

    # RBF: given destination coords, return source coords (inverse mapping)
    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0.0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0.0)

    # Build destination pixel grid
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xx.ravel(), yy.ravel()])  # (N, 2) in (x, y) order

    # Map destination pixels to source pixels
    map_x = rbf_x(grid).reshape(h, w).astype(np.float32)
    map_y = rbf_y(grid).reshape(h, w).astype(np.float32)

    warped = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def ssim_gray(a, b):
    """Compute SSIM between two uint8 grayscale images."""
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=255)


def draw_landmarks(img, pts, color, radius=4, thickness=2):
    """Draw landmark circles on a BGR image."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for name in KP_ORDER:
        x, y = pts[name]
        cv2.circle(out, (int(x), int(y)), radius, color, thickness)
        cv2.putText(out, name[:3], (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Approach A: 2D TPS warp of DRR for flexion synthesis")
    print("=" * 60)

    # ── 1. Load volumes ──
    print("\n[1] Loading 180deg volume (Series 4) ...")
    vol180, spacing180, lat180, vmm180 = load_ct_volume(
        CT_DIR, target_size=TARGET_SIZE, laterality=LATERALITY,
        series_num=SERIES_180, hu_min=HU_MIN, hu_max=HU_MAX,
    )
    print(f"    shape={vol180.shape}, voxel_mm={vmm180:.3f}")

    print("\n[1] Loading 90deg volume (Series 12) ...")
    vol90, spacing90, lat90, vmm90 = load_ct_volume(
        CT_DIR, target_size=TARGET_SIZE, laterality=LATERALITY,
        series_num=SERIES_90, hu_min=HU_MIN, hu_max=HU_MAX,
    )
    print(f"    shape={vol90.shape}, voxel_mm={vmm90:.3f}")

    # ── 2. Generate LAT DRRs ──
    print("\n[2] Generating LAT DRRs ...")
    drr180 = generate_drr(vol180, axis="LAT", sid_mm=SID_MM, voxel_mm=vmm180)
    drr90  = generate_drr(vol90,  axis="LAT", sid_mm=SID_MM, voxel_mm=vmm90)
    print(f"    DRR 180deg: {drr180.shape}, DRR 90deg: {drr90.shape}")

    # Resize to common size if needed
    target_h = max(drr180.shape[0], drr90.shape[0])
    target_w = max(drr180.shape[1], drr90.shape[1])
    if drr180.shape != (target_h, target_w):
        drr180 = cv2.resize(drr180, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if drr90.shape != (target_h, target_w):
        drr90 = cv2.resize(drr90, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # ── 3. Detect landmarks ──
    print("\n[3] Detecting landmarks ...")
    lm180 = auto_detect_landmarks(vol180, laterality=LATERALITY)
    lm90  = auto_detect_landmarks(vol90,  laterality=LATERALITY)

    print("    180deg landmarks (normalised PD, AP, ML):")
    for name in KP_ORDER:
        print(f"      {name:25s}: {lm180[name]}")
    print("    90deg landmarks (normalised PD, AP, ML):")
    for name in KP_ORDER:
        print(f"      {name:25s}: {lm90[name]}")

    # ── 4. Project landmarks to 2D ──
    print("\n[4] Projecting landmarks to 2D ...")
    # Use the DRR image size for projection
    # Note: project using each volume's own shape and voxel_mm
    pts180, (h180, w180) = project_landmarks_2d(
        lm180, vol180.shape, SID_MM, vmm180, axis="LAT"
    )
    pts90, (h90, w90) = project_landmarks_2d(
        lm90, vol90.shape, SID_MM, vmm90, axis="LAT"
    )

    # Scale projected points to common image size
    def scale_pts(pts, orig_hw, target_hw):
        sh = target_hw[0] / orig_hw[0]
        sw = target_hw[1] / orig_hw[1]
        return {k: (v[0] * sw, v[1] * sh) for k, v in pts.items()}

    pts180_sc = scale_pts(pts180, (h180, w180), (target_h, target_w))
    pts90_sc  = scale_pts(pts90,  (h90, w90),   (target_h, target_w))

    print("    Projected 2D landmarks (pixel coords on common image):")
    print(f"    {'Landmark':25s} {'180deg (x,y)':20s} {'90deg (x,y)':20s} {'Delta':15s}")
    for name in KP_ORDER:
        x1, y1 = pts180_sc[name]
        x2, y2 = pts90_sc[name]
        dx, dy = x2 - x1, y2 - y1
        print(f"    {name:25s} ({x1:6.1f}, {y1:6.1f})    ({x2:6.1f}, {y2:6.1f})    ({dx:+5.1f}, {dy:+5.1f})")

    # ── 5-6. TPS Warp ──
    print("\n[5-6] Applying TPS warp ...")
    src_pts = [pts180_sc[name] for name in KP_ORDER]
    dst_pts = [pts90_sc[name]  for name in KP_ORDER]

    warped = tps_warp(drr180, src_pts, dst_pts)
    print(f"    Warped image shape: {warped.shape}")

    # ── 7-9. Compare and save ──
    print("\n[7-9] Computing metrics and saving ...")

    # SSIM
    score_warped_vs_real = ssim_gray(warped, drr90)
    score_orig_vs_real   = ssim_gray(drr180, drr90)
    print(f"    SSIM (180deg orig  vs real 90deg): {score_orig_vs_real:.4f}")
    print(f"    SSIM (warped 180   vs real 90deg): {score_warped_vs_real:.4f}")
    print(f"    SSIM improvement: {score_warped_vs_real - score_orig_vs_real:+.4f}")

    # Mean Absolute Error (in foreground region)
    fg_mask = (drr90 > 10)  # non-background
    if fg_mask.any():
        mae_orig   = np.abs(drr180.astype(float) - drr90.astype(float))[fg_mask].mean()
        mae_warped = np.abs(warped.astype(float)  - drr90.astype(float))[fg_mask].mean()
        print(f"    MAE foreground (orig  vs real): {mae_orig:.2f}")
        print(f"    MAE foreground (warped vs real): {mae_warped:.2f}")

    # Landmark displacement error after warp
    # Warp moves 180deg landmarks to 90deg positions, so measure how close
    # the warped landmark positions are to the real 90deg positions.
    # (By construction, TPS exactly maps control points, so displacement is 0.
    #  The real question is overall image quality.)

    # ── Build comparison figure ──
    pad = 10
    # Add landmark overlays
    drr180_lm = draw_landmarks(drr180, pts180_sc, (0, 255, 0))   # green = 180
    warped_lm = draw_landmarks(warped, pts90_sc,  (0, 0, 255))   # red = target 90
    drr90_lm  = draw_landmarks(drr90,  pts90_sc,  (0, 0, 255))   # red = 90

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(out, text, (10, 25), font, 0.6, (0, 0, 0), 1)
        return out

    drr180_lm = add_label(drr180_lm, "180deg LAT (source)")
    warped_lm = add_label(warped_lm, f"Warped->90 (SSIM={score_warped_vs_real:.3f})")
    drr90_lm  = add_label(drr90_lm,  "Real 90deg LAT (target)")

    # Difference images
    diff_orig   = cv2.absdiff(drr180, drr90)
    diff_warped = cv2.absdiff(warped, drr90)
    # Enhance contrast of diff
    diff_orig_vis   = cv2.applyColorMap((diff_orig * 2).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    diff_warped_vis = cv2.applyColorMap((diff_warped * 2).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    diff_orig_vis   = add_label(diff_orig_vis, "|180-90| diff")
    diff_warped_vis = add_label(diff_warped_vis, "|warped-90| diff")

    # Row 1: source, warped, target
    row1 = np.hstack([drr180_lm, np.full((target_h, pad, 3), 128, np.uint8),
                       warped_lm, np.full((target_h, pad, 3), 128, np.uint8),
                       drr90_lm])
    # Row 2: diff before, diff after, blank
    blank = np.full_like(diff_orig_vis, 40)
    blank = add_label(blank, f"SSIM orig={score_orig_vs_real:.3f}")
    row2 = np.hstack([diff_orig_vis, np.full((target_h, pad, 3), 128, np.uint8),
                       diff_warped_vis, np.full((target_h, pad, 3), 128, np.uint8),
                       blank])
    canvas = np.vstack([row1, np.full((pad, row1.shape[1], 3), 128, np.uint8), row2])

    out_path = os.path.join(OUT_DIR, "approachA_2d_warp.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n    Saved: {out_path}")

    # Also save individual images
    cv2.imwrite(os.path.join(OUT_DIR, "approachA_warped_only.png"), warped)

    # ── Summary report ──
    report = f"""
Approach A: 2D TPS Warp - Flexion Synthesis Report
===================================================
Source: 180deg LAT DRR (Series 4, FC85 bone kernel)
Target: 90deg LAT DRR (Series 12, FC85 bone kernel)
Laterality: {LATERALITY}
Image size: {target_h} x {target_w}

Control points: {len(KP_ORDER)} anatomical landmarks + 16 border anchors
Warping method: scipy RBFInterpolator (thin_plate_spline kernel)

Results:
  SSIM (180deg orig  vs real 90deg): {score_orig_vs_real:.4f}
  SSIM (warped 180   vs real 90deg): {score_warped_vs_real:.4f}
  SSIM improvement:                  {score_warped_vs_real - score_orig_vs_real:+.4f}
"""
    if fg_mask.any():
        report += f"""  MAE foreground (orig  vs real):     {mae_orig:.2f}
  MAE foreground (warped vs real):    {mae_warped:.2f}
"""
    report += f"""
Output: {out_path}
"""
    print(report)

    report_path = os.path.join(OUT_DIR, "approachA_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"    Report saved: {report_path}")


if __name__ == "__main__":
    main()
