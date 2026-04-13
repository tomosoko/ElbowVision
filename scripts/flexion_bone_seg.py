#!/usr/bin/env python3
"""
Flexion synthesis approach 1: 3D bone segmentation + rigid rotation.

Instead of splitting the volume by PD slice (too crude), this script:
  1. Segments bone voxels using 2-stage Otsu thresholding
  2. Separates humerus from forearm using 3D connected components at the joint level
  3. Creates a smooth 3D forearm mask (bone CC + distance-based soft tissue)
  4. Rotates ONLY the forearm voxels by -90 deg around the trochlea/capitellum axis
  5. Composites with Gaussian seam blending
  6. Generates LAT DRR and compares with real 90 deg

Usage:
  cd /Users/kohei/develop/research/ElbowVision
  elbow-api/venv/bin/python scripts/flexion_bone_seg.py
"""

import sys
import os
import math

import numpy as np
import cv2
from scipy.ndimage import (
    label, affine_transform, binary_erosion, binary_dilation,
    gaussian_filter,
)
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, "elbow-train")
from elbow_synth import load_ct_volume, auto_detect_landmarks, generate_drr

# ── paths ──
CT_DIR = "data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000"
OUT_DIR = "results/flexion_synthesis"
os.makedirs(OUT_DIR, exist_ok=True)

LATERALITY = "L"
HU_MIN = 50
HU_MAX = 1000
TARGET_SIZE = 256


# ── helpers ──

def otsu_threshold(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    best_var, best_thresh = 0.0, float(bin_centers[len(bin_centers) // 2])
    cumsum_w, cumsum_mu = 0.0, 0.0
    mu_total = float((hist * bin_centers).sum())
    for i in range(len(hist)):
        cumsum_w += hist[i]
        cumsum_mu += hist[i] * bin_centers[i]
        w0 = cumsum_w / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cumsum_mu / cumsum_w
        mu1 = (mu_total - cumsum_mu) / (total - cumsum_w)
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_thresh = var, bin_centers[i]
    return best_thresh


def two_stage_otsu(volume: np.ndarray) -> float:
    flat = volume.flatten()
    thresh1 = otsu_threshold(flat)
    foreground = flat[flat > thresh1]
    if len(foreground) > 100:
        return otsu_threshold(foreground)
    return thresh1


def rotation_matrix_z(deg: float) -> np.ndarray:
    """axis2(ML) fixed, rotate PD/AP -- flexion/extension."""
    r = math.radians(deg)
    return np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r),  math.cos(r), 0],
        [0,            0,           1],
    ])


def bone_dice(img1: np.ndarray, img2: np.ndarray, threshold: int = 80) -> float:
    b1 = (img1 > threshold).astype(np.uint8)
    b2 = (img2 > threshold).astype(np.uint8)
    inter = (b1 & b2).sum()
    total = b1.sum() + b2.sum()
    return 2.0 * inter / total if total > 0 else 1.0


def separate_bones_cc(bone_mask, joint_pd_vox, pd_size):
    """Separate humerus from forearm using erosion at joint + connected components."""
    margin = max(3, int(pd_size * 0.04))
    js, je = max(0, joint_pd_vox - margin), min(pd_size, joint_pd_vox + margin)

    eroded = bone_mask.copy()
    for pd_i in range(js, je):
        eroded[pd_i] = binary_erosion(eroded[pd_i], iterations=3)

    labeled, nf = label(eroded)
    print(f"    Connected components after joint erosion: {nf}")

    def largest_label(zone):
        vals = zone[zone > 0]
        if len(vals) == 0:
            return -1
        u, c = np.unique(vals, return_counts=True)
        return u[c.argmax()]

    hl = largest_label(labeled[:joint_pd_vox])
    fl = largest_label(labeled[joint_pd_vox:])
    print(f"    Humerus label={hl}, Forearm label={fl}")

    if hl == fl or hl < 0 or fl < 0:
        print("    WARNING: CC failed, PD-split fallback")
        fm = np.zeros_like(bone_mask, dtype=bool)
        hm = np.zeros_like(bone_mask, dtype=bool)
        for i in range(pd_size):
            (fm if i > joint_pd_vox else hm)[i] = bone_mask[i]
        return hm, fm

    fm = np.zeros_like(bone_mask, dtype=bool)
    hm = np.zeros_like(bone_mask, dtype=bool)
    for pd_i in range(pd_size):
        bs = bone_mask[pd_i]
        ls = labeled[pd_i]
        fm[pd_i] = bs & (ls == fl)
        hm[pd_i] = bs & (ls == hl)
        unassigned = bs & ~fm[pd_i] & ~hm[pd_i]
        if unassigned.any():
            if pd_i > joint_pd_vox + margin:
                fm[pd_i] |= unassigned
            elif pd_i < joint_pd_vox - margin:
                hm[pd_i] |= unassigned
            else:
                fm[pd_i] |= unassigned & (ls != hl)
    return hm, fm


def build_forearm_weight_3d(vol_shape, joint_center_vox, forearm_bone_mask, blend_sigma=8.0):
    """
    Build a smooth 3D weight mask: 0 = humerus region, 1 = forearm region.
    Uses the forearm bone mask dilated + distance from joint center for blending.
    """
    pd, ap, ml = vol_shape
    jc_pd = joint_center_vox[0]

    # Start with a PD-based weight
    weight = np.zeros(vol_shape, dtype=np.float32)
    blend_half = max(6, int(pd * 0.05))

    for i in range(pd):
        if i >= jc_pd + blend_half:
            weight[i] = 1.0
        elif i > jc_pd - blend_half:
            weight[i] = (i - (jc_pd - blend_half)) / (2.0 * blend_half)

    # Smooth the weight field to avoid sharp transitions
    weight = gaussian_filter(weight, sigma=[blend_sigma, blend_sigma / 2, blend_sigma / 2])

    # Ensure forearm bone voxels always get weight=1, humerus bone always 0
    # (dilate forearm mask slightly for soft tissue around bone)
    fb_dilated = binary_dilation(forearm_bone_mask, iterations=2)
    weight[fb_dilated & (np.arange(pd)[:, None, None] > jc_pd - blend_half)] = np.maximum(
        weight[fb_dilated & (np.arange(pd)[:, None, None] > jc_pd - blend_half)], 0.8
    )

    return np.clip(weight, 0.0, 1.0)


# ── main ──

def main():
    print("=" * 60)
    print("Approach 1: 3D bone segmentation + rigid rotation")
    print("=" * 60)

    # ── Step 1: Load 180 deg volume ──
    print("\n[1] Loading 180 deg volume (Series 4) ...")
    vol_180, spacing, lat, voxel_mm = load_ct_volume(
        CT_DIR, target_size=TARGET_SIZE, laterality=LATERALITY,
        series_num=4, hu_min=HU_MIN, hu_max=HU_MAX,
    )
    pd_size, ap_size, ml_size = vol_180.shape
    print(f"    Shape: {vol_180.shape}, voxel_mm: {voxel_mm:.3f}")

    # ── Step 2: Bone segmentation ──
    print("\n[2] Bone segmentation (2-stage Otsu) ...")
    bone_thresh = two_stage_otsu(vol_180)
    bone_mask = vol_180 > bone_thresh
    print(f"    Threshold: {bone_thresh:.4f}, bone voxels: {bone_mask.sum()}")

    # ── Step 3: Detect landmarks ──
    print("\n[3] Detecting landmarks ...")
    landmarks = auto_detect_landmarks(vol_180, laterality=LATERALITY)
    jc = landmarks["joint_center"]
    jc_vox = np.array([jc[0] * pd_size, jc[1] * ap_size, jc[2] * ml_size])
    joint_pd_vox = int(jc_vox[0])
    print(f"    Joint center voxel: PD={jc_vox[0]:.1f}, AP={jc_vox[1]:.1f}, ML={jc_vox[2]:.1f}")

    # Trochlea/capitellum line for rotation axis info
    lat_epi = landmarks["lateral_epicondyle"]
    med_epi = landmarks["medial_epicondyle"]
    troch_cap_ml = np.array([
        lat_epi[2] * ml_size - med_epi[2] * ml_size,
    ])
    print(f"    Trochlea-Capitellum ML span: {abs(troch_cap_ml[0]):.1f} voxels")

    # ── Step 4: Separate humerus / forearm bones ──
    print("\n[4] Separating bones (3D connected components) ...")
    humerus_mask, forearm_mask = separate_bones_cc(bone_mask, joint_pd_vox, pd_size)
    print(f"    Humerus: {humerus_mask.sum()} voxels, Forearm: {forearm_mask.sum()} voxels")

    # ── Step 5: Build smooth 3D forearm weight mask ──
    print("\n[5] Building 3D forearm weight mask ...")
    forearm_weight = build_forearm_weight_3d(
        vol_180.shape, jc_vox, forearm_mask, blend_sigma=6.0,
    )
    print(f"    Weight range: [{forearm_weight.min():.3f}, {forearm_weight.max():.3f}]")
    print(f"    Forearm-weighted volume (>0.5): {(forearm_weight > 0.5).sum()} voxels")

    # ── Step 6: Rotate forearm by -90 deg ──
    print("\n[6] Rotating forearm by -90 deg (flexion 180->90) ...")
    R = rotation_matrix_z(-90.0)
    R_inv = R.T
    center = jc_vox.copy()
    offset = center - R_inv @ center

    # Split volume using the smooth weight
    humerus_vol = vol_180 * (1.0 - forearm_weight)
    forearm_vol = vol_180 * forearm_weight

    # Rotate forearm portion
    forearm_rotated = affine_transform(
        forearm_vol, R_inv, offset=offset, order=3, mode='constant', cval=0.0,
    )

    # Also rotate a dilated bone mask to know where bone ended up
    forearm_bone_float = forearm_mask.astype(np.float32)
    forearm_bone_rot = affine_transform(
        forearm_bone_float, R_inv, offset=offset, order=1, mode='constant', cval=0.0,
    )

    # Composite with seam smoothing
    composite = humerus_vol + forearm_rotated
    composite = np.clip(composite, 0.0, 1.0).astype(np.float32)

    # Light Gaussian smooth at the seam (joint zone only)
    seam_zone = (forearm_weight > 0.05) & (forearm_weight < 0.95)
    composite_smooth = gaussian_filter(composite, sigma=1.0)
    composite[seam_zone] = composite_smooth[seam_zone]
    composite = np.clip(composite, 0.0, 1.0).astype(np.float32)

    print(f"    Composite non-zero: {(composite > 0.01).sum()}")

    # ── Step 7: Also try +90 deg (opposite direction) in case sign is wrong ──
    print("\n[7] Testing +90 deg variant ...")
    R_pos = rotation_matrix_z(+90.0)
    R_pos_inv = R_pos.T
    offset_pos = center - R_pos_inv @ center
    forearm_rot_pos = affine_transform(
        forearm_vol, R_pos_inv, offset=offset_pos, order=3, mode='constant', cval=0.0,
    )
    composite_pos = humerus_vol + forearm_rot_pos
    composite_pos = np.clip(composite_pos, 0.0, 1.0).astype(np.float32)
    composite_pos_smooth = gaussian_filter(composite_pos, sigma=1.0)
    composite_pos[seam_zone] = composite_pos_smooth[seam_zone]
    composite_pos = np.clip(composite_pos, 0.0, 1.0).astype(np.float32)

    # ── Step 8: Generate DRRs ──
    print("\n[8] Generating LAT DRRs ...")
    drr_neg90 = generate_drr(composite, axis="LAT", voxel_mm=voxel_mm)
    drr_pos90 = generate_drr(composite_pos, axis="LAT", voxel_mm=voxel_mm)
    drr_180_lat = generate_drr(vol_180, axis="LAT", voxel_mm=voxel_mm)

    # ── Step 9: Load real 90 deg and generate DRR ──
    print("\n[9] Loading real 90 deg (Series 12) ...")
    vol_90, _, _, voxel_mm_90 = load_ct_volume(
        CT_DIR, target_size=TARGET_SIZE, laterality=LATERALITY,
        series_num=12, hu_min=HU_MIN, hu_max=HU_MAX,
    )
    drr_real_90 = generate_drr(vol_90, axis="LAT", voxel_mm=voxel_mm_90)

    # ── Step 10: Metrics ──
    print("\n[10] Computing metrics ...")
    h_ref, w_ref = drr_real_90.shape[:2]

    def to_ref(img):
        if img.shape[:2] != (h_ref, w_ref):
            return cv2.resize(img, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        return img

    drr_neg90_r = to_ref(drr_neg90)
    drr_pos90_r = to_ref(drr_pos90)
    drr_180_r = to_ref(drr_180_lat)

    metrics = {}
    for name, img in [("180_norot", drr_180_r), ("neg90", drr_neg90_r), ("pos90", drr_pos90_r)]:
        s = ssim(drr_real_90, img, data_range=255)
        d = bone_dice(drr_real_90, img)
        metrics[name] = (s, d)

    print(f"\n    {'Method':<20} {'SSIM':<10} {'Bone Dice':<10}")
    print(f"    {'=' * 40}")
    for name, (s, d) in metrics.items():
        print(f"    {name:<20} {s:<10.4f} {d:<10.4f}")

    # Pick the best rotation direction
    best_key = max(["neg90", "pos90"], key=lambda k: metrics[k][0])
    print(f"\n    Best rotation: {best_key}")
    if best_key == "neg90":
        drr_best = drr_neg90_r
        best_label = "-90 deg (flex)"
    else:
        drr_best = drr_pos90_r
        best_label = "+90 deg (flex)"

    best_ssim = metrics[best_key][0]
    best_dice = metrics[best_key][1]

    # ── Step 11: Save comparison figure ──
    print("\n[11] Saving comparison ...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 8

    def label_img(img, title, info=""):
        h, w = img.shape[:2]
        canvas = np.zeros((h + 45, w), dtype=np.uint8)
        canvas[45:, :] = img
        cv2.putText(canvas, title, (4, 20), font, 0.45, 255, 1, cv2.LINE_AA)
        if info:
            cv2.putText(canvas, info, (4, 38), font, 0.35, 200, 1, cv2.LINE_AA)
        return canvas

    panels = [
        label_img(drr_real_90, "Real 90deg (S12)", "Ground truth"),
        label_img(drr_180_r, "180deg LAT (no rot)",
                  f"SSIM={metrics['180_norot'][0]:.3f} Dice={metrics['180_norot'][1]:.3f}"),
        label_img(drr_neg90_r, "Bone-seg -90deg",
                  f"SSIM={metrics['neg90'][0]:.3f} Dice={metrics['neg90'][1]:.3f}"),
        label_img(drr_pos90_r, "Bone-seg +90deg",
                  f"SSIM={metrics['pos90'][0]:.3f} Dice={metrics['pos90'][1]:.3f}"),
    ]

    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)

    def pad_to(img, h, w):
        out = np.zeros((h, w), dtype=np.uint8)
        out[:img.shape[0], :img.shape[1]] = img
        return out

    panels = [pad_to(p, max_h, max_w) for p in panels]
    sep_v = np.zeros((max_h, pad), dtype=np.uint8)
    sep_h = np.zeros((pad, 2 * max_w + pad), dtype=np.uint8)

    top = np.hstack([panels[0], sep_v, panels[1]])
    bot = np.hstack([panels[2], sep_v, panels[3]])
    grid = np.vstack([top, sep_h, bot])

    out_path = os.path.join(OUT_DIR, "approach1_bone_seg.png")
    cv2.imwrite(out_path, grid)
    print(f"    Saved: {out_path}")

    # Individual images
    cv2.imwrite(os.path.join(OUT_DIR, "approach1_real_90.png"), drr_real_90)
    cv2.imwrite(os.path.join(OUT_DIR, "approach1_synth_best.png"), drr_best)
    cv2.imwrite(os.path.join(OUT_DIR, "approach1_synth_neg90.png"), drr_neg90)
    cv2.imwrite(os.path.join(OUT_DIR, "approach1_synth_pos90.png"), drr_pos90)

    # ── Save axial slice comparison for debugging ──
    print("\n[12] Saving axial slice debug ...")
    debug_slices = [
        int(jc_vox[0] - 20), int(jc_vox[0]), int(jc_vox[0] + 20), int(jc_vox[0] + 40)
    ]
    rows = []
    for sl_pd in debug_slices:
        if 0 <= sl_pd < pd_size:
            orig_sl = (vol_180[sl_pd] * 255).astype(np.uint8)
            comp_sl = (composite[sl_pd] * 255).astype(np.uint8)
            comp_pos_sl = (composite_pos[sl_pd] * 255).astype(np.uint8)
            # Weight overlay
            wt_sl = (forearm_weight[sl_pd] * 128).astype(np.uint8)
            row = np.hstack([orig_sl, comp_sl, comp_pos_sl, wt_sl])
            # Add PD label
            cv2.putText(row, f"PD={sl_pd}", (2, 15), font, 0.4, 255, 1)
            rows.append(row)
    if rows:
        debug_img = np.vstack(rows)
        cv2.imwrite(os.path.join(OUT_DIR, "approach1_axial_debug.png"), debug_img)
        print(f"    Saved axial debug (cols: orig, -90, +90, weight)")

    print(f"\n{'=' * 60}")
    print(f"SSIM (best): {best_ssim:.4f}  Bone Dice (best): {best_dice:.4f}")
    print(f"Best rotation direction: {best_label}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
