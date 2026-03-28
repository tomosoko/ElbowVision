#!/usr/bin/env python3
"""
CT-to-X-ray Direct Synthesis Prototype
========================================
Generate a 90-degree flexed lateral X-ray image directly from an extended (180 deg)
CT volume WITHOUT synthesizing a 3D flexed volume first.

Key idea:
  - Bone-only volumes (soft tissue removed)
  - Separate humerus and forearm at joint_center
  - Keep humerus DRR unchanged
  - Rotate forearm voxels by -90 deg around joint_center (ML axis)
  - Generate DRR from each part independently
  - Composite: humerus_drr + forearm_drr = predicted 90-deg LAT X-ray
  - Compare with real 90-deg DRR and real CR X-ray

LEFT arm, FC85 bone kernel, hu_min=50, hu_max=1000
"""

import sys
import os
import math
import time

import cv2
import numpy as np
from scipy.ndimage import affine_transform
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = "/Users/kohei/develop/Dev/vision/ElbowVision"
sys.path.insert(0, os.path.join(ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume, auto_detect_landmarks, generate_drr,
    rotation_matrix_z,
)

CT_DIR = os.path.join(ROOT, "data/raw_dicom/ct_volume",
                      "ﾃｽﾄ 008_0009900008_20260310_108Y_F_000")
OUT_DIR = os.path.join(ROOT, "results/ct_to_xray_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)

REAL_XRAY_LAT = os.path.join(ROOT, "data/real_xray/images/cr_008_3_52kVp.png")
REAL_XRAY_LAT2 = os.path.join(ROOT, "data/real_xray/images/008_LAT.png")

# Series 4=180deg, 8=135deg, 12=90deg
SERIES = [(4, 180.0), (8, 135.0), (12, 90.0)]
TARGET_SIZE = 128
HU_MIN, HU_MAX = 50, 1000
BONE_THRESHOLD_FRAC = 0.15  # fraction of max intensity to consider "bone"


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load all 3 volumes
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 1: Loading 3 CT volumes (180/135/90 deg)")
print("=" * 70)

volumes = {}
for sn, angle in SERIES:
    print(f"\n--- Series {sn}, flexion={angle} deg ---")
    vol, spacing, lat, vox_mm = load_ct_volume(
        CT_DIR, laterality='L', series_num=sn,
        hu_min=HU_MIN, hu_max=HU_MAX, target_size=TARGET_SIZE)
    lm = auto_detect_landmarks(vol, laterality=lat)
    volumes[angle] = dict(vol=vol, lm=lm, voxel_mm=vox_mm, laterality=lat)
    print(f"  Volume shape: {vol.shape}, voxel_mm: {vox_mm:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Generate ground-truth LAT DRRs from each volume
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Generating ground-truth LAT DRRs")
print("=" * 70)

gt_drrs = {}
for angle in [180.0, 135.0, 90.0]:
    v = volumes[angle]
    drr = generate_drr(v['vol'], axis="LAT", sid_mm=1000.0, voxel_mm=v['voxel_mm'])
    gt_drrs[angle] = drr
    outpath = os.path.join(OUT_DIR, f"gt_drr_lat_{int(angle)}deg.png")
    cv2.imwrite(outpath, drr)
    print(f"  {int(angle)} deg LAT DRR: {drr.shape}, saved to {outpath}")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Bone segmentation and humerus/forearm separation (180 deg vol)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: Bone segmentation & humerus/forearm separation")
print("=" * 70)

vol_180 = volumes[180.0]['vol']
lm_180 = volumes[180.0]['lm']
vox_mm_180 = volumes[180.0]['voxel_mm']
pd_size, ap_size, ml_size = vol_180.shape

# Bone threshold: use Otsu-like approach on the volume
# Since vol is already HU-windowed (50-1000) and normalized 0-1,
# bone should be the higher intensity values
bone_vals = vol_180[vol_180 > 0.01]
if len(bone_vals) > 0:
    # Use percentile-based threshold for bone
    bone_thresh = float(np.percentile(bone_vals, 50))
else:
    bone_thresh = 0.3
print(f"  Bone threshold: {bone_thresh:.3f}")

bone_mask = vol_180 > bone_thresh

# Joint center in voxel coordinates
jc = lm_180['joint_center']
jc_vox = np.array([jc[0] * pd_size, jc[1] * ap_size, jc[2] * ml_size])
joint_pd_idx = int(jc_vox[0])
print(f"  Joint center (voxel): PD={jc_vox[0]:.1f}, AP={jc_vox[1]:.1f}, ML={jc_vox[2]:.1f}")
print(f"  Joint PD index: {joint_pd_idx} / {pd_size}")

# Smooth transition zone around joint
blend_half = max(3, int(pd_size * 0.04))
print(f"  Blend half-width: {blend_half} voxels")

# Create humerus mask: PD < joint_center (with blend zone)
humerus_weight = np.zeros(pd_size, dtype=np.float32)
for i in range(pd_size):
    if i < joint_pd_idx - blend_half:
        humerus_weight[i] = 1.0
    elif i > joint_pd_idx + blend_half:
        humerus_weight[i] = 0.0
    else:
        humerus_weight[i] = 0.5 * (1.0 - (i - joint_pd_idx) / blend_half)

forearm_weight = 1.0 - humerus_weight

# Create bone-only sub-volumes
vol_bone = vol_180 * bone_mask.astype(np.float32)
vol_humerus = vol_bone * humerus_weight[:, None, None]
vol_forearm = vol_bone * forearm_weight[:, None, None]

n_hum_vox = int((vol_humerus > 0.01).sum())
n_fore_vox = int((vol_forearm > 0.01).sum())
print(f"  Humerus bone voxels: {n_hum_vox}")
print(f"  Forearm bone voxels: {n_fore_vox}")


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Generate humerus DRR (unchanged)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Generating humerus DRR (unchanged)")
print("=" * 70)

humerus_drr = generate_drr(vol_humerus, axis="LAT", sid_mm=1000.0, voxel_mm=vox_mm_180)
cv2.imwrite(os.path.join(OUT_DIR, "humerus_drr_180deg.png"), humerus_drr)
print(f"  Humerus DRR shape: {humerus_drr.shape}")


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Rotate forearm by -90 deg around joint center and generate DRR
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Rotating forearm -90 deg around joint center")
print("=" * 70)

# Rotation: 180 deg -> 90 deg means flexion_delta = 90 - 180 = -90 deg
# rotation_matrix_z rotates around ML axis (axis2) in the PD-AP plane
flexion_delta = -90.0
R_flex = rotation_matrix_z(flexion_delta)
R_inv = R_flex.T  # For affine_transform (inverse mapping)

# Offset for rotation around joint center
offset = jc_vox - R_inv @ jc_vox

print(f"  Flexion delta: {flexion_delta} deg")
print(f"  Rotation center: {jc_vox}")
print(f"  Offset: {offset}")

vol_forearm_rotated = affine_transform(
    vol_forearm, R_inv, offset=offset,
    order=3, mode='constant', cval=0.0
)

cv2.imwrite(os.path.join(OUT_DIR, "forearm_rotated_volume_mid.png"),
            (vol_forearm_rotated[:, :, ml_size // 2] * 255).astype(np.uint8))

forearm_drr = generate_drr(vol_forearm_rotated, axis="LAT", sid_mm=1000.0, voxel_mm=vox_mm_180)
cv2.imwrite(os.path.join(OUT_DIR, "forearm_drr_rotated90.png"), forearm_drr)
print(f"  Forearm rotated DRR shape: {forearm_drr.shape}")


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Composite humerus + forearm DRRs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 6: Compositing humerus + forearm DRRs")
print("=" * 70)

# Both DRRs are uint8, bone=bright. Additive composite with saturation.
h_f = humerus_drr.astype(np.float32)
f_f = forearm_drr.astype(np.float32)

# Method 1: Max composite (keeps strongest bone signal)
composite_max = np.maximum(h_f, f_f).astype(np.uint8)

# Method 2: Additive with saturation (more realistic for overlapping bones)
composite_add = np.clip(h_f + f_f, 0, 255).astype(np.uint8)

# Method 3: Screen blend (like layering two X-ray films)
# screen(a,b) = 1 - (1-a/255)*(1-b/255)  -- gives brighter result for overlap
a_n = h_f / 255.0
b_n = f_f / 255.0
composite_screen = ((1.0 - (1.0 - a_n) * (1.0 - b_n)) * 255).astype(np.uint8)

# Apply CLAHE to composites for consistent appearance
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
composite_max_clahe = clahe.apply(composite_max)
composite_add_clahe = clahe.apply(composite_add)
composite_screen_clahe = clahe.apply(composite_screen)

cv2.imwrite(os.path.join(OUT_DIR, "composite_max.png"), composite_max_clahe)
cv2.imwrite(os.path.join(OUT_DIR, "composite_add.png"), composite_add_clahe)
cv2.imwrite(os.path.join(OUT_DIR, "composite_screen.png"), composite_screen_clahe)
print("  Saved 3 composite methods: max, additive, screen")


# ═══════════════════════════════════════════════════════════════════════
# Step 7: Also generate intermediate angle (135 deg) for validation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 7: Generating 135-deg predicted DRR for validation")
print("=" * 70)

R_135 = rotation_matrix_z(-45.0)  # 180 -> 135 = -45 deg
R_135_inv = R_135.T
offset_135 = jc_vox - R_135_inv @ jc_vox

vol_forearm_135 = affine_transform(
    vol_forearm, R_135_inv, offset=offset_135,
    order=3, mode='constant', cval=0.0
)

forearm_drr_135 = generate_drr(vol_forearm_135, axis="LAT", sid_mm=1000.0, voxel_mm=vox_mm_180)
composite_135 = np.clip(h_f + forearm_drr_135.astype(np.float32), 0, 255).astype(np.uint8)
composite_135_clahe = clahe.apply(composite_135)
cv2.imwrite(os.path.join(OUT_DIR, "composite_predicted_135deg.png"), composite_135_clahe)
print("  Saved predicted 135-deg composite")


# ═══════════════════════════════════════════════════════════════════════
# Step 8: Compute SSIM and Bone Dice metrics
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 8: Computing metrics")
print("=" * 70)

gt_90 = gt_drrs[90.0]
gt_135 = gt_drrs[135.0]

# Resize composites to match GT if needed
def resize_to_match(img, target):
    if img.shape != target.shape:
        return cv2.resize(img, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return img

metrics = {}

# 90-deg comparison
for name, pred in [("max", composite_max_clahe),
                   ("add", composite_add_clahe),
                   ("screen", composite_screen_clahe)]:
    pred_r = resize_to_match(pred, gt_90)
    s = ssim(gt_90, pred_r, data_range=255)

    # Bone Dice: binarize both at Otsu threshold
    _, gt_bin = cv2.threshold(gt_90, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred_bin = cv2.threshold(pred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intersection = np.logical_and(gt_bin > 0, pred_bin > 0).sum()
    dice = 2.0 * intersection / (np.sum(gt_bin > 0) + np.sum(pred_bin > 0) + 1e-8)

    metrics[f"90deg_{name}"] = {"ssim": s, "bone_dice": dice}
    print(f"  90-deg [{name:6s}]: SSIM={s:.4f}, Bone Dice={dice:.4f}")

# 135-deg comparison
pred_135_r = resize_to_match(composite_135_clahe, gt_135)
s_135 = ssim(gt_135, pred_135_r, data_range=255)
_, gt_135_bin = cv2.threshold(gt_135, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, pred_135_bin = cv2.threshold(pred_135_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inter_135 = np.logical_and(gt_135_bin > 0, pred_135_bin > 0).sum()
dice_135 = 2.0 * inter_135 / (np.sum(gt_135_bin > 0) + np.sum(pred_135_bin > 0) + 1e-8)
metrics["135deg_add"] = {"ssim": s_135, "bone_dice": dice_135}
print(f"  135-deg [add   ]: SSIM={s_135:.4f}, Bone Dice={dice_135:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 9: Load real CR X-ray for visual comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 9: Loading real CR X-ray for comparison")
print("=" * 70)

real_xray = None
for rpath in [REAL_XRAY_LAT, REAL_XRAY_LAT2]:
    if os.path.exists(rpath):
        real_xray = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
        print(f"  Loaded: {rpath} ({real_xray.shape})")
        break

if real_xray is not None:
    real_resized = cv2.resize(real_xray, (gt_90.shape[1], gt_90.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
    s_real = ssim(gt_90, real_resized, data_range=255)
    print(f"  Real CR vs GT-90 SSIM: {s_real:.4f}")

    # Compare prediction vs real
    pred_best = resize_to_match(composite_add_clahe, real_resized)
    s_pred_real = ssim(real_resized, pred_best, data_range=255)
    print(f"  Predicted vs Real CR SSIM: {s_pred_real:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 10: Generate summary figure
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 10: Generating summary figure")
print("=" * 70)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("CT-to-X-ray Direct Synthesis: Bone-Only Split-Rotate-Reproject",
             fontsize=14, fontweight='bold')

# Row 0: Ground truth DRRs
axes[0, 0].imshow(gt_drrs[180.0], cmap='gray')
axes[0, 0].set_title("GT: 180-deg LAT DRR\n(extended)")
axes[0, 1].imshow(gt_drrs[135.0], cmap='gray')
axes[0, 1].set_title("GT: 135-deg LAT DRR")
axes[0, 2].imshow(gt_drrs[90.0], cmap='gray')
axes[0, 2].set_title("GT: 90-deg LAT DRR\n(target)")
if real_xray is not None:
    axes[0, 3].imshow(real_xray, cmap='gray')
    axes[0, 3].set_title("Real CR X-ray\n(52kVp)")
else:
    axes[0, 3].set_visible(False)

# Row 1: Component DRRs
axes[1, 0].imshow(humerus_drr, cmap='gray')
axes[1, 0].set_title("Humerus DRR\n(from 180-deg, fixed)")
axes[1, 1].imshow(forearm_drr, cmap='gray')
axes[1, 1].set_title("Forearm DRR\n(rotated -90 deg)")

# Mid-slice visualization of bone separation
mid_sl = ml_size // 2
vis_sep = np.zeros((pd_size, ap_size, 3), dtype=np.uint8)
vis_sep[:, :, 2] = (vol_humerus[:, :, mid_sl] * 255).astype(np.uint8)  # blue=humerus
vis_sep[:, :, 1] = (vol_forearm[:, :, mid_sl] * 255).astype(np.uint8)  # green=forearm
axes[1, 2].imshow(vis_sep)
axes[1, 2].set_title("Bone separation\n(blue=humerus, green=forearm)")

# Rotated forearm mid-slice
vis_rot = np.zeros((pd_size, ap_size, 3), dtype=np.uint8)
vis_rot[:, :, 2] = (vol_humerus[:, :, mid_sl] * 255).astype(np.uint8)
vis_rot[:, :, 1] = (vol_forearm_rotated[:, :, mid_sl] * 255).astype(np.uint8)
axes[1, 3].imshow(vis_rot)
axes[1, 3].set_title("After rotation\n(blue=humerus, green=forearm@90)")

# Row 2: Composite methods and comparison
axes[2, 0].imshow(composite_max_clahe, cmap='gray')
m = metrics["90deg_max"]
axes[2, 0].set_title(f"Composite: Max\nSSIM={m['ssim']:.3f} Dice={m['bone_dice']:.3f}")

axes[2, 1].imshow(composite_add_clahe, cmap='gray')
m = metrics["90deg_add"]
axes[2, 1].set_title(f"Composite: Additive\nSSIM={m['ssim']:.3f} Dice={m['bone_dice']:.3f}")

axes[2, 2].imshow(composite_screen_clahe, cmap='gray')
m = metrics["90deg_screen"]
axes[2, 2].set_title(f"Composite: Screen\nSSIM={m['ssim']:.3f} Dice={m['bone_dice']:.3f}")

# Difference map: predicted vs GT
diff_map = cv2.absdiff(resize_to_match(composite_add_clahe, gt_90), gt_90)
axes[2, 3].imshow(diff_map, cmap='hot')
axes[2, 3].set_title("Difference: Additive vs GT-90\n(hot = error)")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "summary_ct_to_xray_direct.png"), dpi=150)
print(f"  Saved summary figure")

# Also save a 135-deg comparison figure
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("135-deg Validation: Predicted vs Ground Truth", fontsize=13)
axes2[0].imshow(gt_drrs[135.0], cmap='gray')
axes2[0].set_title("GT: 135-deg LAT DRR")
axes2[1].imshow(composite_135_clahe, cmap='gray')
m135 = metrics["135deg_add"]
axes2[1].set_title(f"Predicted 135-deg\nSSIM={m135['ssim']:.3f} Dice={m135['bone_dice']:.3f}")
diff_135 = cv2.absdiff(resize_to_match(composite_135_clahe, gt_135), gt_135)
axes2[2].imshow(diff_135, cmap='hot')
axes2[2].set_title("Difference (hot = error)")
for ax in axes2:
    ax.axis('off')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "validation_135deg.png"), dpi=150)
print(f"  Saved 135-deg validation figure")


# ═══════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\nOutput directory: {OUT_DIR}")
print(f"\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: SSIM={v['ssim']:.4f}, Bone Dice={v['bone_dice']:.4f}")

if real_xray is not None:
    print(f"\n  Real CR vs GT-90: SSIM={s_real:.4f}")
    print(f"  Predicted vs Real CR: SSIM={s_pred_real:.4f}")

print(f"\nApproach: Bone-only split-rotate-reproject")
print(f"  1. Bone threshold at {bone_thresh:.3f} (removes soft tissue)")
print(f"  2. Split at joint_center PD={joint_pd_idx}")
print(f"  3. Humerus: fixed, Forearm: rotated {flexion_delta} deg")
print(f"  4. Independent DRR projection + 2D composite")
print(f"\nFiles saved:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
