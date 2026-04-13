#!/usr/bin/env python3
"""
CT-to-X-ray Improved Synthesis: Data-Driven Rotation Axis
==========================================================
Improves upon ct_to_xray_direct.py by computing the actual elbow flexion
axis from 3-volume landmark motion, rather than assuming a simple ML axis.

Key improvements:
  1. Load all 3 volumes (Series 4/8/12) and detect landmarks
  2. Align humerus across volumes (rigid registration)
  3. Compute the true rotation axis from forearm landmark motion:
     - Forearm landmarks trace circular arcs around the flexion axis
     - Fit plane to 3 forearm_shaft positions -> plane normal = rotation axis
     - Cross-validate with radial_head and olecranon trajectories
  4. Rotate forearm around the TRUE axis (not simple ML)
  5. Generate composite LAT DRR
  6. Compare with real 90-deg DRR (SSIM/Dice) vs baseline ML-axis

LEFT arm, FC85 bone kernel, hu_min=50, hu_max=1000
"""

import sys
import os
import math
import time

import cv2
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = "/Users/kohei/develop/research/ElbowVision"
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

FOREARM_NAMES = ["forearm_shaft", "radial_head", "olecranon"]
HUMERUS_NAMES = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                 "joint_center"]


# ═══════════════════════════════════════════════════════════════════════
# Helper functions from Approach B
# ═══════════════════════════════════════════════════════════════════════

def to_mm(lm_dict, voxel_mm, vol_shape):
    """Convert normalised landmark dict to mm-scale arrays."""
    out = {}
    pd_s, ap_s, ml_s = vol_shape
    for name, (pd, ap, ml) in lm_dict.items():
        out[name] = np.array([pd * pd_s * voxel_mm,
                              ap * ap_s * voxel_mm,
                              ml * ml_s * voxel_mm])
    return out


def to_voxel(lm_dict, voxel_mm, vol_shape):
    """Convert normalised landmark dict to voxel coordinates."""
    out = {}
    pd_s, ap_s, ml_s = vol_shape
    for name, (pd, ap, ml) in lm_dict.items():
        out[name] = np.array([pd * pd_s, ap * ap_s, ml * ml_s])
    return out


def build_humerus_frame(lm_mm):
    """
    Build humerus-fixed coordinate system.
    Origin = joint_center
    Y-axis = humerus_shaft -> joint_center (along humerus, distal direction)
    Z-axis = medial -> lateral epicondyle, orthogonalized
    X-axis = Y cross Z (roughly anterior)
    """
    jc = lm_mm["joint_center"]
    hs = lm_mm["humerus_shaft"]
    y_ax = jc - hs
    y_ax = y_ax / (np.linalg.norm(y_ax) + 1e-12)

    le = lm_mm["lateral_epicondyle"]
    me = lm_mm["medial_epicondyle"]
    ml_vec = le - me
    z_ax = ml_vec - np.dot(ml_vec, y_ax) * y_ax
    z_ax = z_ax / (np.linalg.norm(z_ax) + 1e-12)

    x_ax = np.cross(y_ax, z_ax)
    x_ax = x_ax / (np.linalg.norm(x_ax) + 1e-12)

    R = np.stack([x_ax, y_ax, z_ax], axis=0)
    return jc, R


def transform_to_frame(lm_mm, origin, R):
    """Transform landmarks into frame defined by origin+R."""
    return {name: R @ (pos - origin) for name, pos in lm_mm.items()}


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
# Step 2: Procrustes alignment of humerus landmarks across volumes
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: Procrustes alignment of humerus across volumes")
print("=" * 70)

obs_angles = np.array([180.0, 135.0, 90.0])

# Convert all landmarks to mm (in each volume's own coordinate system)
all_lm_mm = {}
all_lm_vox = {}
for angle in obs_angles:
    v = volumes[angle]
    all_lm_mm[angle] = to_mm(v['lm'], v['voxel_mm'], v['vol'].shape)
    all_lm_vox[angle] = to_voxel(v['lm'], v['voxel_mm'], v['vol'].shape)


def procrustes_align(source_pts, target_pts):
    """
    Compute rigid transform (R, t) that maps source -> target using SVD.
    Returns R, t such that target ~= R @ source + t
    """
    src_c = source_pts.mean(axis=0)
    tgt_c = target_pts.mean(axis=0)
    src_centered = source_pts - src_c
    tgt_centered = target_pts - tgt_c
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c
    return R, t


# Use 180-deg as reference. Align 135-deg and 90-deg humerus landmarks to 180-deg.
ref_angle = 180.0
ref_hum_pts = np.array([all_lm_mm[ref_angle][n] for n in HUMERUS_NAMES])

aligned_lm_mm = {ref_angle: all_lm_mm[ref_angle].copy()}
procrustes_transforms = {ref_angle: (np.eye(3), np.zeros(3))}

for angle in [135.0, 90.0]:
    src_hum_pts = np.array([all_lm_mm[angle][n] for n in HUMERUS_NAMES])
    R_proc, t_proc = procrustes_align(src_hum_pts, ref_hum_pts)
    procrustes_transforms[angle] = (R_proc, t_proc)

    # Transform ALL landmarks of this volume into the 180-deg reference frame
    aligned = {}
    for name, pos in all_lm_mm[angle].items():
        aligned[name] = R_proc @ pos + t_proc
    aligned_lm_mm[angle] = aligned

    # Report alignment quality
    aligned_hum = np.array([aligned[n] for n in HUMERUS_NAMES])
    residuals = np.linalg.norm(aligned_hum - ref_hum_pts, axis=1)
    print(f"\n  Procrustes alignment {angle:.0f} -> {ref_angle:.0f} deg:")
    for i, n in enumerate(HUMERUS_NAMES):
        print(f"    {n:25s}: residual = {residuals[i]:.3f} mm")
    print(f"    Mean residual: {residuals.mean():.3f} mm")

# Print aligned forearm landmarks
print(f"\n  Aligned forearm landmarks (in 180-deg reference frame, mm):")
for name in FOREARM_NAMES:
    print(f"  {name}:")
    for angle in obs_angles:
        pos = aligned_lm_mm[angle][name]
        print(f"    {angle:5.0f} deg: ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Compute TRUE rotation axis from aligned forearm landmark motion
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: Computing TRUE rotation axis from aligned landmark motion")
print("=" * 70)

# Now all landmarks are in the 180-deg mm coordinate system.
# The forearm landmarks should trace circular arcs around the flexion axis.

# Collect forearm landmark positions
forearm_trajectories = {}
for name in FOREARM_NAMES:
    forearm_trajectories[name] = np.array([aligned_lm_mm[a][name] for a in obs_angles])

# Joint center in 180-deg frame (average of aligned joint_centers for robustness)
jc_aligned = np.mean([aligned_lm_mm[a]["joint_center"] for a in obs_angles], axis=0)
print(f"  Joint center (aligned average): ({jc_aligned[0]:.2f}, {jc_aligned[1]:.2f}, {jc_aligned[2]:.2f}) mm")


def compute_rotation_axis_from_arc(pts):
    """
    Given 3 points on a circular arc, compute the rotation axis (plane normal)
    and the center of the circle.
    """
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-10:
        print("  WARNING: Points are collinear, cannot determine rotation plane")
        return np.array([0, 0, 1]), pts.mean(axis=0)
    normal = normal / n_len

    # Circumcenter
    e1 = v1 / np.linalg.norm(v1)
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    pts_2d = np.zeros((3, 2))
    for i in range(3):
        d = pts[i] - pts[0]
        pts_2d[i, 0] = np.dot(d, e1)
        pts_2d[i, 1] = np.dot(d, e2)

    ax, ay = pts_2d[0]
    bx, by = pts_2d[1]
    cx, cy = pts_2d[2]
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        center_3d = pts.mean(axis=0)
    else:
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / D
        center_3d = pts[0] + ux * e1 + uy * e2

    return normal, center_3d


# Method A: Compute axis from each forearm landmark independently
axes_from_landmarks = {}
centers_from_landmarks = {}
for name in FOREARM_NAMES:
    pts = forearm_trajectories[name]
    axis, center = compute_rotation_axis_from_arc(pts)
    # Radius as quality measure (larger = more reliable axis estimate)
    radii = [np.linalg.norm(pts[i] - center) for i in range(3)]
    avg_radius = np.mean(radii)
    axes_from_landmarks[name] = axis
    centers_from_landmarks[name] = center

    print(f"\n  {name}:")
    print(f"    Axis (mm-space): [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print(f"    Circle center:   [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"    Avg radius: {avg_radius:.2f} mm")

# Method B: SVD on all forearm landmark displacements jointly
# Stack all forearm displacement vectors relative to joint center
# and find the axis that minimises out-of-plane motion
print(f"\n  Method B: SVD-based axis from all forearm landmarks jointly")
all_forearm_pts = []
for name in FOREARM_NAMES:
    pts = forearm_trajectories[name]
    for pt in pts:
        all_forearm_pts.append(pt - jc_aligned)  # relative to joint center
all_forearm_pts = np.array(all_forearm_pts)

# The rotation axis is the direction of LEAST variance in the motion
# (all points move in a plane perpendicular to the rotation axis)
# Actually we want the normal to the best-fit plane of all displacements
U, S, Vt = np.linalg.svd(all_forearm_pts - all_forearm_pts.mean(axis=0))
# The last right singular vector (smallest singular value) = plane normal = rotation axis
svd_axis = Vt[-1]
print(f"    SVD axis (mm-space): [{svd_axis[0]:.4f}, {svd_axis[1]:.4f}, {svd_axis[2]:.4f}]")
print(f"    Singular values: {S}")

# Method C: Cross-product of forearm displacement vectors
# (forearm_180 - jc) x (forearm_90 - jc) gives axis
fs_180 = aligned_lm_mm[180.0]["forearm_shaft"] - jc_aligned
fs_90 = aligned_lm_mm[90.0]["forearm_shaft"] - jc_aligned
cross_axis = np.cross(fs_180, fs_90)
cross_axis = cross_axis / (np.linalg.norm(cross_axis) + 1e-12)
print(f"\n  Method C: Cross-product axis (forearm_shaft)")
print(f"    Cross axis (mm-space): [{cross_axis[0]:.4f}, {cross_axis[1]:.4f}, {cross_axis[2]:.4f}]")

# Weight the axes by reliability:
# - forearm_shaft has the largest lever arm (most reliable plane normal)
# - SVD uses all data (robust to individual noise)
# - Cross-product is a sanity check
# Use radius-weighted average of per-landmark axes + SVD
all_radii = []
for name in FOREARM_NAMES:
    pts = forearm_trajectories[name]
    center = centers_from_landmarks[name]
    r = np.mean([np.linalg.norm(pts[i] - center) for i in range(3)])
    all_radii.append(r)
all_radii = np.array(all_radii)

# Ensure consistent axis orientation before averaging
ref_dir = axes_from_landmarks["forearm_shaft"]
for name in FOREARM_NAMES:
    if np.dot(axes_from_landmarks[name], ref_dir) < 0:
        axes_from_landmarks[name] = -axes_from_landmarks[name]
if np.dot(svd_axis, ref_dir) < 0:
    svd_axis = -svd_axis
if np.dot(cross_axis, ref_dir) < 0:
    cross_axis = -cross_axis

# Weighted combination: per-landmark (weighted by radius) + SVD + cross-product
per_lm_axes = np.array([axes_from_landmarks[n] for n in FOREARM_NAMES])
radius_weights = all_radii / all_radii.sum()
per_lm_avg = np.sum(per_lm_axes * radius_weights[:, None], axis=0)

# Final axis: 40% per-landmark weighted, 40% SVD, 20% cross-product
final_axis_mm = 0.4 * per_lm_avg + 0.4 * svd_axis + 0.2 * cross_axis
final_axis_mm = final_axis_mm / np.linalg.norm(final_axis_mm)

print(f"\n  FINAL rotation axis (mm-space): [{final_axis_mm[0]:.4f}, {final_axis_mm[1]:.4f}, {final_axis_mm[2]:.4f}]")

# Compare with simple ML axis in this volume space
# In the 180-deg volume, the ML axis is along axis2 = [0, 0, 1] in (PD, AP, ML) voxel coords
# In mm-space this is [0, 0, vox_mm] -> normalized [0, 0, 1]
simple_ml_mm = np.array([0.0, 0.0, 1.0])
tilt_angle = np.rad2deg(np.arccos(np.clip(abs(np.dot(final_axis_mm, simple_ml_mm)), 0, 1)))
print(f"  Tilt from simple ML axis: {tilt_angle:.2f} degrees")

# Also compute in humerus-fixed frame for reporting
origin_180_hf, R_180_hf = build_humerus_frame(all_lm_mm[180.0])
avg_axis_hf = R_180_hf @ final_axis_mm
avg_axis_hf = avg_axis_hf / np.linalg.norm(avg_axis_hf)
print(f"  Axis in humerus-fixed frame: [{avg_axis_hf[0]:.4f}, {avg_axis_hf[1]:.4f}, {avg_axis_hf[2]:.4f}]")


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Set up rotation in 180-deg volume voxel space
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Setting up rotation in 180-deg volume voxel space")
print("=" * 70)

vox_mm_180 = volumes[180.0]['voxel_mm']
vol_180 = volumes[180.0]['vol']
pd_size, ap_size, ml_size = vol_180.shape

# Rotation axis in voxel space (same as mm since isotropic after resize to 128)
axis_vox = final_axis_mm.copy()
axis_vox = axis_vox / np.linalg.norm(axis_vox)
print(f"  Rotation axis (voxel): [{axis_vox[0]:.4f}, {axis_vox[1]:.4f}, {axis_vox[2]:.4f}]")

# Joint center in voxel coords
jc_norm = volumes[180.0]['lm']['joint_center']
jc_vox = np.array([jc_norm[0] * pd_size, jc_norm[1] * ap_size, jc_norm[2] * ml_size])
print(f"  Joint center (voxel): [{jc_vox[0]:.1f}, {jc_vox[1]:.1f}, {jc_vox[2]:.1f}]")

# Use joint_center as rotation pivot
pivot_vox = jc_vox.copy()


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Bone segmentation and humerus/forearm separation (same as baseline)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Bone segmentation & humerus/forearm separation")
print("=" * 70)

bone_vals = vol_180[vol_180 > 0.01]
if len(bone_vals) > 0:
    bone_thresh = float(np.percentile(bone_vals, 50))
else:
    bone_thresh = 0.3
print(f"  Bone threshold: {bone_thresh:.3f}")

bone_mask = vol_180 > bone_thresh

joint_pd_idx = int(jc_vox[0])
print(f"  Joint center PD index: {joint_pd_idx} / {pd_size}")

# Smooth transition zone
blend_half = max(3, int(pd_size * 0.04))
print(f"  Blend half-width: {blend_half} voxels")

humerus_weight = np.zeros(pd_size, dtype=np.float32)
for i in range(pd_size):
    if i < joint_pd_idx - blend_half:
        humerus_weight[i] = 1.0
    elif i > joint_pd_idx + blend_half:
        humerus_weight[i] = 0.0
    else:
        humerus_weight[i] = 0.5 * (1.0 - (i - joint_pd_idx) / blend_half)

forearm_weight = 1.0 - humerus_weight

vol_bone = vol_180 * bone_mask.astype(np.float32)
vol_humerus = vol_bone * humerus_weight[:, None, None]
vol_forearm = vol_bone * forearm_weight[:, None, None]

n_hum_vox = int((vol_humerus > 0.01).sum())
n_fore_vox = int((vol_forearm > 0.01).sum())
print(f"  Humerus bone voxels: {n_hum_vox}")
print(f"  Forearm bone voxels: {n_fore_vox}")


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Rotate forearm using the TRUE axis (improved)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 6: Rotating forearm around TRUE axis (-90 deg)")
print("=" * 70)


def build_rotation_around_axis(axis, angle_deg, center):
    """
    Build a 3x3 rotation matrix for rotation around an arbitrary axis
    passing through a given center point.

    Uses scipy Rotation for the rotation part, then computes the
    affine_transform offset for the center.

    Returns (R_inv, offset) suitable for scipy.ndimage.affine_transform.
    """
    rot = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis)
    R = rot.as_matrix()
    R_inv = R.T  # affine_transform uses inverse mapping
    offset = center - R_inv @ center
    return R_inv, offset


# Improved: rotate around true axis
flexion_delta = -90.0
R_inv_improved, offset_improved = build_rotation_around_axis(
    axis_vox, flexion_delta, pivot_vox)

print(f"  Flexion delta: {flexion_delta} deg")
print(f"  Rotation axis (voxel): {axis_vox}")
print(f"  Rotation center (voxel): {pivot_vox}")
print(f"  R_inv:\n{R_inv_improved}")
print(f"  Offset: {offset_improved}")

vol_forearm_improved = affine_transform(
    vol_forearm, R_inv_improved, offset=offset_improved,
    order=3, mode='constant', cval=0.0
)

# Baseline: rotate around simple ML axis (for comparison)
print("\n  Also computing baseline (simple ML axis) for comparison...")
R_baseline = rotation_matrix_z(flexion_delta)
R_inv_baseline = R_baseline.T
offset_baseline = jc_vox - R_inv_baseline @ jc_vox

vol_forearm_baseline = affine_transform(
    vol_forearm, R_inv_baseline, offset=offset_baseline,
    order=3, mode='constant', cval=0.0
)


# ═══════════════════════════════════════════════════════════════════════
# Step 7: Generate DRRs and composites
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 7: Generating DRRs and composites")
print("=" * 70)

# Ground truth DRRs
gt_drrs = {}
for angle in [180.0, 135.0, 90.0]:
    v = volumes[angle]
    drr = generate_drr(v['vol'], axis="LAT", sid_mm=1000.0, voxel_mm=v['voxel_mm'])
    gt_drrs[angle] = drr

# Humerus DRR (same for both methods)
humerus_drr = generate_drr(vol_humerus, axis="LAT", sid_mm=1000.0, voxel_mm=vox_mm_180)

# Forearm DRRs
forearm_drr_improved = generate_drr(vol_forearm_improved, axis="LAT",
                                     sid_mm=1000.0, voxel_mm=vox_mm_180)
forearm_drr_baseline = generate_drr(vol_forearm_baseline, axis="LAT",
                                     sid_mm=1000.0, voxel_mm=vox_mm_180)

# Composites (additive with saturation, best from ct_to_xray_direct.py)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
h_f = humerus_drr.astype(np.float32)

composite_improved = np.clip(h_f + forearm_drr_improved.astype(np.float32), 0, 255).astype(np.uint8)
composite_improved_clahe = clahe.apply(composite_improved)

composite_baseline = np.clip(h_f + forearm_drr_baseline.astype(np.float32), 0, 255).astype(np.uint8)
composite_baseline_clahe = clahe.apply(composite_baseline)

# Save intermediate outputs
cv2.imwrite(os.path.join(OUT_DIR, "improved_forearm_drr.png"), forearm_drr_improved)
cv2.imwrite(os.path.join(OUT_DIR, "improved_composite.png"), composite_improved_clahe)
print("  Saved DRRs and composites")


# ═══════════════════════════════════════════════════════════════════════
# Step 8: Also generate 135-deg predictions for validation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 8: Generating 135-deg predictions for validation")
print("=" * 70)

# Improved 135-deg
R_inv_135_imp, offset_135_imp = build_rotation_around_axis(
    axis_vox, -45.0, pivot_vox)
vol_forearm_135_imp = affine_transform(
    vol_forearm, R_inv_135_imp, offset=offset_135_imp,
    order=3, mode='constant', cval=0.0
)
forearm_drr_135_imp = generate_drr(vol_forearm_135_imp, axis="LAT",
                                    sid_mm=1000.0, voxel_mm=vox_mm_180)
composite_135_imp = np.clip(h_f + forearm_drr_135_imp.astype(np.float32), 0, 255).astype(np.uint8)
composite_135_imp_clahe = clahe.apply(composite_135_imp)

# Baseline 135-deg
R_135_base = rotation_matrix_z(-45.0)
R_135_base_inv = R_135_base.T
offset_135_base = jc_vox - R_135_base_inv @ jc_vox
vol_forearm_135_base = affine_transform(
    vol_forearm, R_135_base_inv, offset=offset_135_base,
    order=3, mode='constant', cval=0.0
)
forearm_drr_135_base = generate_drr(vol_forearm_135_base, axis="LAT",
                                     sid_mm=1000.0, voxel_mm=vox_mm_180)
composite_135_base = np.clip(h_f + forearm_drr_135_base.astype(np.float32), 0, 255).astype(np.uint8)
composite_135_base_clahe = clahe.apply(composite_135_base)

print("  Saved 135-deg predictions (improved + baseline)")


# ═══════════════════════════════════════════════════════════════════════
# Step 9: Compute metrics (SSIM / Bone Dice)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 9: Computing metrics (SSIM / Bone Dice)")
print("=" * 70)

gt_90 = gt_drrs[90.0]
gt_135 = gt_drrs[135.0]


def resize_to_match(img, target):
    if img.shape != target.shape:
        return cv2.resize(img, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return img


def compute_metrics(pred, gt):
    """Compute SSIM and Bone Dice between predicted and ground truth DRR."""
    pred_r = resize_to_match(pred, gt)
    s = ssim(gt, pred_r, data_range=255)
    _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred_bin = cv2.threshold(pred_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intersection = np.logical_and(gt_bin > 0, pred_bin > 0).sum()
    dice = 2.0 * intersection / (np.sum(gt_bin > 0) + np.sum(pred_bin > 0) + 1e-8)
    return s, dice


metrics = {}

# 90-deg comparisons
s_imp, d_imp = compute_metrics(composite_improved_clahe, gt_90)
s_base, d_base = compute_metrics(composite_baseline_clahe, gt_90)
metrics["90deg_improved"] = {"ssim": s_imp, "bone_dice": d_imp}
metrics["90deg_baseline"] = {"ssim": s_base, "bone_dice": d_base}

print(f"\n  90-deg comparison:")
print(f"    Improved  (true axis):  SSIM={s_imp:.4f}, Bone Dice={d_imp:.4f}")
print(f"    Baseline  (ML axis):    SSIM={s_base:.4f}, Bone Dice={d_base:.4f}")
print(f"    Delta SSIM: {s_imp - s_base:+.4f}")
print(f"    Delta Dice: {d_imp - d_base:+.4f}")

# 135-deg comparisons
s_imp135, d_imp135 = compute_metrics(composite_135_imp_clahe, gt_135)
s_base135, d_base135 = compute_metrics(composite_135_base_clahe, gt_135)
metrics["135deg_improved"] = {"ssim": s_imp135, "bone_dice": d_imp135}
metrics["135deg_baseline"] = {"ssim": s_base135, "bone_dice": d_base135}

print(f"\n  135-deg comparison:")
print(f"    Improved  (true axis):  SSIM={s_imp135:.4f}, Bone Dice={d_imp135:.4f}")
print(f"    Baseline  (ML axis):    SSIM={s_base135:.4f}, Bone Dice={d_base135:.4f}")
print(f"    Delta SSIM: {s_imp135 - s_base135:+.4f}")
print(f"    Delta Dice: {d_imp135 - d_base135:+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 10: Load real CR X-ray for comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 10: Real CR X-ray comparison")
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
    s_real_imp, _ = compute_metrics(composite_improved_clahe, real_resized)
    s_real_base, _ = compute_metrics(composite_baseline_clahe, real_resized)
    print(f"  Predicted (improved) vs Real CR: SSIM={s_real_imp:.4f}")
    print(f"  Predicted (baseline) vs Real CR: SSIM={s_real_base:.4f}")
    print(f"  Delta: {s_real_imp - s_real_base:+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Step 11: Generate summary figure
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 11: Generating summary figure")
print("=" * 70)

fig, axes = plt.subplots(4, 4, figsize=(22, 22))
fig.suptitle("CT-to-X-ray Improved Synthesis: Data-Driven Rotation Axis\n"
             f"Axis tilt from ML: {tilt_angle:.1f} deg | "
             f"Axis (hf): [{avg_axis_hf[0]:.3f}, {avg_axis_hf[1]:.3f}, {avg_axis_hf[2]:.3f}]",
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
    axes[0, 3].set_title("Real CR X-ray")
else:
    axes[0, 3].set_visible(False)

# Row 1: Bone separation and rotation visualization
mid_sl = ml_size // 2
vis_sep = np.zeros((pd_size, ap_size, 3), dtype=np.uint8)
vis_sep[:, :, 2] = (vol_humerus[:, :, mid_sl] * 255).astype(np.uint8)  # blue=humerus
vis_sep[:, :, 1] = (vol_forearm[:, :, mid_sl] * 255).astype(np.uint8)  # green=forearm
axes[1, 0].imshow(vis_sep)
axes[1, 0].set_title("Bone separation\n(blue=humerus, green=forearm)")

vis_imp = np.zeros((pd_size, ap_size, 3), dtype=np.uint8)
vis_imp[:, :, 2] = (vol_humerus[:, :, mid_sl] * 255).astype(np.uint8)
vis_imp[:, :, 1] = (vol_forearm_improved[:, :, mid_sl] * 255).astype(np.uint8)
axes[1, 1].imshow(vis_imp)
axes[1, 1].set_title("Improved: True axis rotation\n(blue=humerus, green=forearm@90)")

vis_base = np.zeros((pd_size, ap_size, 3), dtype=np.uint8)
vis_base[:, :, 2] = (vol_humerus[:, :, mid_sl] * 255).astype(np.uint8)
vis_base[:, :, 1] = (vol_forearm_baseline[:, :, mid_sl] * 255).astype(np.uint8)
axes[1, 2].imshow(vis_base)
axes[1, 2].set_title("Baseline: Simple ML axis\n(blue=humerus, green=forearm@90)")

# Show rotation axis in 3D
ax3d = fig.add_subplot(4, 4, 4, projection='3d')
# Draw rotation axis
t_range = np.linspace(-30, 30, 50)
axis_line = pivot_vox[None, :] + t_range[:, None] * axis_vox[None, :]
ax3d.plot(axis_line[:, 2], axis_line[:, 1], axis_line[:, 0],
          'r-', linewidth=2, label='True axis')
# Simple ML axis
ml_line = jc_vox[None, :] + t_range[:, None] * np.array([0, 0, 1])[None, :]
ax3d.plot(ml_line[:, 2], ml_line[:, 1], ml_line[:, 0],
          'b--', linewidth=1.5, label='Simple ML axis')
ax3d.scatter([jc_vox[2]], [jc_vox[1]], [jc_vox[0]], c='k', s=100, marker='+',
             linewidths=2, label='Joint center')
ax3d.set_xlabel("ML")
ax3d.set_ylabel("AP")
ax3d.set_zlabel("PD")
ax3d.set_title(f"Rotation axis comparison\nTilt: {tilt_angle:.1f} deg")
ax3d.legend(fontsize=7)

# Row 2: Composite comparisons (90 deg)
m_imp = metrics["90deg_improved"]
m_base = metrics["90deg_baseline"]

axes[2, 0].imshow(composite_improved_clahe, cmap='gray')
axes[2, 0].set_title(f"Improved (90-deg)\nSSIM={m_imp['ssim']:.3f} Dice={m_imp['bone_dice']:.3f}")

axes[2, 1].imshow(composite_baseline_clahe, cmap='gray')
axes[2, 1].set_title(f"Baseline (90-deg)\nSSIM={m_base['ssim']:.3f} Dice={m_base['bone_dice']:.3f}")

axes[2, 2].imshow(gt_90, cmap='gray')
axes[2, 2].set_title("Ground Truth\n(90-deg LAT DRR)")

# Difference maps
diff_imp = cv2.absdiff(resize_to_match(composite_improved_clahe, gt_90), gt_90)
diff_base = cv2.absdiff(resize_to_match(composite_baseline_clahe, gt_90), gt_90)
diff_compare = diff_base.astype(np.float32) - diff_imp.astype(np.float32)
# Positive = improved is better (less error), negative = baseline is better
axes[2, 3].imshow(diff_compare, cmap='RdBu', vmin=-50, vmax=50)
axes[2, 3].set_title("Error difference\n(blue=improved better, red=baseline better)")

# Row 3: 135-deg validation + metrics summary
m_imp135 = metrics["135deg_improved"]
m_base135 = metrics["135deg_baseline"]

axes[3, 0].imshow(composite_135_imp_clahe, cmap='gray')
axes[3, 0].set_title(f"Improved (135-deg)\nSSIM={m_imp135['ssim']:.3f} Dice={m_imp135['bone_dice']:.3f}")

axes[3, 1].imshow(composite_135_base_clahe, cmap='gray')
axes[3, 1].set_title(f"Baseline (135-deg)\nSSIM={m_base135['ssim']:.3f} Dice={m_base135['bone_dice']:.3f}")

axes[3, 2].imshow(gt_135, cmap='gray')
axes[3, 2].set_title("Ground Truth\n(135-deg LAT DRR)")

# Metrics bar chart
ax_bar = axes[3, 3]
labels = ['90-deg\nSSIM', '90-deg\nDice', '135-deg\nSSIM', '135-deg\nDice']
vals_imp = [m_imp['ssim'], m_imp['bone_dice'], m_imp135['ssim'], m_imp135['bone_dice']]
vals_base = [m_base['ssim'], m_base['bone_dice'], m_base135['ssim'], m_base135['bone_dice']]

x = np.arange(len(labels))
w = 0.35
bars1 = ax_bar.bar(x - w/2, vals_imp, w, label='Improved (true axis)', color='tab:green', alpha=0.8)
bars2 = ax_bar.bar(x + w/2, vals_base, w, label='Baseline (ML axis)', color='tab:orange', alpha=0.8)
ax_bar.set_ylabel('Score')
ax_bar.set_title('Metrics Comparison')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=8)
ax_bar.legend(fontsize=8)
ax_bar.set_ylim(0, 1.0)
ax_bar.grid(True, alpha=0.3, axis='y')

for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

for ax in axes.flat:
    if not isinstance(ax, plt.Axes) or hasattr(ax, 'get_zlim'):
        continue
    ax.axis('off')
# Re-enable axis for bar chart
ax_bar.axis('on')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "improved_axis.png"), dpi=150)
print(f"  Saved: {os.path.join(OUT_DIR, 'improved_axis.png')}")


# ═══════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nRotation Axis Analysis:")
print(f"  True flexion axis (humerus-fixed): [{avg_axis_hf[0]:.4f}, {avg_axis_hf[1]:.4f}, {avg_axis_hf[2]:.4f}]")
print(f"  Simple ML axis (humerus-fixed):    [0.0000, 0.0000, 1.0000]")
print(f"  Axis tilt from ML: {tilt_angle:.2f} degrees")
print(f"  Per-landmark axes:")
for name in FOREARM_NAMES:
    ax = axes_from_landmarks[name]
    print(f"    {name:20s}: [{ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f}]")

print(f"\nMetrics comparison (90-deg target):")
print(f"  {'Method':25s} {'SSIM':>8s} {'Dice':>8s}")
print(f"  {'-'*45}")
print(f"  {'Improved (true axis)':25s} {m_imp['ssim']:8.4f} {m_imp['bone_dice']:8.4f}")
print(f"  {'Baseline (ML axis)':25s} {m_base['ssim']:8.4f} {m_base['bone_dice']:8.4f}")
print(f"  {'Delta':25s} {m_imp['ssim']-m_base['ssim']:+8.4f} {m_imp['bone_dice']-m_base['bone_dice']:+8.4f}")

print(f"\nMetrics comparison (135-deg validation):")
print(f"  {'Method':25s} {'SSIM':>8s} {'Dice':>8s}")
print(f"  {'-'*45}")
print(f"  {'Improved (true axis)':25s} {m_imp135['ssim']:8.4f} {m_imp135['bone_dice']:8.4f}")
print(f"  {'Baseline (ML axis)':25s} {m_base135['ssim']:8.4f} {m_base135['bone_dice']:8.4f}")
print(f"  {'Delta':25s} {m_imp135['ssim']-m_base135['ssim']:+8.4f} {m_imp135['bone_dice']-m_base135['bone_dice']:+8.4f}")

if real_xray is not None:
    print(f"\n  vs Real CR X-ray:")
    print(f"    Improved SSIM: {s_real_imp:.4f}")
    print(f"    Baseline SSIM: {s_real_base:.4f}")
    print(f"    Delta: {s_real_imp - s_real_base:+.4f}")

print(f"\nOutput directory: {OUT_DIR}")
print(f"Main figure: {os.path.join(OUT_DIR, 'improved_axis.png')}")
print("\nDone.")
