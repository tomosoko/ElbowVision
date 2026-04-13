#!/usr/bin/env python3
"""
Experiment: Virtual Flexion Synthesis Accuracy Improvement
==========================================================
Phase 1: Cross-volume baseline (quantify how bad rigid rotation is)
Phase 2: Root cause analysis (rotation axis, deformation patterns)
Phase 3: Improved approaches (nearest-volume, full-range, data-driven axis)
Phase 4: Validation and comparison

Left arm phantom, FC85 bone kernel, hu_min=50, hu_max=1000
3 CT volumes: Series 4=180°, Series 8=135°, Series 12=90°
"""

import sys
import os
import json
import time
import math

import cv2
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = "/Users/kohei/develop/research/ElbowVision"
sys.path.insert(0, os.path.join(ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume, auto_detect_landmarks, generate_drr,
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    rotate_volume_and_landmarks,
)

CT_DIR = os.path.join(ROOT, "data/raw_dicom/ct_volume",
                      "ﾃｽﾄ 008_0009900008_20260310_108Y_F_000")
OUT_DIR = os.path.join(ROOT, "results/flexion_accuracy_experiment")
os.makedirs(OUT_DIR, exist_ok=True)

SERIES = [(4, 180.0), (8, 135.0), (12, 90.0)]
TARGET_SIZE = 128
HU_MIN, HU_MAX = 50, 1000

KP_NAMES = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
            "forearm_shaft", "radial_head", "olecranon"]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_dice(pred, gt):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    _, gb = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pb = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inter = np.logical_and(gb > 0, pb > 0).sum()
    return 2.0 * inter / (np.sum(gb > 0) + np.sum(pb > 0) + 1e-8)


def compute_ssim(pred, gt):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    return ssim(gt, pred, data_range=255)


def landmark_error_mm(lm_pred, lm_actual, vol_shape, voxel_mm):
    """Compute per-landmark error in mm."""
    pd, ap, ml = vol_shape
    errors = {}
    for name in KP_NAMES:
        if name not in lm_pred or name not in lm_actual:
            continue
        p = np.array(lm_pred[name])
        a = np.array(lm_actual[name])
        # Convert normalized to mm
        scale = np.array([pd, ap, ml]) * voxel_mm
        err = np.linalg.norm((p - a) * scale)
        errors[name] = err
    return errors


def to_mm(lm_dict, voxel_mm, vol_shape):
    """Convert normalized landmark dict to mm-scale."""
    out = {}
    pd_s, ap_s, ml_s = vol_shape
    for name, (pd, ap, ml) in lm_dict.items():
        out[name] = np.array([pd * pd_s * voxel_mm,
                              ap * ap_s * voxel_mm,
                              ml * ml_s * voxel_mm])
    return out


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load all 3 volumes
# ════════════════════════════════════════════════════���══════════════════
print("=" * 70)
print("LOADING 3 CT VOLUMES")
print("=" * 70)

volumes = {}
for sn, angle in SERIES:
    print(f"\n--- Series {sn}  angle={angle}° ---")
    vol, spacing, lat, vox_mm = load_ct_volume(
        CT_DIR, laterality='L', series_num=sn,
        hu_min=HU_MIN, hu_max=HU_MAX, target_size=TARGET_SIZE)
    lm = auto_detect_landmarks(vol, laterality=lat)
    volumes[angle] = dict(vol=vol, lm=lm, voxel_mm=vox_mm, lat=lat,
                          series=sn, shape=vol.shape)
    print(f"  Shape: {vol.shape}, voxel: {vox_mm:.2f}mm")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Cross-Volume Baseline Validation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: CROSS-VOLUME BASELINE (rigid rotation accuracy)")
print("=" * 70)

# For each pair of volumes: synthesize target angle from source volume
# and compare with actual DRR from the target's real CT
baseline_results = []

# Generate ground truth DRRs from each real volume
gt_drrs = {}
for angle in [180.0, 135.0, 90.0]:
    v = volumes[angle]
    gt_drrs[angle] = {
        'LAT': generate_drr(v['vol'], axis="LAT" if angle > 120 else "AP",
                            sid_mm=1000.0, voxel_mm=v['voxel_mm']),
    }
    # For 90° volume, the LAT projection is along AP axis (forearm bends into ML)
    if angle <= 120:
        gt_drrs[angle]['LAT_axis'] = "AP"
    else:
        gt_drrs[angle]['LAT_axis'] = "LAT"

# Cross-volume synthesis: rotate source to target angle
source_angles = [180.0, 135.0, 90.0]
target_angles_all = [180.0, 165.0, 150.0, 135.0, 120.0, 105.0, 90.0]

print("\nGenerating cross-volume synthesis matrix...")
cross_results = {}

for src_angle in source_angles:
    src = volumes[src_angle]
    for tgt_angle in [a for a in [180.0, 135.0, 90.0] if a != src_angle]:
        delta = tgt_angle - src_angle
        print(f"\n  {src_angle}° → {tgt_angle}° (delta={delta:+.0f}°)")

        # Rotate source volume to target angle
        rot_vol, rot_lm = rotate_volume_and_landmarks(
            src['vol'], src['lm'],
            forearm_rotation_deg=0.0,
            flexion_deg=tgt_angle,
            base_flexion=src_angle,
            valgus_deg=0.0,
        )

        # Generate DRR from rotated volume
        # Use the appropriate projection axis for the target angle
        if tgt_angle <= 120:
            proj_axis = "AP"
        else:
            proj_axis = "LAT"

        synth_drr = generate_drr(rot_vol, axis=proj_axis,
                                 sid_mm=1000.0, voxel_mm=src['voxel_mm'])

        # Compare with ground truth DRR
        gt = gt_drrs[tgt_angle]['LAT']

        dice = compute_dice(synth_drr, gt)
        ssim_val = compute_ssim(synth_drr, gt)

        # Landmark error
        tgt_v = volumes[tgt_angle]
        lm_err = landmark_error_mm(rot_lm, tgt_v['lm'], src['shape'], src['voxel_mm'])
        mean_lm_err = np.mean(list(lm_err.values())) if lm_err else 0

        result = {
            'source': src_angle,
            'target': tgt_angle,
            'delta': abs(delta),
            'dice': dice,
            'ssim': ssim_val,
            'mean_lm_err_mm': mean_lm_err,
            'per_lm_err': lm_err,
        }
        cross_results[(src_angle, tgt_angle)] = result
        baseline_results.append(result)

        print(f"    Dice={dice:.3f}  SSIM={ssim_val:.3f}  LM_err={mean_lm_err:.1f}mm")

# Also measure self-consistency (0° rotation)
for angle in [180.0, 135.0, 90.0]:
    v = volumes[angle]
    rot_vol, rot_lm = rotate_volume_and_landmarks(
        v['vol'], v['lm'], 0.0, angle, base_flexion=angle, valgus_deg=0.0)
    proj_axis = "AP" if angle <= 120 else "LAT"
    synth = generate_drr(rot_vol, axis=proj_axis, sid_mm=1000.0, voxel_mm=v['voxel_mm'])
    gt = gt_drrs[angle]['LAT']
    dice = compute_dice(synth, gt)
    ssim_val = compute_ssim(synth, gt)
    cross_results[(angle, angle)] = {
        'source': angle, 'target': angle, 'delta': 0,
        'dice': dice, 'ssim': ssim_val, 'mean_lm_err_mm': 0.0,
    }
    print(f"\n  {angle}° → {angle}° (self): Dice={dice:.3f}  SSIM={ssim_val:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Root Cause Analysis — Data-driven rotation axis
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: DATA-DRIVEN ROTATION AXIS ANALYSIS")
print("=" * 70)

# Extract forearm landmarks in mm for all 3 angles
forearm_names = ["forearm_shaft", "radial_head", "olecranon"]
humerus_names = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle"]

all_lm_mm = {}
for angle in [180.0, 135.0, 90.0]:
    v = volumes[angle]
    all_lm_mm[angle] = to_mm(v['lm'], v['voxel_mm'], v['shape'])

# Procrustes alignment on humerus to find the rotation between volumes
def procrustes_rotation(pts_from, pts_to):
    """Find rotation R such that pts_to ≈ R @ pts_from (centered)."""
    c_from = pts_from.mean(axis=0)
    c_to = pts_to.mean(axis=0)
    p = pts_from - c_from
    q = pts_to - c_to
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, c_from, c_to


# Compute rotation axes between pairs
print("\nComputing data-driven rotation axes between volume pairs:")
axis_results = {}

for (a1, a2) in [(180.0, 135.0), (135.0, 90.0), (180.0, 90.0)]:
    # Get humerus landmarks to align
    hum_pts1 = np.array([all_lm_mm[a1][n] for n in humerus_names if n in all_lm_mm[a1]])
    hum_pts2 = np.array([all_lm_mm[a2][n] for n in humerus_names if n in all_lm_mm[a2]])

    # Procrustes on humerus
    R_hum, c1, c2 = procrustes_rotation(hum_pts1, hum_pts2)

    # Get forearm landmarks (aligned to humerus frame)
    fa_pts1 = np.array([all_lm_mm[a1][n] - c1 for n in forearm_names])
    fa_pts2 = np.array([all_lm_mm[a2][n] - c2 for n in forearm_names])

    # Align humerus
    fa_pts2_aligned = (R_hum.T @ fa_pts2.T).T

    # Find forearm rotation
    R_fa, _, _ = procrustes_rotation(fa_pts1, fa_pts2_aligned)

    # Extract rotation axis and angle from R_fa
    rot = Rotation.from_matrix(R_fa)
    rotvec = rot.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)
    if angle_rad > 1e-6:
        axis = rotvec / angle_rad
    else:
        axis = np.array([0, 0, 1])

    expected_angle = a1 - a2
    actual_angle = np.degrees(angle_rad)

    # Compare with fixed ML axis [0, 0, 1]
    ml_axis = np.array([0, 0, 1])
    axis_deviation = np.degrees(np.arccos(np.clip(abs(np.dot(axis, ml_axis)), 0, 1)))

    axis_results[(a1, a2)] = {
        'rotation_axis': axis.tolist(),
        'rotation_angle_deg': actual_angle,
        'expected_angle_deg': expected_angle,
        'angle_ratio': actual_angle / expected_angle if expected_angle != 0 else 0,
        'axis_deviation_from_ML_deg': axis_deviation,
    }

    print(f"\n  {a1}° → {a2}° (expected Δ={expected_angle}°):")
    print(f"    Actual rotation: {actual_angle:.1f}°")
    print(f"    Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
    print(f"    Deviation from ML axis: {axis_deviation:.1f}°")
    print(f"    ML axis [0,0,1] vs actual: angle ratio={actual_angle/expected_angle:.2f}")

# Compute average data-driven axis
axes = [np.array(axis_results[k]['rotation_axis']) for k in axis_results]
# Weight by angle magnitude
weights = [abs(axis_results[k]['expected_angle_deg']) for k in axis_results]
avg_axis = np.zeros(3)
for a, w in zip(axes, weights):
    # Ensure consistent sign (point in same direction)
    if np.dot(a, axes[0]) < 0:
        a = -a
    avg_axis += a * w
avg_axis = avg_axis / (np.linalg.norm(avg_axis) + 1e-12)
print(f"\n  Average data-driven axis: [{avg_axis[0]:.3f}, {avg_axis[1]:.3f}, {avg_axis[2]:.3f}]")
print(f"  Fixed ML axis:           [0.000, 0.000, 1.000]")
deviation = np.degrees(np.arccos(np.clip(abs(np.dot(avg_axis, np.array([0, 0, 1]))), 0, 1)))
print(f"  Deviation: {deviation:.1f}°")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Improved Approaches
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: IMPROVED APPROACHES")
print("=" * 70)


# --- Approach A: Nearest-volume selection ---
# For each target angle, use the closest real CT volume
# This minimizes rotation magnitude

def nearest_volume_for_angle(target_angle, available_angles=[180.0, 135.0, 90.0]):
    """Select the volume requiring minimum rotation."""
    return min(available_angles, key=lambda a: abs(a - target_angle))


# --- Approach B: Data-driven rotation ---
# Use the computed rotation axis instead of fixed ML axis

def rotation_matrix_custom_axis(axis, angle_deg):
    """Rotation matrix around arbitrary axis by angle_deg."""
    rot = Rotation.from_rotvec(np.radians(angle_deg) * np.array(axis))
    return rot.as_matrix()


def rotate_volume_data_driven(volume, landmarks_norm, target_flexion,
                               base_flexion, rotation_axis, voxel_mm=1.0):
    """Rotate forearm using data-driven axis instead of fixed ML axis."""
    pd, ap, ml = volume.shape
    delta_deg = target_flexion - base_flexion

    if "joint_center" in landmarks_norm:
        jc = landmarks_norm["joint_center"]
        center = np.array([jc[0] * pd, jc[1] * ap, jc[2] * ml])
    else:
        center = np.array([pd / 2, ap / 2, ml / 2])

    # Build rotation matrix from data-driven axis
    R = rotation_matrix_custom_axis(rotation_axis, delta_deg)
    R_inv = R.T

    # Same forearm separation as original
    joint_pd_vox = int(center[0])
    blend_half = max(2, int(pd * 0.03))

    rotated = volume.copy()
    offset_forearm = center - R_inv @ center
    forearm_rotated = affine_transform(
        volume, R_inv, offset=offset_forearm, order=3, mode='constant', cval=0.0
    )

    blend_mask = np.zeros(pd, dtype=np.float32)
    for i in range(pd):
        if i >= joint_pd_vox + blend_half:
            blend_mask[i] = 1.0
        elif i > joint_pd_vox - blend_half:
            blend_mask[i] = (i - (joint_pd_vox - blend_half)) / (2 * blend_half)

    blend_3d = blend_mask[:, None, None]
    rotated = (1.0 - blend_3d) * volume + blend_3d * forearm_rotated

    # Rotate landmarks
    humerus_landmarks = {"humerus_shaft"}
    rotated_lm = {}
    for name, (nPD, nAP, nML) in landmarks_norm.items():
        if name in humerus_landmarks:
            rotated_lm[name] = (nPD, nAP, nML)
        else:
            p = np.array([nPD * pd, nAP * ap, nML * ml]) - center
            p_rot = R @ p + center
            rotated_lm[name] = (
                float(np.clip(p_rot[0] / pd, 0, 1)),
                float(np.clip(p_rot[1] / ap, 0, 1)),
                float(np.clip(p_rot[2] / ml, 0, 1)),
            )

    return rotated, rotated_lm


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Compare all approaches
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: COMPARISON OF APPROACHES")
print("=" * 70)

# Test approaches at each known angle (cross-volume validation)
approaches = {
    'baseline_current': 'Current (fixed volume + ML axis)',
    'nearest_ml': 'Nearest volume + ML axis',
    'nearest_datadriven': 'Nearest volume + Data-driven axis',
    'original_datadriven': 'Original volume + Data-driven axis',
}

comparison_results = {name: [] for name in approaches}

# Test pairs: synthesize each GT angle from each source
test_pairs = []
for tgt in [180.0, 135.0, 90.0]:
    for src in [180.0, 135.0, 90.0]:
        if src != tgt:
            test_pairs.append((src, tgt))

for src_angle, tgt_angle in test_pairs:
    src = volumes[src_angle]
    tgt = volumes[tgt_angle]
    gt_drr = gt_drrs[tgt_angle]['LAT']
    proj_axis = "AP" if tgt_angle <= 120 else "LAT"

    # --- Baseline: current approach (specific source + ML axis) ---
    rot_vol, rot_lm = rotate_volume_and_landmarks(
        src['vol'], src['lm'], 0.0, tgt_angle,
        base_flexion=src_angle, valgus_deg=0.0)
    synth = generate_drr(rot_vol, axis=proj_axis, sid_mm=1000.0, voxel_mm=src['voxel_mm'])
    dice = compute_dice(synth, gt_drr)
    ssim_v = compute_ssim(synth, gt_drr)
    lm_err = landmark_error_mm(rot_lm, tgt['lm'], src['shape'], src['voxel_mm'])
    comparison_results['baseline_current'].append({
        'src': src_angle, 'tgt': tgt_angle,
        'delta': abs(tgt_angle - src_angle),
        'dice': dice, 'ssim': ssim_v,
        'lm_err': np.mean(list(lm_err.values())),
    })

    # --- Data-driven axis (same source) ---
    rot_vol_dd, rot_lm_dd = rotate_volume_data_driven(
        src['vol'], src['lm'], tgt_angle, src_angle, avg_axis, src['voxel_mm'])
    synth_dd = generate_drr(rot_vol_dd, axis=proj_axis, sid_mm=1000.0, voxel_mm=src['voxel_mm'])
    dice_dd = compute_dice(synth_dd, gt_drr)
    ssim_dd = compute_ssim(synth_dd, gt_drr)
    lm_err_dd = landmark_error_mm(rot_lm_dd, tgt['lm'], src['shape'], src['voxel_mm'])
    comparison_results['original_datadriven'].append({
        'src': src_angle, 'tgt': tgt_angle,
        'delta': abs(tgt_angle - src_angle),
        'dice': dice_dd, 'ssim': ssim_dd,
        'lm_err': np.mean(list(lm_err_dd.values())),
    })

# Nearest-volume tests
for tgt_angle in [180.0, 135.0, 90.0]:
    nearest = nearest_volume_for_angle(tgt_angle)
    if nearest == tgt_angle:
        # Self — use a different angle to test nearest-volume concept
        continue
    src = volumes[nearest]
    tgt = volumes[tgt_angle]
    gt_drr = gt_drrs[tgt_angle]['LAT']
    proj_axis = "AP" if tgt_angle <= 120 else "LAT"
    delta = abs(tgt_angle - nearest)

    # ML axis
    rot_vol, rot_lm = rotate_volume_and_landmarks(
        src['vol'], src['lm'], 0.0, tgt_angle,
        base_flexion=nearest, valgus_deg=0.0)
    synth = generate_drr(rot_vol, axis=proj_axis, sid_mm=1000.0, voxel_mm=src['voxel_mm'])
    dice = compute_dice(synth, gt_drr)
    ssim_v = compute_ssim(synth, gt_drr)
    lm_err = landmark_error_mm(rot_lm, tgt['lm'], src['shape'], src['voxel_mm'])
    comparison_results['nearest_ml'].append({
        'src': nearest, 'tgt': tgt_angle,
        'delta': delta,
        'dice': dice, 'ssim': ssim_v,
        'lm_err': np.mean(list(lm_err.values())),
    })

    # Data-driven axis
    rot_vol_dd, rot_lm_dd = rotate_volume_data_driven(
        src['vol'], src['lm'], tgt_angle, nearest, avg_axis, src['voxel_mm'])
    synth_dd = generate_drr(rot_vol_dd, axis=proj_axis, sid_mm=1000.0, voxel_mm=src['voxel_mm'])
    dice_dd = compute_dice(synth_dd, gt_drr)
    ssim_dd = compute_ssim(synth_dd, gt_drr)
    lm_err_dd = landmark_error_mm(rot_lm_dd, tgt['lm'], src['shape'], src['voxel_mm'])
    comparison_results['nearest_datadriven'].append({
        'src': nearest, 'tgt': tgt_angle,
        'delta': delta,
        'dice': dice_dd, 'ssim': ssim_dd,
        'lm_err': np.mean(list(lm_err_dd.values())),
    })

# Print comparison table
print("\n" + "=" * 90)
print(f"{'Approach':<40s} {'Src→Tgt':>12s} {'Δ':>5s} {'Dice':>7s} {'SSIM':>7s} {'LM mm':>7s}")
print("-" * 90)
for approach_name, results in comparison_results.items():
    for r in sorted(results, key=lambda x: x['delta']):
        label = approaches.get(approach_name, approach_name)
        print(f"  {label:<38s} {r['src']:3.0f}→{r['tgt']:3.0f}° {r['delta']:5.0f}° "
              f"{r['dice']:7.3f} {r['ssim']:7.3f} {r['lm_err']:7.1f}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Generate comprehensive comparison figure
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: GENERATING COMPARISON FIGURES")
print("=" * 70)

fig = plt.figure(figsize=(28, 20))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel 1: Cross-volume Dice vs rotation magnitude ---
ax1 = fig.add_subplot(gs[0, 0:2])
for approach_name, results in comparison_results.items():
    if results:
        deltas = [r['delta'] for r in results]
        dices = [r['dice'] for r in results]
        ax1.scatter(deltas, dices, label=approaches.get(approach_name, approach_name),
                   s=80, alpha=0.8)
ax1.set_xlabel("Rotation Magnitude (°)")
ax1.set_ylabel("Dice Score")
ax1.set_title("DRR Accuracy vs Rotation Magnitude")
ax1.legend(fontsize=8, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# --- Panel 2: SSIM vs rotation magnitude ---
ax2 = fig.add_subplot(gs[0, 2:4])
for approach_name, results in comparison_results.items():
    if results:
        deltas = [r['delta'] for r in results]
        ssims = [r['ssim'] for r in results]
        ax2.scatter(deltas, ssims, label=approaches.get(approach_name, approach_name),
                   s=80, alpha=0.8)
ax2.set_xlabel("Rotation Magnitude (°)")
ax2.set_ylabel("SSIM")
ax2.set_title("DRR Quality vs Rotation Magnitude")
ax2.legend(fontsize=8, loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# --- Panel 3: Visual comparison for 180→90° ---
# Show: GT 90° DRR, baseline 180→90, nearest (135→90), data-driven
visual_pairs = [
    (180.0, 90.0, "180→90° (worst case)"),
    (135.0, 90.0, "135→90° (nearest)"),
    (180.0, 135.0, "180→135°"),
    (90.0, 135.0, "90→135°"),
]

for idx, (src_a, tgt_a, title) in enumerate(visual_pairs):
    ax = fig.add_subplot(gs[1, idx])
    src = volumes[src_a]
    tgt = volumes[tgt_a]
    proj_axis = "AP" if tgt_a <= 120 else "LAT"

    # Synthesize with baseline
    rot_vol, _ = rotate_volume_and_landmarks(
        src['vol'], src['lm'], 0.0, tgt_a,
        base_flexion=src_a, valgus_deg=0.0)
    synth = generate_drr(rot_vol, axis=proj_axis, sid_mm=1000.0, voxel_mm=src['voxel_mm'])
    gt = gt_drrs[tgt_a]['LAT']

    # Side by side: synth | gt
    if synth.shape != gt.shape:
        synth = cv2.resize(synth, (gt.shape[1], gt.shape[0]))
    combined = np.hstack([synth, gt])
    ax.imshow(combined, cmap='gray')
    ax.set_title(f"{title}\nSynth | GT\nDice={compute_dice(synth, gt):.3f}", fontsize=9)
    ax.axis('off')
    h = synth.shape[0]
    ax.axvline(synth.shape[1], color='yellow', lw=1)

# --- Panel 4: GT DRRs at all 3 angles ---
for idx, angle in enumerate([180.0, 135.0, 90.0]):
    ax = fig.add_subplot(gs[2, idx])
    gt = gt_drrs[angle]['LAT']
    ax.imshow(gt, cmap='gray')
    ax.set_title(f"GT {angle:.0f}° ({gt_drrs[angle].get('LAT_axis', 'LAT')} proj)", fontsize=9)
    ax.axis('off')

# Summary stats panel
ax_sum = fig.add_subplot(gs[2, 3])
ax_sum.axis('off')
summary_text = "SUMMARY\n" + "=" * 40 + "\n\n"

for approach_name, results in comparison_results.items():
    if results:
        avg_dice = np.mean([r['dice'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lm = np.mean([r['lm_err'] for r in results])
        label = approaches.get(approach_name, approach_name)
        summary_text += f"{label}:\n"
        summary_text += f"  Dice={avg_dice:.3f}  SSIM={avg_ssim:.3f}  LM={avg_lm:.1f}mm\n\n"

summary_text += f"\nData-driven axis: [{avg_axis[0]:.3f}, {avg_axis[1]:.3f}, {avg_axis[2]:.3f}]\n"
summary_text += f"Deviation from ML: {deviation:.1f}°\n"
summary_text += f"\nTime: {time.time()-t0:.0f}s"

ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Virtual Flexion Synthesis: Accuracy Experiment\n"
             "Left phantom, 3 CT volumes (180°/135°/90°), Cross-volume validation",
             fontsize=14, fontweight='bold')

out_fig = os.path.join(OUT_DIR, "comparison_all_approaches.png")
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_fig}")


# ═══════════════════════════════════════════════════════════════════════
# Save detailed results as JSON
# ═══════════════════════════════════════════════════════════════════════

# Flatten comparison results for JSON
json_results = {
    'baseline_cross_volume': baseline_results,
    'axis_analysis': {f"{k[0]}_{k[1]}": v for k, v in axis_results.items()},
    'average_data_driven_axis': avg_axis.tolist(),
    'axis_deviation_from_ML_deg': float(deviation),
    'comparison': {},
}

for approach_name, results in comparison_results.items():
    json_results['comparison'][approach_name] = results

# Convert numpy types
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

json_path = os.path.join(OUT_DIR, "experiment_results.json")
with open(json_path, 'w') as f:
    json.dump(convert_numpy(json_results), f, indent=2, ensure_ascii=False)
print(f"Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR elbow_synth.py")
print("=" * 70)

# Compute per-approach averages
print("\nAverage metrics across all cross-volume tests:")
for approach_name, results in comparison_results.items():
    if results:
        avg_dice = np.mean([r['dice'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lm = np.mean([r['lm_err'] for r in results])
        print(f"  {approaches.get(approach_name, approach_name):40s}: "
              f"Dice={avg_dice:.3f}  SSIM={avg_ssim:.3f}  LM={avg_lm:.1f}mm")

print(f"""
PROPOSED CHANGES TO elbow_synth.py:

1. USE ALL 3 VOLUMES (currently 135° volume is UNUSED)
   - Current: LAT uses only ≤120° volumes, AP uses only ≥150°
   - Fix: For each target angle, select nearest volume
   - Range per volume:
     * 90° vol  → target 60-112°  (max Δ=22°)
     * 135° vol → target 113-157° (max Δ=22°)
     * 180° vol → target 158-180° (max Δ=22°)
   - Benefit: max rotation drops from 30° to 22°, fills 120-150° gap

2. DATA-DRIVEN ROTATION AXIS
   - Current: fixed ML axis [0, 0, 1]
   - Proposed: [{avg_axis[0]:.3f}, {avg_axis[1]:.3f}, {avg_axis[2]:.3f}]
   - Deviation: {deviation:.1f}° from ML axis

3. FULL 60-180° LAT COVERAGE
   - Current LAT: 60-120° only
   - Proposed: 60-180° using nearest volume selection
   - This enables ConvNeXt to predict the full range

Total time: {time.time()-t0:.0f}s
""")
