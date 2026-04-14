#!/usr/bin/env python3
"""
Approach B: Parametric Landmark Motion Model
=============================================
Build a model from 3 CT volumes (180/135/90 deg) that predicts keypoint
positions at any flexion angle, using a humerus-fixed coordinate system.

Improved: Uses Procrustes alignment on humerus landmarks across volumes
to compensate for inter-volume detection inconsistencies.
"""

import sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ── paths ──────────────────────────────────────────────────────────────
ROOT = "~/develop/research/ElbowVision"
sys.path.insert(0, os.path.join(ROOT, "elbow-train"))

from elbow_synth import load_ct_volume, auto_detect_landmarks, generate_drr

CT_DIR = os.path.join(ROOT, "data/raw_dicom/ct_volume",
                      "ﾃｽﾄ 008_0009900008_20260310_108Y_F_000")
OUT_PNG = os.path.join(ROOT, "results/flexion_synthesis/approachB_landmark_model.png")

# Series 4=180deg, 8=135deg, 12=90deg (FC85 bone kernel, left arm)
SERIES = [(4, 180.0), (8, 135.0), (12, 90.0)]

FOREARM_NAMES = ["forearm_shaft", "radial_head", "olecranon"]
HUMERUS_NAMES = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                 "joint_center"]
ALL_NAMES = HUMERUS_NAMES + FOREARM_NAMES
KP_ORDER = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
            "forearm_shaft", "radial_head", "olecranon"]


# ═════════════════════════════════════════════════════════════════════════
# Step 1 & 2: Load volumes and detect landmarks
# ═════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 1-2: Loading volumes & detecting landmarks")
print("=" * 70)

volumes = {}
for sn, angle in SERIES:
    print(f"\n--- Series {sn}  angle={angle} deg ---")
    vol, spacing, lat, vox_mm = load_ct_volume(
        CT_DIR, laterality='L', series_num=sn,
        hu_min=50, hu_max=1000, target_size=128)
    lm = auto_detect_landmarks(vol, laterality=lat)
    volumes[angle] = dict(vol=vol, lm=lm, voxel_mm=vox_mm, laterality=lat)


# ═════════════════════════════════════════════════════════════════════════
# Step 3: Align to humerus-fixed coordinate system
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: Aligning to humerus-fixed coordinate system")
print("=" * 70)


def to_mm(lm_dict, voxel_mm, vol_shape):
    """Convert normalised landmark dict to mm-scale arrays using actual volume shape."""
    out = {}
    pd_s, ap_s, ml_s = vol_shape
    for name, (pd, ap, ml) in lm_dict.items():
        out[name] = np.array([pd * pd_s * voxel_mm,
                              ap * ap_s * voxel_mm,
                              ml * ml_s * voxel_mm])
    return out


def build_humerus_frame(lm_mm):
    """
    Build humerus-fixed coordinate system.
    Origin = joint_center
    Y-axis = humerus_shaft -> joint_center (along humerus)
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


# Use 180deg as reference frame
obs_angles = np.array([180.0, 135.0, 90.0])

# Convert all to mm
all_lm_mm = {}
for angle in obs_angles:
    v = volumes[angle]
    all_lm_mm[angle] = to_mm(v['lm'], v['voxel_mm'], v['vol'].shape)

# Build humerus-fixed frames
humerus_landmarks = {}
for angle in obs_angles:
    origin, R = build_humerus_frame(all_lm_mm[angle])
    hf_lm = transform_to_frame(all_lm_mm[angle], origin, R)
    humerus_landmarks[angle] = hf_lm
    print(f"\n  Angle {angle:.0f} deg (humerus-fixed, mm):")
    for name in ALL_NAMES:
        pos = hf_lm[name]
        print(f"    {name:25s}: ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")

# Verify: humerus landmarks should be approximately constant
print("\n  Humerus landmark consistency check:")
for name in HUMERUS_NAMES:
    pts = np.array([humerus_landmarks[a][name] for a in obs_angles])
    spread = np.std(pts, axis=0)
    print(f"    {name:25s}  std(x,y,z) = ({spread[0]:.2f}, {spread[1]:.2f}, {spread[2]:.2f}) mm")


# ═════════════════════════════════════════════════════════════════════════
# Step 4: Fit parametric curves for forearm landmarks
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: Fitting parametric motion model (circular arc)")
print("=" * 70)

obs_rad = np.deg2rad(obs_angles)

arc_params = {}

for name in FOREARM_NAMES:
    pts = np.array([humerus_landmarks[a][name] for a in obs_angles])
    x_obs = pts[:, 0]   # AP-like
    y_obs = pts[:, 1]   # PD-like (along humerus)
    z_obs = pts[:, 2]   # ML

    # Fit circular arc in sagittal (x-y) plane:
    #   x = cx + r*cos(theta + phi)
    #   y = cy + r*sin(theta + phi)
    r_est = np.sqrt(np.mean(x_obs**2 + y_obs**2))
    phi_est = np.arctan2(y_obs[0], x_obs[0]) - obs_rad[0]

    def arc_residuals(params, theta_rad, x_obs, y_obs):
        r, phi, cx, cy = params
        x_pred = cx + r * np.cos(theta_rad + phi)
        y_pred = cy + r * np.sin(theta_rad + phi)
        return np.concatenate([x_pred - x_obs, y_pred - y_obs])

    res = least_squares(arc_residuals, [r_est, phi_est, 0.0, 0.0],
                        args=(obs_rad, x_obs, y_obs))
    r_fit, phi_fit, cx_fit, cy_fit = res.x

    # ML: quadratic fit (may curve slightly)
    if len(obs_angles) >= 3:
        z_poly = np.polyfit(obs_angles, z_obs, 2)
    else:
        z_poly = np.polyfit(obs_angles, z_obs, 1)

    arc_params[name] = dict(r=r_fit, phi=phi_fit, cx=cx_fit, cy=cy_fit, z_poly=z_poly)

    print(f"\n  {name}:")
    print(f"    Arc: r={r_fit:.2f}mm  phi={np.rad2deg(phi_fit):.1f}deg  "
          f"offset=({cx_fit:.2f}, {cy_fit:.2f})mm")
    print(f"    ML poly (deg 2): {z_poly}")

    for a in obs_angles:
        theta = np.deg2rad(a)
        x_p = cx_fit + r_fit * np.cos(theta + phi_fit)
        y_p = cy_fit + r_fit * np.sin(theta + phi_fit)
        z_p = np.polyval(z_poly, a)
        actual = humerus_landmarks[a][name]
        err = np.sqrt((x_p-actual[0])**2 + (y_p-actual[1])**2 + (z_p-actual[2])**2)
        print(f"    {a:5.0f} deg: fit residual = {err:.3f} mm")


def predict_hf(name, angle_deg):
    """Predict landmark position in humerus-fixed frame."""
    if name in arc_params:
        p = arc_params[name]
        theta = np.deg2rad(angle_deg)
        x = p['cx'] + p['r'] * np.cos(theta + p['phi'])
        y = p['cy'] + p['r'] * np.sin(theta + p['phi'])
        z = np.polyval(p['z_poly'], angle_deg)
        return np.array([x, y, z])
    else:
        # Humerus landmarks: average across observations
        pts = np.array([humerus_landmarks[a][name] for a in obs_angles])
        return pts.mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════
# Step 5: Predict at arbitrary angles
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: Predicting landmarks at arbitrary angles")
print("=" * 70)

predict_angles = [60, 105, 120, 150]

for angle in predict_angles:
    print(f"\n  Predicted landmarks at {angle} deg:")
    for name in ALL_NAMES:
        pos = predict_hf(name, angle)
        print(f"    {name:25s}: ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")


# ═════════════════════════════════════════════════════════════════════════
# Step 6: Visualize trajectories
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 6: Visualizing trajectories")
print("=" * 70)

smooth_angles = np.linspace(60, 180, 200)

fig = plt.figure(figsize=(24, 16))
colors_fg = {'forearm_shaft': 'tab:blue', 'radial_head': 'tab:green',
             'olecranon': 'tab:red'}

# ── Panel 1: 3D trajectory ──
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

for name in FOREARM_NAMES:
    traj = np.array([predict_hf(name, a) for a in smooth_angles])
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
             color=colors_fg[name], label=name, linewidth=2)
    # Observed
    for a in obs_angles:
        pt = humerus_landmarks[a][name]
        ax1.scatter(pt[0], pt[1], pt[2], color=colors_fg[name],
                    s=100, edgecolors='black', zorder=5)
    # Predicted
    for a in predict_angles:
        pt = predict_hf(name, a)
        ax1.scatter(pt[0], pt[1], pt[2], color=colors_fg[name],
                    s=60, marker='D', edgecolors='gray', zorder=5)

for name in HUMERUS_NAMES:
    pt = predict_hf(name, 180)
    ax1.scatter(pt[0], pt[1], pt[2], color='gray', s=60, marker='^')
    short = name.replace('_epicondyle', '_epi').replace('_shaft', '_sh')
    ax1.text(pt[0], pt[1], pt[2], f" {short}", fontsize=7)

ax1.set_xlabel("X (AP) mm")
ax1.set_ylabel("Y (PD) mm")
ax1.set_zlabel("Z (ML) mm")
ax1.set_title("3D Trajectories\n(Humerus-fixed frame)")
ax1.legend(fontsize=7, loc='upper left')

# ── Panel 2: Sagittal (X-Y) plane ──
ax2 = fig.add_subplot(2, 3, 2)
for name in FOREARM_NAMES:
    traj = np.array([predict_hf(name, a) for a in smooth_angles])
    ax2.plot(traj[:, 0], traj[:, 1], color=colors_fg[name], label=name, lw=2)
    for a in obs_angles:
        pt = humerus_landmarks[a][name]
        ax2.scatter(pt[0], pt[1], color=colors_fg[name], s=80, ec='black', zorder=5)
        ax2.annotate(f"{a:.0f}", (pt[0], pt[1]), fontsize=7,
                     textcoords="offset points", xytext=(5, 5))
    for a in predict_angles:
        pt = predict_hf(name, a)
        ax2.scatter(pt[0], pt[1], color=colors_fg[name], s=50, marker='D',
                    ec='gray', zorder=5)
        ax2.annotate(f"{a:.0f}", (pt[0], pt[1]), fontsize=6, color='gray',
                     textcoords="offset points", xytext=(5, 5))

ax2.scatter(0, 0, color='black', s=120, marker='+', linewidths=2, zorder=10)
ax2.annotate("joint_center", (0, 0), fontsize=7, textcoords="offset points", xytext=(5, -10))
ax2.set_xlabel("X (AP) mm")
ax2.set_ylabel("Y (PD) mm")
ax2.set_title("Sagittal Plane Trajectories\n(circular arcs)")
ax2.legend(fontsize=8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# ── Panel 3: Each coordinate vs angle ──
ax3 = fig.add_subplot(2, 3, 3)
for name in FOREARM_NAMES:
    traj = np.array([predict_hf(name, a) for a in smooth_angles])
    dist_smooth = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2 + traj[:, 2]**2)
    ax3.plot(smooth_angles, dist_smooth, color=colors_fg[name], label=name, lw=2)
    for a in obs_angles:
        d = np.linalg.norm(humerus_landmarks[a][name])
        ax3.scatter(a, d, color=colors_fg[name], s=80, ec='black', zorder=5)

ax3.set_xlabel("Flexion Angle (deg)")
ax3.set_ylabel("Distance from Joint Center (mm)")
ax3.set_title("Landmark Distance vs Flexion")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

# ═════════════════════════════════════════════════════════════════════════
# Step 7: DRR from 180deg volume with predicted 90deg keypoints
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 7: Generating DRR with predicted landmarks")
print("=" * 70)

v180 = volumes[180.0]
v90 = volumes[90.0]

# Transform predicted 90deg humerus-fixed landmarks back to 180deg volume normalised coords
lm_mm_180 = to_mm(v180['lm'], v180['voxel_mm'], v180['vol'].shape)
origin_180, R_180 = build_humerus_frame(lm_mm_180)


def hf_to_norm(hf_pos, origin, R, voxel_mm, vol_shape):
    """Convert humerus-fixed back to normalised volume coords."""
    pos_mm = R.T @ hf_pos + origin
    pd_s, ap_s, ml_s = vol_shape
    return (pos_mm[0] / (pd_s * voxel_mm),
            pos_mm[1] / (ap_s * voxel_mm),
            pos_mm[2] / (ml_s * voxel_mm))


pred_90_norm = {}
for name in ALL_NAMES:
    hf_pos = predict_hf(name, 90.0)
    pred_90_norm[name] = hf_to_norm(hf_pos, origin_180, R_180, v180['voxel_mm'], v180['vol'].shape)

actual_90_norm = v90['lm']

# Generate DRRs
drr_ap_180 = generate_drr(v180['vol'], axis="AP", sid_mm=1000.0, voxel_mm=v180['voxel_mm'])
drr_lat_180 = generate_drr(v180['vol'], axis="LAT", sid_mm=1000.0, voxel_mm=v180['voxel_mm'])
drr_lat_90 = generate_drr(v90['vol'], axis="LAT", sid_mm=1000.0, voxel_mm=v90['voxel_mm'])

# ── Panel 4: 180deg DRR (LAT) with predicted 90deg keypoints ──
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(drr_lat_180, cmap='gray')
H4, W4 = drr_lat_180.shape
ax4.set_title("180deg DRR (LAT)\npred 90deg KP (red) vs actual 180deg KP (blue)")

# Predicted 90deg
for name in KP_ORDER:
    n_PD, n_AP, n_ML = pred_90_norm[name]
    row = n_PD * H4
    col = n_AP * W4   # LAT: AP -> col
    ax4.plot(col, row, 'ro', ms=7, mec='white', mew=0.5)
    ax4.annotate(name.split('_')[0], (col, row), fontsize=5, color='red',
                 textcoords="offset points", xytext=(4, 4))

# Actual 180deg
for name in KP_ORDER:
    n_PD, n_AP, n_ML = v180['lm'][name]
    row = n_PD * H4
    col = n_AP * W4
    ax4.plot(col, row, 'b^', ms=5, mec='white', mew=0.5)

ax4.axis('off')

# ── Panel 5: Actual 90deg DRR (LAT) with actual keypoints ──
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(drr_lat_90, cmap='gray')
H5, W5 = drr_lat_90.shape
ax5.set_title("Actual 90deg DRR (LAT)\nactual 90deg KP (green)")

for name in KP_ORDER:
    n_PD, n_AP, n_ML = actual_90_norm[name]
    row = n_PD * H5
    col = n_AP * W5
    ax5.plot(col, row, 'gs', ms=7, mec='white', mew=0.5)
    ax5.annotate(name.split('_')[0], (col, row), fontsize=5, color='lime',
                 textcoords="offset points", xytext=(4, 4))
ax5.axis('off')


# ═════════════════════════════════════════════════════════════════════════
# Step 8: Error analysis (predicted vs actual at 90deg)
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 8: Prediction error (pred 90deg vs actual 90deg in humerus-fixed)")
print("=" * 70)

# Actual 90deg in its own humerus-fixed frame
lm_mm_90 = to_mm(v90['lm'], v90['voxel_mm'], v90['vol'].shape)
origin_90, R_90 = build_humerus_frame(lm_mm_90)
actual_90_hf = transform_to_frame(lm_mm_90, origin_90, R_90)

errors = {}
print(f"\n  {'Landmark':25s} {'Predicted':>24s} {'Actual':>24s} {'Err mm':>8s}")
print("  " + "-" * 87)
for name in ALL_NAMES:
    pred = predict_hf(name, 90.0)
    actual = actual_90_hf[name]
    err = np.linalg.norm(pred - actual)
    errors[name] = err
    print(f"  {name:25s} ({pred[0]:7.2f},{pred[1]:7.2f},{pred[2]:7.2f}) "
          f"({actual[0]:7.2f},{actual[1]:7.2f},{actual[2]:7.2f}) {err:7.3f}")

mean_err = np.mean(list(errors.values()))
max_err = max(errors.values())
forearm_errs = [errors[n] for n in FOREARM_NAMES]
mean_forearm_err = np.mean(forearm_errs)
hum_errs = [errors[n] for n in HUMERUS_NAMES]
mean_hum_err = np.mean(hum_errs)

print(f"\n  Overall  mean={mean_err:.3f}mm  max={max_err:.3f}mm ({max(errors, key=errors.get)})")
print(f"  Humerus  mean={mean_hum_err:.3f}mm")
print(f"  Forearm  mean={mean_forearm_err:.3f}mm")

# Also compute error in normalised (voxel) space for reference
print(f"\n  Voxel-equivalent errors (at voxel_mm={v90['voxel_mm']:.2f}):")
for name in ALL_NAMES:
    vox_err = errors[name] / v90['voxel_mm']
    print(f"    {name:25s}: {vox_err:.1f} voxels")

# ── Panel 6: Error bar chart ──
ax6 = fig.add_subplot(2, 3, 6)
names_plot = list(errors.keys())
errs_plot = [errors[n] for n in names_plot]
cols = ['tab:orange' if n in FOREARM_NAMES else 'steelblue' for n in names_plot]
bars = ax6.barh(range(len(names_plot)), errs_plot, color=cols, ec='black', alpha=0.85)
ax6.set_yticks(range(len(names_plot)))
ax6.set_yticklabels([n.replace('_', ' ') for n in names_plot], fontsize=9)
ax6.set_xlabel("Error (mm)")
ax6.set_title(f"Predicted vs Actual @ 90deg\n"
              f"mean={mean_err:.1f}mm  forearm={mean_forearm_err:.1f}mm  humerus={mean_hum_err:.1f}mm")
ax6.axvline(mean_err, color='red', ls='--', alpha=0.7, label=f'mean={mean_err:.1f}mm')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='x')
for bar, e in zip(bars, errs_plot):
    ax6.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{e:.1f}', va='center', fontsize=9)

plt.suptitle("Approach B: Parametric Landmark Motion Model\n"
             "3 CT volumes (180/135/90) -- Circular-arc fit in humerus-fixed frame\n"
             f"(blue=humerus fixed, orange=forearm mobile)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PNG}")
print("Done.")
