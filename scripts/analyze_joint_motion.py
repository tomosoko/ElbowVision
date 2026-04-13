"""
analyze_joint_motion.py -- 3つの屈曲角度CTから関節運動を解析する

同一ファントムの3ポジション（~180, ~135, ~90 度）CT から
ランドマークを自動検出し、関節運動の特性を可視化・報告する。

Usage:
  cd /Users/kohei/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/analyze_joint_motion.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, 'elbow-train')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from elbow_synth import load_ct_volume, auto_detect_landmarks

# ── Configuration ────────────────────────────────────────────────────────────

CT_DIRS = [
    ("data/raw_dicom/ct_180", "~180 deg (extended)"),
    ("data/raw_dicom/ct_135", "~135 deg"),
    ("data/raw_dicom/ct_90",  "~90 deg (flexed)"),
]
NOMINAL_ANGLES = [180, 135, 90]

LATERALITY = 'L'
HU_MIN = 50
HU_MAX = 800
TARGET_SIZE = 256

OUT_DIR = "results/domain_gap_analysis"
OUT_FIG = os.path.join(OUT_DIR, "joint_motion.png")
OUT_REPORT = os.path.join(OUT_DIR, "joint_motion_report.txt")

LANDMARK_NAMES = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "joint_center",
    "forearm_shaft",
    "radial_head",
    "olecranon",
]

LANDMARK_COLORS = {
    "humerus_shaft":      "#1f77b4",
    "lateral_epicondyle": "#ff7f0e",
    "medial_epicondyle":  "#2ca02c",
    "joint_center":       "#d62728",
    "forearm_shaft":      "#9467bd",
    "radial_head":        "#8c564b",
    "olecranon":          "#e377c2",
}

VOLUME_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ── Helper functions ─────────────────────────────────────────────────────────

def calc_flexion_angle(humerus_shaft, joint_center, forearm_shaft):
    """
    Calculate flexion angle from 3D landmark coordinates (PD, AP, ML).
    Uses the angle at joint_center between humerus_shaft and forearm_shaft.
    Returns angle in degrees.
    """
    h = np.array(humerus_shaft)
    j = np.array(joint_center)
    f = np.array(forearm_shaft)
    v1 = h - j
    v2 = f - j
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def bone_outline_2d(volume, axis, bone_thresh=0.15):
    """
    Project volume along an axis to get a 2D bone outline.
    axis=1 -> AP projection (PD x ML image)
    axis=2 -> ML projection (PD x AP image)
    Returns binary mask (rows=PD, cols=ML or AP).
    """
    proj = volume.max(axis=axis)
    return proj > bone_thresh


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all 3 volumes and detect landmarks
    volumes = []
    all_landmarks = []
    voxel_mms = []

    for (ct_dir, label), nom_angle in zip(CT_DIRS, NOMINAL_ANGLES):
        print(f"\n{'='*60}")
        print(f"Loading {label} from {ct_dir} ...")
        print(f"{'='*60}")
        vol, spacing, lat, vox_mm = load_ct_volume(
            ct_dir, target_size=TARGET_SIZE, laterality=LATERALITY,
            hu_min=HU_MIN, hu_max=HU_MAX,
        )
        print(f"  Volume shape: {vol.shape}, voxel_mm: {vox_mm:.3f}")
        lm = auto_detect_landmarks(vol, laterality=lat)
        volumes.append(vol)
        all_landmarks.append(lm)
        voxel_mms.append(vox_mm)
        print(f"  Landmarks detected for {label}")

    # ── Calculate flexion angles ─────────────────────────────────────────────
    measured_angles = []
    for i, lm in enumerate(all_landmarks):
        angle = calc_flexion_angle(
            lm["humerus_shaft"], lm["joint_center"], lm["forearm_shaft"]
        )
        measured_angles.append(angle)
        print(f"\n{CT_DIRS[i][1]}: measured flexion angle = {angle:.1f} deg (nominal {NOMINAL_ANGLES[i]} deg)")

    # ── Compute landmark trajectories (in voxel-normalized coords) ───────────
    # Convert normalized coords to mm for meaningful comparison
    # Use volume shape * voxel_mm for physical coordinates
    landmarks_mm = []
    for i, lm in enumerate(all_landmarks):
        shape = np.array(volumes[i].shape, dtype=float)
        vmm = voxel_mms[i]
        lm_mm = {}
        for name in LANDMARK_NAMES:
            norm = np.array(lm[name])
            lm_mm[name] = norm * shape * vmm  # physical mm
        landmarks_mm.append(lm_mm)

    # ── Analyze motion relative to humerus shaft ─────────────────────────────
    # Align all landmarks relative to humerus_shaft position for comparison
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Joint Motion Analysis Report")
    report_lines.append("Elbow phantom at 3 flexion angles (~180, ~135, ~90 deg)")
    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append("1. Measured flexion angles (humerus-joint-forearm angle)")
    report_lines.append("-" * 50)
    for i in range(3):
        report_lines.append(
            f"  {CT_DIRS[i][1]:30s}: {measured_angles[i]:6.1f} deg "
            f"(nominal {NOMINAL_ANGLES[i]} deg, diff {measured_angles[i] - NOMINAL_ANGLES[i]:+.1f} deg)"
        )
    report_lines.append("")

    report_lines.append("2. Landmark positions (normalized 0-1, PD/AP/ML)")
    report_lines.append("-" * 50)
    for i in range(3):
        report_lines.append(f"  [{CT_DIRS[i][1]}]")
        for name in LANDMARK_NAMES:
            pd, ap, ml = all_landmarks[i][name]
            report_lines.append(f"    {name:25s}: PD={pd:.3f}  AP={ap:.3f}  ML={ml:.3f}")
        report_lines.append("")

    report_lines.append("3. Joint center motion relative to humerus shaft")
    report_lines.append("-" * 50)
    for i in range(3):
        jc = np.array(all_landmarks[i]["joint_center"])
        hs = np.array(all_landmarks[i]["humerus_shaft"])
        delta = jc - hs
        shape = np.array(volumes[i].shape, dtype=float)
        delta_vox = delta * shape
        delta_mm = delta_vox * voxel_mms[i]
        report_lines.append(
            f"  {CT_DIRS[i][1]:30s}: "
            f"dPD={delta_mm[0]:+6.1f}mm  dAP={delta_mm[1]:+6.1f}mm  dML={delta_mm[2]:+6.1f}mm"
        )
    report_lines.append("")

    report_lines.append("4. Epicondyle separation (lateral - medial) in ML direction")
    report_lines.append("-" * 50)
    for i in range(3):
        lat_ml = all_landmarks[i]["lateral_epicondyle"][2]
        med_ml = all_landmarks[i]["medial_epicondyle"][2]
        sep_norm = lat_ml - med_ml
        sep_mm = sep_norm * volumes[i].shape[2] * voxel_mms[i]
        report_lines.append(
            f"  {CT_DIRS[i][1]:30s}: separation = {sep_mm:.1f} mm  "
            f"(lat ML={lat_ml:.3f}, med ML={med_ml:.3f})"
        )
    report_lines.append("")

    report_lines.append("5. Forearm shaft trajectory (movement during flexion)")
    report_lines.append("-" * 50)
    for i in range(3):
        fs = all_landmarks[i]["forearm_shaft"]
        shape = np.array(volumes[i].shape, dtype=float)
        fs_mm = np.array(fs) * shape * voxel_mms[i]
        report_lines.append(
            f"  {CT_DIRS[i][1]:30s}: "
            f"PD={fs_mm[0]:.1f}mm  AP={fs_mm[1]:.1f}mm  ML={fs_mm[2]:.1f}mm"
        )
    # Show relative displacement from 180 to 90
    if len(all_landmarks) == 3:
        fs180 = np.array(all_landmarks[0]["forearm_shaft"])
        fs90 = np.array(all_landmarks[2]["forearm_shaft"])
        # Use average shape/voxel for approximate comparison
        avg_shape = np.mean([np.array(v.shape, dtype=float) for v in volumes], axis=0)
        avg_vmm = np.mean(voxel_mms)
        delta_mm = (fs90 - fs180) * avg_shape * avg_vmm
        report_lines.append(
            f"  180->90 displacement (approx):  "
            f"dPD={delta_mm[0]:+.1f}mm  dAP={delta_mm[1]:+.1f}mm  dML={delta_mm[2]:+.1f}mm"
        )
    report_lines.append("")

    report_lines.append("6. Olecranon and radial head movement")
    report_lines.append("-" * 50)
    for name in ["olecranon", "radial_head"]:
        report_lines.append(f"  [{name}]")
        for i in range(3):
            lm_pos = all_landmarks[i][name]
            shape = np.array(volumes[i].shape, dtype=float)
            pos_mm = np.array(lm_pos) * shape * voxel_mms[i]
            report_lines.append(
                f"    {CT_DIRS[i][1]:30s}: "
                f"PD={pos_mm[0]:.1f}mm  AP={pos_mm[1]:.1f}mm  ML={pos_mm[2]:.1f}mm"
            )
    report_lines.append("")

    report_lines.append("7. Implications for rotation model in elbow_synth.py")
    report_lines.append("-" * 50)
    # Check if joint center translates relative to humerus during flexion
    jc_rel = []
    for i in range(3):
        jc = np.array(all_landmarks[i]["joint_center"])
        hs = np.array(all_landmarks[i]["humerus_shaft"])
        shape = np.array(volumes[i].shape, dtype=float)
        jc_rel.append((jc - hs) * shape * voxel_mms[i])

    jc_rel = np.array(jc_rel)
    jc_range = jc_rel.max(axis=0) - jc_rel.min(axis=0)
    report_lines.append(
        f"  Joint center displacement range (relative to humerus shaft):"
    )
    report_lines.append(
        f"    PD: {jc_range[0]:.1f}mm   AP: {jc_range[1]:.1f}mm   ML: {jc_range[2]:.1f}mm"
    )
    if jc_range.max() > 3.0:
        report_lines.append(
            "  -> Joint center translates significantly during flexion."
        )
        report_lines.append(
            "     Consider adding translation component to the rotation model."
        )
    else:
        report_lines.append(
            "  -> Joint center is relatively stable (pure rotation model may suffice)."
        )
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(OUT_REPORT, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {OUT_REPORT}")

    # ── Visualization ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Elbow Joint Motion Analysis (3 flexion angles)", fontsize=16, y=0.98)

    # Layout: 2 rows x 3 cols
    # Row 1: Bone outlines superimposed (AP view, LAT view, 3D scatter)
    # Row 2: Landmark trajectories (PD-ML, PD-AP, angle comparison)

    # --- Row 1, Col 1: AP projection (PD x ML) bone outlines ---
    ax1 = fig.add_subplot(2, 3, 1)
    for i, vol in enumerate(volumes):
        outline = bone_outline_2d(vol, axis=1)  # AP projection -> (PD, ML)
        from scipy.ndimage import binary_dilation, binary_erosion
        edge = binary_dilation(outline) ^ outline
        pd_idx, ml_idx = np.where(edge)
        ax1.scatter(ml_idx, pd_idx, s=0.3, alpha=0.5, color=VOLUME_COLORS[i],
                   label=CT_DIRS[i][1])
    ax1.set_xlabel("ML (medial -> lateral)")
    ax1.set_ylabel("PD (proximal -> distal)")
    ax1.set_title("AP projection (bone outlines)")
    ax1.legend(fontsize=8, markerscale=10)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')

    # --- Row 1, Col 2: LAT projection (PD x AP) bone outlines ---
    ax2 = fig.add_subplot(2, 3, 2)
    for i, vol in enumerate(volumes):
        outline = bone_outline_2d(vol, axis=2)  # ML projection -> (PD, AP)
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(outline) ^ outline
        pd_idx, ap_idx = np.where(edge)
        ax2.scatter(ap_idx, pd_idx, s=0.3, alpha=0.5, color=VOLUME_COLORS[i],
                   label=CT_DIRS[i][1])
    ax2.set_xlabel("AP (anterior -> posterior)")
    ax2.set_ylabel("PD (proximal -> distal)")
    ax2.set_title("LAT projection (bone outlines)")
    ax2.legend(fontsize=8, markerscale=10)
    ax2.invert_yaxis()
    ax2.set_aspect('equal')

    # --- Row 1, Col 3: Measured vs nominal angles ---
    ax3 = fig.add_subplot(2, 3, 3)
    x_pos = np.arange(3)
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, NOMINAL_ANGLES, width, label='Nominal',
                    color='lightblue', edgecolor='navy')
    bars2 = ax3.bar(x_pos + width/2, measured_angles, width, label='Measured',
                    color='salmon', edgecolor='darkred')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['180', '135', '90'])
    ax3.set_xlabel("Nominal flexion angle (deg)")
    ax3.set_ylabel("Angle (deg)")
    ax3.set_title("Nominal vs Measured flexion angle")
    ax3.legend()
    for bar, val in zip(bars2, measured_angles):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # --- Row 2, Col 1: Landmark trajectories in PD-ML plane (AP view) ---
    ax4 = fig.add_subplot(2, 3, 4)
    for name in LANDMARK_NAMES:
        pds = [all_landmarks[i][name][0] for i in range(3)]
        mls = [all_landmarks[i][name][2] for i in range(3)]
        ax4.plot(mls, pds, 'o-', color=LANDMARK_COLORS[name], label=name,
                markersize=6, linewidth=1.5)
        # Arrow from 180 to 90
        ax4.annotate('', xy=(mls[2], pds[2]), xytext=(mls[0], pds[0]),
                    arrowprops=dict(arrowstyle='->', color=LANDMARK_COLORS[name],
                                   alpha=0.4, lw=1.5))
    ax4.set_xlabel("ML (normalized, medial -> lateral)")
    ax4.set_ylabel("PD (normalized, proximal -> distal)")
    ax4.set_title("Landmark trajectories (PD-ML / AP view)")
    ax4.legend(fontsize=7, loc='upper left')
    ax4.invert_yaxis()

    # --- Row 2, Col 2: Landmark trajectories in PD-AP plane (LAT view) ---
    ax5 = fig.add_subplot(2, 3, 5)
    for name in LANDMARK_NAMES:
        pds = [all_landmarks[i][name][0] for i in range(3)]
        aps = [all_landmarks[i][name][1] for i in range(3)]
        ax5.plot(aps, pds, 'o-', color=LANDMARK_COLORS[name], label=name,
                markersize=6, linewidth=1.5)
        ax5.annotate('', xy=(aps[2], pds[2]), xytext=(aps[0], pds[0]),
                    arrowprops=dict(arrowstyle='->', color=LANDMARK_COLORS[name],
                                   alpha=0.4, lw=1.5))
    ax5.set_xlabel("AP (normalized, anterior -> posterior)")
    ax5.set_ylabel("PD (normalized, proximal -> distal)")
    ax5.set_title("Landmark trajectories (PD-AP / LAT view)")
    ax5.legend(fontsize=7, loc='upper left')
    ax5.invert_yaxis()

    # --- Row 2, Col 3: Schematic of the 3 arm positions ---
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(3):
        lm = all_landmarks[i]
        hs = lm["humerus_shaft"]
        jc = lm["joint_center"]
        fs = lm["forearm_shaft"]
        # Plot in PD-AP plane (lateral view shows flexion best)
        # Humerus line: hs -> jc
        ax6.plot([hs[1], jc[1]], [hs[0], jc[0]], '-', color=VOLUME_COLORS[i],
                linewidth=3, solid_capstyle='round')
        # Forearm line: jc -> fs
        ax6.plot([jc[1], fs[1]], [jc[0], fs[0]], '--', color=VOLUME_COLORS[i],
                linewidth=3, solid_capstyle='round')
        # Joint center marker
        ax6.plot(jc[1], jc[0], 'o', color=VOLUME_COLORS[i], markersize=10,
                markeredgecolor='black', markeredgewidth=1)
        # Angle annotation
        ax6.annotate(f"{measured_angles[i]:.0f} deg",
                    xy=(jc[1], jc[0]),
                    xytext=(jc[1] + 0.05 + i*0.03, jc[0] - 0.02 - i*0.03),
                    fontsize=9, color=VOLUME_COLORS[i],
                    arrowprops=dict(arrowstyle='->', color=VOLUME_COLORS[i], alpha=0.6))
    ax6.set_xlabel("AP (normalized)")
    ax6.set_ylabel("PD (normalized)")
    ax6.set_title("Arm positions (PD-AP schematic)")
    ax6.invert_yaxis()
    # Legend
    for i in range(3):
        ax6.plot([], [], '-', color=VOLUME_COLORS[i], linewidth=2,
                label=f"{CT_DIRS[i][1]} ({measured_angles[i]:.0f} deg)")
    ax6.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {OUT_FIG}")
    plt.close()


if __name__ == "__main__":
    main()
