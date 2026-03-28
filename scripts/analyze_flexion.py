#!/usr/bin/env python3
"""
Flexion Analysis: Compare 3 CT volumes (180/135/90) and evaluate synthesis quality.

Loads all 3 FC85 bone kernel series, detects landmarks, compares joint motion,
synthesizes 90 from 180, and computes quantitative metrics (SSIM, Dice, landmark error).
"""

import sys
import os
import math
import numpy as np
import cv2
from pathlib import Path

# Setup paths
PROJECT_ROOT = "/Users/kohei/develop/Dev/vision/ElbowVision"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume, auto_detect_landmarks,
    rotate_volume_and_landmarks, generate_drr,
    compute_flexion_angle
)

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "flexion_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)

CT_DIR = os.path.join(
    PROJECT_ROOT,
    "data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000"
)

# Series mapping: series_num -> (series_name, base_flexion)
SERIES = {
    4:  ("S4_180deg",  180),
    8:  ("S8_135deg",  135),
    12: ("S12_90deg",   90),
}

TARGET_SIZE = 256
HU_MIN = 50
HU_MAX = 1000


def load_all_volumes():
    """Load all 3 CT volumes and detect landmarks."""
    results = {}
    for sn, (name, flexion) in SERIES.items():
        print(f"\n{'='*60}")
        print(f"Loading {name} (Series {sn}, flexion={flexion}deg)")
        print(f"{'='*60}")
        vol, spacing, lat, voxel_mm = load_ct_volume(
            CT_DIR, target_size=TARGET_SIZE, laterality='R',
            series_num=sn, hu_min=HU_MIN, hu_max=HU_MAX
        )
        print(f"  Volume shape: {vol.shape}, voxel_mm: {voxel_mm:.3f}")

        lm = auto_detect_landmarks(vol, laterality='R')

        results[flexion] = {
            'volume': vol,
            'landmarks': lm,
            'spacing': spacing,
            'voxel_mm': voxel_mm,
            'name': name,
            'series_num': sn,
        }
    return results


def compare_landmarks(data):
    """Compare landmarks across 3 volumes."""
    print(f"\n{'='*60}")
    print("LANDMARK COMPARISON ACROSS FLEXION ANGLES")
    print(f"{'='*60}")

    angles = sorted(data.keys(), reverse=True)  # 180, 135, 90
    kp_names = list(data[angles[0]]['landmarks'].keys())

    report_lines = []
    report_lines.append(f"{'Landmark':<22} | {'Coord':>5} | " +
                        " | ".join(f"{a:>8}deg" for a in angles) +
                        " | 180->90 delta")
    report_lines.append("-" * 100)

    for kp in kp_names:
        for ci, cname in enumerate(["PD", "AP", "ML"]):
            vals = [data[a]['landmarks'][kp][ci] for a in angles]
            delta = vals[-1] - vals[0]  # 90 - 180
            line = f"{kp:<22} | {cname:>5} | " + \
                   " | ".join(f"{v:>10.4f}" for v in vals) + \
                   f" | {delta:>+10.4f}"
            report_lines.append(line)
        report_lines.append("")

    report = "\n".join(report_lines)
    print(report)

    # Compute flexion angles from landmarks
    print("\nComputed flexion angles from landmarks:")
    for a in angles:
        fa = compute_flexion_angle(data[a]['landmarks'])
        print(f"  {a}deg volume: computed flexion = {fa:.1f}deg")

    # Analyze rotation axis
    print("\n--- Joint Center Movement ---")
    for a in angles:
        jc = data[a]['landmarks']['joint_center']
        shape = data[a]['volume'].shape
        print(f"  {a}deg: JC = PD={jc[0]:.4f} AP={jc[1]:.4f} ML={jc[2]:.4f} "
              f"(voxel: {jc[0]*shape[0]:.1f}, {jc[1]*shape[1]:.1f}, {jc[2]*shape[2]:.1f})")

    print("\n--- Forearm Shaft Movement ---")
    for a in angles:
        fs = data[a]['landmarks']['forearm_shaft']
        shape = data[a]['volume'].shape
        print(f"  {a}deg: FS = PD={fs[0]:.4f} AP={fs[1]:.4f} ML={fs[2]:.4f} "
              f"(voxel: {fs[0]*shape[0]:.1f}, {fs[1]*shape[1]:.1f}, {fs[2]*shape[2]:.1f})")

    print("\n--- Radial Head Movement ---")
    for a in angles:
        rh = data[a]['landmarks']['radial_head']
        shape = data[a]['volume'].shape
        print(f"  {a}deg: RH = PD={rh[0]:.4f} AP={rh[1]:.4f} ML={rh[2]:.4f} "
              f"(voxel: {rh[0]*shape[0]:.1f}, {rh[1]*shape[1]:.1f}, {rh[2]*shape[2]:.1f})")

    # Estimate rotation axis from 180->90 motion
    print("\n--- Rotation Axis Analysis (180 -> 90) ---")
    jc_180 = np.array(data[180]['landmarks']['joint_center'])
    jc_90 = np.array(data[90]['landmarks']['joint_center'])
    fs_180 = np.array(data[180]['landmarks']['forearm_shaft'])
    fs_90 = np.array(data[90]['landmarks']['forearm_shaft'])

    # The forearm shaft moves in the PD-AP plane (sagittal) around the joint center
    # Motion vector of forearm shaft
    motion = fs_90 - fs_180
    print(f"  Forearm shaft motion vector (PD,AP,ML): ({motion[0]:+.4f}, {motion[1]:+.4f}, {motion[2]:+.4f})")
    print(f"  Primary motion plane: PD-AP (sagittal) as expected for flexion")
    print(f"  ML component magnitude: {abs(motion[2]):.4f} (should be near 0 for pure flexion)")

    return report


def generate_drrs_for_all(data):
    """Generate AP and LAT DRRs for all volumes."""
    print(f"\n{'='*60}")
    print("GENERATING DRRs FOR ALL VOLUMES")
    print(f"{'='*60}")

    for a in sorted(data.keys(), reverse=True):
        vol = data[a]['volume']
        voxel_mm = data[a]['voxel_mm']
        name = data[a]['name']

        for axis in ["AP", "LAT"]:
            drr = generate_drr(vol, axis=axis, sid_mm=1000.0, voxel_mm=voxel_mm)
            path = os.path.join(OUT_DIR, f"drr_{name}_{axis}.png")
            cv2.imwrite(path, drr)
            print(f"  Saved: {path} shape={drr.shape}")
            data[a][f'drr_{axis}'] = drr

    # Side-by-side comparison of all 3 volumes
    for axis in ["AP", "LAT"]:
        imgs = []
        for a in [180, 135, 90]:
            drr = data[a][f'drr_{axis}']
            # Add label
            labeled = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)
            cv2.putText(labeled, f"{a}deg (real)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            imgs.append(labeled)

        # Pad to same height
        max_h = max(im.shape[0] for im in imgs)
        max_w = max(im.shape[1] for im in imgs)
        padded = []
        for im in imgs:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:im.shape[0], :im.shape[1]] = im
            padded.append(pad)

        combined = np.hstack(padded)
        path = os.path.join(OUT_DIR, f"comparison_real_{axis}_180_135_90.png")
        cv2.imwrite(path, combined)
        print(f"  Saved comparison: {path}")


def synthesize_and_compare(data):
    """Synthesize 90deg from 180deg and compare with real 90deg."""
    print(f"\n{'='*60}")
    print("SYNTHESIS: 90deg from 180deg volume")
    print(f"{'='*60}")

    vol_180 = data[180]['volume']
    lm_180 = data[180]['landmarks']
    voxel_mm_180 = data[180]['voxel_mm']

    vol_90_real = data[90]['volume']
    voxel_mm_90 = data[90]['voxel_mm']

    # Synthesize 90deg from 180deg using current rotate_volume_and_landmarks
    print("  Rotating 180deg volume to 90deg (flexion_deg=90, base_flexion=180)...")
    vol_90_synth, lm_90_synth = rotate_volume_and_landmarks(
        vol_180, lm_180, forearm_rotation_deg=0, flexion_deg=90, base_flexion=180
    )
    print(f"  Synthetic volume shape: {vol_90_synth.shape}")

    # Also synthesize 135deg for intermediate comparison
    print("  Rotating 180deg volume to 135deg...")
    vol_135_synth, lm_135_synth = rotate_volume_and_landmarks(
        vol_180, lm_180, forearm_rotation_deg=0, flexion_deg=135, base_flexion=180
    )

    # Generate DRRs
    for axis in ["AP", "LAT"]:
        drr_synth_90 = generate_drr(vol_90_synth, axis=axis, sid_mm=1000.0, voxel_mm=voxel_mm_180)
        drr_real_90 = data[90][f'drr_{axis}']

        drr_synth_135 = generate_drr(vol_135_synth, axis=axis, sid_mm=1000.0, voxel_mm=voxel_mm_180)
        drr_real_135 = data[135][f'drr_{axis}']

        # Save individual synthetic DRRs
        cv2.imwrite(os.path.join(OUT_DIR, f"drr_synth_90from180_{axis}.png"), drr_synth_90)
        cv2.imwrite(os.path.join(OUT_DIR, f"drr_synth_135from180_{axis}.png"), drr_synth_135)

        # Side-by-side: synthetic vs real for 90deg
        def make_comparison(drr_s, drr_r, label_s, label_r, title):
            """Create side-by-side comparison with difference map."""
            # Resize to same shape
            h = max(drr_s.shape[0], drr_r.shape[0])
            w = max(drr_s.shape[1], drr_r.shape[1])
            s_pad = np.zeros((h, w), dtype=np.uint8)
            r_pad = np.zeros((h, w), dtype=np.uint8)
            s_pad[:drr_s.shape[0], :drr_s.shape[1]] = drr_s
            r_pad[:drr_r.shape[0], :drr_r.shape[1]] = drr_r

            # Difference map (absolute)
            diff = cv2.absdiff(s_pad, r_pad)
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

            # Label images
            s_bgr = cv2.cvtColor(s_pad, cv2.COLOR_GRAY2BGR)
            r_bgr = cv2.cvtColor(r_pad, cv2.COLOR_GRAY2BGR)

            cv2.putText(s_bgr, label_s, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(r_bgr, label_r, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(diff_color, "Difference", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            combined = np.hstack([s_bgr, r_bgr, diff_color])
            return combined, s_pad, r_pad

        comp_90, s90, r90 = make_comparison(
            drr_synth_90, drr_real_90,
            "Synthetic 90 (from 180)", "Real 90",
            f"90deg {axis}"
        )
        path = os.path.join(OUT_DIR, f"synth_vs_real_90deg_{axis}.png")
        cv2.imwrite(path, comp_90)
        print(f"  Saved: {path}")

        comp_135, s135, r135 = make_comparison(
            drr_synth_135, drr_real_135,
            "Synthetic 135 (from 180)", "Real 135",
            f"135deg {axis}"
        )
        path = os.path.join(OUT_DIR, f"synth_vs_real_135deg_{axis}.png")
        cv2.imwrite(path, comp_135)
        print(f"  Saved: {path}")

        # Store for metrics
        data[f'synth_90_{axis}'] = s90
        data[f'real_90_{axis}'] = r90
        data[f'synth_135_{axis}'] = s135
        data[f'real_135_{axis}'] = r135

    # Save synthetic landmarks
    data['lm_90_synth'] = lm_90_synth
    data['lm_135_synth'] = lm_135_synth

    return data


def compute_metrics(data):
    """Compute SSIM, Dice, and landmark error between synthetic and real."""
    from skimage.metrics import structural_similarity as ssim

    print(f"\n{'='*60}")
    print("QUANTITATIVE METRICS")
    print(f"{'='*60}")

    results = {}

    for angle, label in [(90, "90deg"), (135, "135deg")]:
        print(f"\n--- {label}: Synthetic (from 180) vs Real ---")

        for axis in ["AP", "LAT"]:
            s = data[f'synth_{angle}_{axis}'].astype(np.float64)
            r = data[f'real_{angle}_{axis}'].astype(np.float64)

            # SSIM
            ssim_val = ssim(s, r, data_range=255.0)
            print(f"  {axis} SSIM: {ssim_val:.4f}")

            # Dice on bone (threshold at mean + 1 std of nonzero pixels)
            def bone_mask(img, name=""):
                # Use Otsu threshold
                img_u8 = img.astype(np.uint8)
                _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return mask > 0

            mask_s = bone_mask(s, "synth")
            mask_r = bone_mask(r, "real")

            intersection = np.logical_and(mask_s, mask_r).sum()
            union = mask_s.sum() + mask_r.sum()
            dice = 2.0 * intersection / (union + 1e-8)
            print(f"  {axis} Bone Dice: {dice:.4f}")

            # Save bone contour overlay
            overlay = np.zeros((*s.shape[:2], 3), dtype=np.uint8)
            overlay[mask_s, 1] = 200  # Green = synthetic
            overlay[mask_r, 2] = 200  # Red = real
            overlay[np.logical_and(mask_s, mask_r)] = [0, 200, 200]  # Yellow = overlap
            path = os.path.join(OUT_DIR, f"bone_overlay_{label}_{axis}.png")
            cv2.imwrite(path, overlay)
            print(f"  Saved bone overlay: {path}")

            results[f'{label}_{axis}_ssim'] = ssim_val
            results[f'{label}_{axis}_dice'] = dice

    # Landmark position difference
    print(f"\n--- Landmark Position Difference (normalized coords) ---")

    lm_synth_key = 'lm_90_synth'
    lm_real = data[90]['landmarks']
    lm_synth = data[lm_synth_key]

    vol_shape = data[90]['volume'].shape
    voxel_mm = data[90]['voxel_mm']

    print(f"  Volume shape: {vol_shape}, voxel_mm: {voxel_mm:.3f}")
    print(f"\n  {'Landmark':<22} | {'Synth (PD,AP,ML)':<30} | {'Real (PD,AP,ML)':<30} | {'Dist(norm)':<10} | {'Dist(mm)':<10}")
    print("  " + "-" * 115)

    for kp in sorted(lm_real.keys()):
        if kp not in lm_synth:
            continue
        s = np.array(lm_synth[kp])
        r = np.array(lm_real[kp])
        diff = s - r
        dist_norm = np.linalg.norm(diff)
        # Convert to mm: multiply normalized distance by volume size * voxel_mm
        dist_voxel = np.linalg.norm(diff * np.array(vol_shape))
        dist_mm = dist_voxel * voxel_mm

        print(f"  {kp:<22} | ({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}){'':<14} | "
              f"({r[0]:.3f},{r[1]:.3f},{r[2]:.3f}){'':<14} | {dist_norm:<10.4f} | {dist_mm:<10.1f}mm")

        results[f'lm_{kp}_dist_mm'] = dist_mm

    return results


def save_volume_slices(data):
    """Save axial/sagittal/coronal slices for visual comparison of bone structure."""
    print(f"\n{'='*60}")
    print("SAVING VOLUME SLICE COMPARISONS")
    print(f"{'='*60}")

    for view_name, axis_idx in [("axial_PD", 0), ("sagittal_ML", 2), ("coronal_AP", 1)]:
        imgs = []
        for a in [180, 135, 90]:
            vol = data[a]['volume']
            mid = vol.shape[axis_idx] // 2
            if axis_idx == 0:
                sl = vol[mid, :, :]
            elif axis_idx == 1:
                sl = vol[:, mid, :]
            else:
                sl = vol[:, :, mid]

            sl_u8 = (sl * 255).astype(np.uint8)
            labeled = cv2.cvtColor(sl_u8, cv2.COLOR_GRAY2BGR)
            cv2.putText(labeled, f"{a}deg", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            imgs.append(labeled)

        # Also add synthetic 90 from 180
        vol_180 = data[180]['volume']
        lm_180 = data[180]['landmarks']
        vol_synth, _ = rotate_volume_and_landmarks(
            vol_180, lm_180, 0, 90, base_flexion=180
        )
        mid = vol_synth.shape[axis_idx] // 2
        if axis_idx == 0:
            sl = vol_synth[mid, :, :]
        elif axis_idx == 1:
            sl = vol_synth[:, mid, :]
        else:
            sl = vol_synth[:, :, mid]
        sl_u8 = (sl * 255).astype(np.uint8)
        labeled = cv2.cvtColor(sl_u8, cv2.COLOR_GRAY2BGR)
        cv2.putText(labeled, "Synth90", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        imgs.append(labeled)

        max_h = max(im.shape[0] for im in imgs)
        max_w = max(im.shape[1] for im in imgs)
        padded = []
        for im in imgs:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:im.shape[0], :im.shape[1]] = im
            padded.append(pad)

        combined = np.hstack(padded)
        path = os.path.join(OUT_DIR, f"slices_{view_name}_comparison.png")
        cv2.imwrite(path, combined)
        print(f"  Saved: {path}")


def write_report(landmark_report, metrics, data):
    """Write final analysis report."""
    path = os.path.join(OUT_DIR, "analysis_report.txt")

    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ELBOW FLEXION ANALYSIS REPORT\n")
        f.write("CT: 3-position phantom (180/135/90 deg)\n")
        f.write(f"Target size: {TARGET_SIZE}, HU window: {HU_MIN}-{HU_MAX}\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. LANDMARK POSITIONS ACROSS FLEXION ANGLES\n")
        f.write("-" * 50 + "\n")
        f.write(landmark_report + "\n\n")

        f.write("2. COMPUTED FLEXION ANGLES\n")
        f.write("-" * 50 + "\n")
        for a in [180, 135, 90]:
            fa = compute_flexion_angle(data[a]['landmarks'])
            f.write(f"  {a}deg volume: computed = {fa:.1f}deg\n")
        f.write("\n")

        f.write("3. JOINT CENTER POSITIONS\n")
        f.write("-" * 50 + "\n")
        for a in [180, 135, 90]:
            jc = data[a]['landmarks']['joint_center']
            shape = data[a]['volume'].shape
            f.write(f"  {a}deg: PD={jc[0]:.4f} AP={jc[1]:.4f} ML={jc[2]:.4f} "
                    f"(voxel: {jc[0]*shape[0]:.1f}, {jc[1]*shape[1]:.1f}, {jc[2]*shape[2]:.1f})\n")
        f.write("\n")

        f.write("4. SYNTHESIS QUALITY METRICS\n")
        f.write("-" * 50 + "\n")
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if 'ssim' in key or 'dice' in key:
                f.write(f"  {key}: {val:.4f}\n")
            elif 'dist_mm' in key:
                f.write(f"  {key}: {val:.1f} mm\n")
        f.write("\n")

        # Summary assessment
        f.write("5. SUMMARY\n")
        f.write("-" * 50 + "\n")

        ssim_90_lat = metrics.get('90deg_LAT_ssim', 0)
        dice_90_lat = metrics.get('90deg_LAT_dice', 0)
        ssim_90_ap = metrics.get('90deg_AP_ssim', 0)
        dice_90_ap = metrics.get('90deg_AP_dice', 0)

        f.write(f"\n  90deg synthesis (from 180deg):\n")
        f.write(f"    LAT: SSIM={ssim_90_lat:.4f}, Dice={dice_90_lat:.4f}\n")
        f.write(f"    AP:  SSIM={ssim_90_ap:.4f}, Dice={dice_90_ap:.4f}\n")

        ssim_135_lat = metrics.get('135deg_LAT_ssim', 0)
        dice_135_lat = metrics.get('135deg_LAT_dice', 0)
        ssim_135_ap = metrics.get('135deg_AP_ssim', 0)
        dice_135_ap = metrics.get('135deg_AP_dice', 0)

        f.write(f"\n  135deg synthesis (from 180deg):\n")
        f.write(f"    LAT: SSIM={ssim_135_lat:.4f}, Dice={dice_135_lat:.4f}\n")
        f.write(f"    AP:  SSIM={ssim_135_ap:.4f}, Dice={dice_135_ap:.4f}\n")

        if ssim_90_lat > 0.7:
            f.write("\n  Assessment: GOOD - LAT synthesis produces usable DRRs\n")
        elif ssim_90_lat > 0.5:
            f.write("\n  Assessment: MODERATE - visible differences but overall structure preserved\n")
        else:
            f.write("\n  Assessment: POOR - significant artifacts, rotation model needs improvement\n")

        f.write("\n  Key limitations of rigid rotation model:\n")
        f.write("    - Real elbow flexion involves complex joint kinematics\n")
        f.write("    - Bone overlap at joint changes with flexion angle\n")
        f.write("    - Carrying angle and valgus shift during flexion\n")
        f.write("    - Soft tissue deformation not modeled\n")

    print(f"\n  Report saved: {path}")


def main():
    print("=" * 70)
    print("ELBOW FLEXION ANALYSIS")
    print("=" * 70)

    # Step 1: Load all volumes
    data = load_all_volumes()

    # Step 2: Compare landmarks
    landmark_report = compare_landmarks(data)

    # Step 3: Generate DRRs for all real volumes
    generate_drrs_for_all(data)

    # Step 4: Save volume slice comparisons
    save_volume_slices(data)

    # Step 5: Synthesize and compare
    data = synthesize_and_compare(data)

    # Step 6: Compute metrics
    metrics = compute_metrics(data)

    # Step 7: Write report
    write_report(landmark_report, metrics, data)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
