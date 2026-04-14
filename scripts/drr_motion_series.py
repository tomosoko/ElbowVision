"""
DRR Elbow Flexion Motion Series — 3つのCTボリュームから肘屈伸のフリップブック風DRR系列を生成

3つのCTボリューム（180°/135°/90°）を使い、各ボリュームから中間角度のDRRも生成して
180°→70°の連続的な肘屈曲シリーズを作成する。

使い方:
  cd ~/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/drr_motion_series.py
"""

import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パス設定 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    rotate_volume_and_landmarks,
    generate_drr,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "results/domain_gap_analysis")
MOTION_DIR = os.path.join(OUT_DIR, "motion")
os.makedirs(MOTION_DIR, exist_ok=True)

# ── CT volume definitions ──
CT_CONFIGS = [
    {
        "name": "ct_180",
        "dir": os.path.join(PROJECT_ROOT, "data/raw_dicom/ct_180"),
        "series_num": 3,
        "base_flexion": 180,
        "flexion_angles": [180, 170, 160],
    },
    {
        "name": "ct_135",
        "dir": os.path.join(PROJECT_ROOT, "data/raw_dicom/ct_135"),
        "series_num": 7,
        "base_flexion": 135,
        "flexion_angles": [150, 140, 135, 130, 120],
    },
    {
        "name": "ct_90",
        "dir": os.path.join(PROJECT_ROOT, "data/raw_dicom/ct_90"),
        "series_num": 11,
        "base_flexion": 90,
        "flexion_angles": [110, 100, 90, 80, 70],
    },
]

LATERALITY = "L"
HU_MIN = 50
HU_MAX = 800
TARGET_SIZE = 256


def load_volume(config):
    """Load a single CT volume and detect landmarks."""
    print(f"  Loading {config['name']} (series {config['series_num']})...")
    volume, voxel_spacing, lat, voxel_mm = load_ct_volume(
        config["dir"],
        target_size=TARGET_SIZE,
        laterality=LATERALITY,
        series_num=config["series_num"],
        hu_min=HU_MIN,
        hu_max=HU_MAX,
    )
    landmarks = auto_detect_landmarks(volume)
    print(f"  {config['name']}: shape={volume.shape}, voxel_mm={voxel_mm:.3f}")
    return volume, landmarks, voxel_mm


def generate_single_drr(args):
    """Worker function for parallel DRR generation.

    Args: (volume, landmarks, voxel_mm, flexion_deg, base_flexion, ct_name)
    Returns: (flexion_deg, drr_image, ct_name)
    """
    volume, landmarks, voxel_mm, flexion_deg, base_flexion, ct_name = args

    # Rotate volume to target flexion
    rot_vol, rot_lm = rotate_volume_and_landmarks(
        volume,
        landmarks,
        forearm_rotation_deg=0.0,
        flexion_deg=flexion_deg,
        base_flexion=base_flexion,
    )

    # Generate AP DRR
    drr = generate_drr(rot_vol, axis="AP", voxel_mm=voxel_mm)

    # Resize to target size
    drr = cv2.resize(drr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    return flexion_deg, drr, ct_name


def main():
    t0 = time.time()
    print("=" * 60)
    print("DRR Elbow Flexion Motion Series")
    print("=" * 60)

    # ── Step 1: Load all CT volumes ──
    print("\n[1/3] Loading CT volumes...")
    volumes = {}
    for cfg in CT_CONFIGS:
        vol, lm, vmm = load_volume(cfg)
        volumes[cfg["name"]] = {
            "volume": vol,
            "landmarks": lm,
            "voxel_mm": vmm,
            "base_flexion": cfg["base_flexion"],
            "flexion_angles": cfg["flexion_angles"],
        }

    # ── Step 2: Prepare work items ──
    print("\n[2/3] Generating DRR series with multiprocessing (5 workers)...")
    work_items = []
    for cfg in CT_CONFIGS:
        v = volumes[cfg["name"]]
        for angle in cfg["flexion_angles"]:
            work_items.append((
                v["volume"],
                v["landmarks"],
                v["voxel_mm"],
                angle,
                v["base_flexion"],
                cfg["name"],
            ))

    # Sort by flexion angle (descending: 180 -> 70)
    work_items.sort(key=lambda x: -x[3])

    # Generate DRRs in parallel
    results = {}
    with Pool(processes=5) as pool:
        for flexion_deg, drr, ct_name in pool.map(generate_single_drr, work_items):
            results[flexion_deg] = (drr, ct_name)
            # Save individual DRR
            fname = f"drr_flexion_{flexion_deg:03d}.png"
            cv2.imwrite(os.path.join(MOTION_DIR, fname), drr)
            print(f"    flexion={flexion_deg:3d}° (from {ct_name}) -> {fname}")

    # ── Step 3: Create composite figure ──
    print("\n[3/3] Creating composite figure...")
    # All angles sorted descending (180 -> 70)
    all_angles = sorted(results.keys(), reverse=True)
    n = len(all_angles)

    # Grid layout: try to make roughly rectangular
    n_cols = min(n, 7)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.3))
    fig.suptitle("Elbow Flexion Series (DRR from 3 CT volumes)",
                 fontsize=16, fontweight="bold", y=0.98)

    # Color map for CT source
    ct_colors = {"ct_180": "#e74c3c", "ct_135": "#2ecc71", "ct_90": "#3498db"}
    ct_labels = {"ct_180": "180° CT", "ct_135": "135° CT", "ct_90": "90° CT"}

    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for i, angle in enumerate(all_angles):
        drr, ct_name = results[angle]
        ax = axes_flat[i]
        ax.imshow(drr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{angle}°", fontsize=13, fontweight="bold",
                     color=ct_colors.get(ct_name, "black"))
        ax.set_xlabel(ct_labels.get(ct_name, ct_name), fontsize=9,
                      color=ct_colors.get(ct_name, "gray"))
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=ct_labels[k])
                       for k, c in ct_colors.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "drr_motion_series.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Composite figure saved: {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {n} DRRs generated")
    print(f"  Individual DRRs: {MOTION_DIR}/")
    print(f"  Composite:       {out_path}")


if __name__ == "__main__":
    main()
