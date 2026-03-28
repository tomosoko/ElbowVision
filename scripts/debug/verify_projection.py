#!/usr/bin/env python3
"""
DRR投影座標の検証スクリプト。
CTボリュームからDRR生成→ランドマーク検出→投影→可視化して正確性を確認。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))

import numpy as np
import cv2
from elbow_synth import (
    load_ct_volume, auto_detect_landmarks, generate_drr,
    make_yolo_label, _project_kp_perspective,
)

def main():
    ct_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_dicom', 'ct')
    if not os.path.isdir(ct_dir):
        print(f"CT directory not found: {ct_dir}")
        return

    print("=== Loading CT volume ===")
    vol, spacing, lat, voxel_mm = load_ct_volume(ct_dir, target_size=128)
    print(f"  Volume shape (PD,AP,ML): {vol.shape}")
    print(f"  Voxel mm: {voxel_mm:.2f}")
    print(f"  Laterality: {lat}")

    print("\n=== Auto-detecting landmarks ===")
    lm = auto_detect_landmarks(vol, laterality=lat)

    sid_mm = 1000.0
    NP, NA, NM = vol.shape
    SID_vox = sid_mm / voxel_mm

    print(f"\n=== Projection parameters ===")
    print(f"  SID_vox = {SID_vox:.1f}")
    print(f"  AP: D_s = {max(SID_vox - NA, 1.0):.1f}, mag_center = {SID_vox / (max(SID_vox - NA, 1.0) + 0.5 * NA):.3f}")
    print(f"  LAT: D_s = {max(SID_vox - NM, 1.0):.1f}, mag_center = {SID_vox / (max(SID_vox - NM, 1.0) + 0.5 * NM):.3f}")

    for axis in ["AP", "LAT"]:
        print(f"\n=== {axis} DRR Generation ===")
        drr = generate_drr(vol, axis=axis, sid_mm=sid_mm, voxel_mm=voxel_mm)
        print(f"  DRR shape: {drr.shape}")

        # Generate YOLO label
        label = make_yolo_label(lm, axis, drr.shape[0], drr.shape[1],
                                vol_shape=vol.shape, sid_mm=sid_mm, voxel_mm=voxel_mm)
        parts = label.split()
        n_fields = len(parts)
        n_kp = (n_fields - 5) // 3
        print(f"  Label fields: {n_fields} ({n_kp} keypoints)")

        # Parse keypoints
        kp_names = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                     "forearm_shaft", "radial_head", "olecranon"]
        kp_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]

        img_h, img_w = drr.shape
        drr_vis = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)

        # Also compute raw projection (without clamping) for diagnostics
        print(f"\n  Keypoint projections ({axis}):")
        print(f"  {'name':<22s} {'3D(PD,AP,ML)':<30s} {'2D(px,py)':<20s} {'pixel(x,y)':<20s} in_bounds")

        if axis == "AP":
            D_s = max(SID_vox - NA, 1.0)
        else:
            D_s = max(SID_vox - NM, 1.0)

        for i, name in enumerate(kp_names):
            n_PD, n_AP, n_ML = lm[name]
            if axis == "AP":
                mag = SID_vox / max(D_s + n_AP * NA, 1e-6)
                px_raw = 0.5 + (n_ML - 0.5) * mag
                py_raw = 0.5 + (n_PD - 0.5) * mag
            else:
                mag = SID_vox / max(D_s + n_ML * NM, 1e-6)
                px_raw = 0.5 + (n_AP - 0.5) * mag
                py_raw = 0.5 + (n_PD - 0.5) * mag

            px_clamp = max(0.0, min(1.0, px_raw))
            py_clamp = max(0.0, min(1.0, py_raw))
            in_bounds = 0.0 <= px_raw <= 1.0 and 0.0 <= py_raw <= 1.0

            pix_x = int(px_clamp * img_w)
            pix_y = int(py_clamp * img_h)

            print(f"  {name:<22s} ({n_PD:.3f},{n_AP:.3f},{n_ML:.3f})  "
                  f"({px_raw:.4f},{py_raw:.4f})  "
                  f"({pix_x:3d},{pix_y:3d})  {'OK' if in_bounds else 'OUT!'}")

            # Draw on visualization
            color = kp_colors[i % len(kp_colors)]
            cv2.circle(drr_vis, (pix_x, pix_y), 2, color, -1)
            cv2.putText(drr_vis, name[:3], (pix_x+3, pix_y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        # Save visualization
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'projection_verify')
        os.makedirs(out_dir, exist_ok=True)
        # Upscale 4x for visibility
        drr_vis_big = cv2.resize(drr_vis, (img_w*4, img_h*4), interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(out_dir, f"verify_{axis}.png")
        cv2.imwrite(out_path, drr_vis_big)
        print(f"\n  Saved: {out_path}")

if __name__ == "__main__":
    main()
