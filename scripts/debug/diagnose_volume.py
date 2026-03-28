#!/usr/bin/env python3
"""ボリュームのHU分布と骨閾値を詳細分析"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))
import numpy as np
import cv2
from elbow_synth import load_ct_volume

def main():
    ct_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_dicom', 'ct')
    vol, spacing, lat, voxel_mm = load_ct_volume(ct_dir, target_size=128)
    NP, NA, NM = vol.shape

    print(f"\n=== Volume value distribution (normalized 0-1) ===")
    flat = vol.flatten()
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        print(f"  P{p:3d}: {np.percentile(flat, p):.4f}")

    print(f"\n  Mean: {flat.mean():.4f}, Std: {flat.std():.4f}")
    print(f"  Non-zero fraction: {(flat > 0.01).mean():.3f}")

    # 複数閾値でスライスプロファイルを表示
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n=== Bone profile at different thresholds ===")
    for thresh in thresholds:
        mask = vol > thresh
        areas = np.array([mask[i].sum() for i in range(NP)])
        ml_widths = np.zeros(NP)
        for i in range(NP):
            ml_proj = mask[i].any(axis=0)
            idx = np.where(ml_proj)[0]
            if len(idx) >= 2:
                ml_widths[i] = idx.max() - idx.min()
        bone_slices = (areas > 20).sum()
        max_w = ml_widths.max()
        max_w_idx = ml_widths.argmax()
        unique_areas = len(set(areas.tolist()))
        print(f"\n  Threshold {thresh:.1f}: {bone_slices}/{NP} slices with bone, "
              f"max_ML_width={max_w:.0f} at PD={max_w_idx} ({max_w_idx/NP:.3f}), "
              f"{unique_areas} unique area values")

        # 各スライスの面積（10刻み表示）
        if unique_areas > 3:
            for pd_i in range(0, NP, 16):
                pd_norm = pd_i / NP
                print(f"    PD={pd_i:3d} ({pd_norm:.2f}): area={areas[pd_i]:5.0f}, ML_w={ml_widths[pd_i]:.0f}")

    # 中央スライスの値分布（AP-ML平面）
    mid_pd = NP // 2
    mid_slice = vol[mid_pd]
    print(f"\n=== Mid PD slice (PD={mid_pd}) value stats ===")
    print(f"  Min: {mid_slice.min():.4f}, Max: {mid_slice.max():.4f}")
    print(f"  Mean: {mid_slice.mean():.4f}")
    print(f"  Fraction > 0.3: {(mid_slice > 0.3).mean():.3f}")
    print(f"  Fraction > 0.5: {(mid_slice > 0.5).mean():.3f}")
    print(f"  Fraction > 0.7: {(mid_slice > 0.7).mean():.3f}")

    # スライスの可視化: 5つのPDレベルのaxialスライス
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'projection_verify')
    os.makedirs(out_dir, exist_ok=True)

    pd_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    slices_vis = []
    for pd_norm in pd_levels:
        pd_i = int(pd_norm * NP)
        sl = vol[pd_i]
        sl_u8 = (sl * 255).astype(np.uint8)
        sl_color = cv2.cvtColor(sl_u8, cv2.COLOR_GRAY2BGR)
        # ラベル追加
        cv2.putText(sl_color, f"PD={pd_norm:.1f}", (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        slices_vis.append(sl_color)

    # 横に並べる
    montage = np.hstack(slices_vis)
    montage_big = cv2.resize(montage, (montage.shape[1]*3, montage.shape[0]*3),
                              interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "pd_slices.png"), montage_big)
    print(f"\n  Saved pd_slices.png (5 axial slices at PD 0.1-0.9)")

    # AP方向のMIP（最大値投影）で骨構造を確認
    mip_ap = vol.max(axis=1)  # (PD, ML) - AP方向のMIP
    mip_u8 = (mip_ap * 255).astype(np.uint8)
    mip_big = cv2.resize(mip_u8, (mip_u8.shape[1]*4, mip_u8.shape[0]*4),
                          interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "mip_ap.png"), mip_big)
    print(f"  Saved mip_ap.png (AP-direction MIP, shape PD×ML)")

    # ML方向のMIP
    mip_ml = vol.max(axis=2)  # (PD, AP) - ML方向のMIP
    mip_u8 = (mip_ml * 255).astype(np.uint8)
    mip_big = cv2.resize(mip_u8, (mip_u8.shape[1]*4, mip_u8.shape[0]*4),
                          interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "mip_lat.png"), mip_big)
    print(f"  Saved mip_lat.png (ML-direction MIP, shape PD×AP)")

if __name__ == "__main__":
    main()
