#!/usr/bin/env python3
"""
投影のズレの根本原因を特定する診断スクリプト。
ボリュームの骨プロファイルとDRR上の骨位置を比較し、PD軸の向きを検証する。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))

import numpy as np
import cv2
from elbow_synth import load_ct_volume, auto_detect_landmarks, generate_drr

def main():
    ct_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_dicom', 'ct')
    vol, spacing, lat, voxel_mm = load_ct_volume(ct_dir, target_size=128)
    lm = auto_detect_landmarks(vol, laterality=lat)

    NP, NA, NM = vol.shape
    print(f"\n=== Volume shape: {vol.shape}, voxel_mm: {voxel_mm:.2f} ===")

    # 1. PD軸沿いの骨プロファイル
    # Otsu閾値で骨マスク作成
    from elbow_synth import _project_kp_perspective
    flat = vol.flatten()
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    w0, best_var, best_thresh = 0.0, 0.0, 0.5
    mu_total = float((hist * bin_centers).sum() / total)
    for i in range(len(hist)):
        w0 += hist[i] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0_val = float((hist[:i+1] * bin_centers[:i+1]).sum()) / (w0 * total) if w0 * total > 0 else 0
        mu1_val = float((hist[i+1:] * bin_centers[i+1:]).sum()) / (w1 * total) if w1 * total > 0 else 0
        var = w0 * w1 * (mu0_val - mu1_val) ** 2
        if var > best_var:
            best_var, best_thresh = var, bin_centers[i]
    bone_mask = vol > best_thresh

    # 各PDスライスの骨面積とML幅
    print(f"\n=== PD axis bone profile ===")
    print(f"PD_idx  PD_norm  bone_area  ML_width  ML_center  bone_present")
    ml_widths = np.zeros(NP)
    bone_areas = np.zeros(NP)
    for pd_i in range(NP):
        sl = bone_mask[pd_i]
        bone_areas[pd_i] = sl.sum()
        ml_proj = sl.any(axis=0)
        ml_idx = np.where(ml_proj)[0]
        if len(ml_idx) >= 2:
            ml_widths[pd_i] = ml_idx.max() - ml_idx.min()

    # サマリー表示（10刻み）
    for pd_i in range(0, NP, 8):
        pd_norm = pd_i / NP
        ml_proj = bone_mask[pd_i].any(axis=0)
        ml_idx = np.where(ml_proj)[0]
        ml_center = float(ml_idx.mean()) / NM if len(ml_idx) > 0 else 0
        present = "YES" if bone_areas[pd_i] > 50 else "no"
        print(f"  {pd_i:4d}  {pd_norm:.3f}   {bone_areas[pd_i]:8.0f}  {ml_widths[pd_i]:8.0f}  {ml_center:.3f}  {present}")

    # 最大ML幅のPD位置（上顆レベル）
    pd_s = int(NP * 0.30)
    pd_e = int(NP * 0.70)
    condyle_pd = pd_s + int(ml_widths[pd_s:pd_e].argmax())
    print(f"\n  Max ML width at PD={condyle_pd} (norm={condyle_pd/NP:.3f}), width={ml_widths[condyle_pd]:.0f}")
    print(f"  Auto-detected condyle: PD={lm['joint_center'][0]:.3f}")

    # 2. 骨の重心位置 vs DRR上の輝度重心位置
    # 骨がある範囲（PD方向）
    bone_present = bone_areas > 50
    bone_pd_indices = np.where(bone_present)[0]
    if len(bone_pd_indices) > 0:
        bone_pd_start = bone_pd_indices[0] / NP
        bone_pd_end = bone_pd_indices[-1] / NP
        bone_pd_center = bone_pd_indices.mean() / NP
        print(f"\n  Bone PD range: {bone_pd_start:.3f} - {bone_pd_end:.3f} (center: {bone_pd_center:.3f})")

    # 3. DRR上の輝度プロファイル
    drr_ap = generate_drr(vol, axis="AP", sid_mm=1000.0, voxel_mm=voxel_mm)
    print(f"\n=== DRR AP image analysis ===")
    print(f"  Shape: {drr_ap.shape}")

    # 各行の平均輝度
    row_means = drr_ap.mean(axis=1).astype(float)
    # 骨領域（輝度が高い行）
    bright_threshold = row_means.max() * 0.3
    bright_rows = np.where(row_means > bright_threshold)[0]
    if len(bright_rows) > 0:
        drr_bone_start = bright_rows[0] / drr_ap.shape[0]
        drr_bone_end = bright_rows[-1] / drr_ap.shape[0]
        drr_bone_center = bright_rows.mean() / drr_ap.shape[0]
        print(f"  Bright rows range: {drr_bone_start:.3f} - {drr_bone_end:.3f} (center: {drr_bone_center:.3f})")

    # 最大輝度行（肘関節の密な骨部分）
    max_row = np.argmax(row_means)
    print(f"  Max brightness row: {max_row} (norm: {max_row/drr_ap.shape[0]:.3f})")

    # DRR上のML幅プロファイル（骨の見える幅）
    drr_ml_widths = np.zeros(drr_ap.shape[0])
    for r in range(drr_ap.shape[0]):
        row_data = drr_ap[r]
        bright_cols = np.where(row_data > 30)[0]  # 30/255 threshold
        if len(bright_cols) >= 2:
            drr_ml_widths[r] = bright_cols[-1] - bright_cols[0]
    drr_widest_row = np.argmax(drr_ml_widths)
    print(f"  Widest bone row in DRR: {drr_widest_row} (norm: {drr_widest_row/drr_ap.shape[0]:.3f})")

    # 4. 比較: ランドマークのDRR上位置 vs DRRの実際の骨位置
    sid_mm = 1000.0
    SID_vox = sid_mm / voxel_mm
    D_s_ap = max(SID_vox - NA, 1.0)

    print(f"\n=== Landmark vs DRR comparison (AP) ===")
    print(f"  {'Landmark':<22s} {'PD_norm':<10s} {'DRR_y_norm':<12s} {'DRR_y_px':<10s}")
    for name in ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                  "forearm_shaft", "radial_head", "olecranon"]:
        n_PD, n_AP, n_ML = lm[name]
        mag = SID_vox / max(D_s_ap + n_AP * NA, 1e-6)
        py = 0.5 + (n_PD - 0.5) * mag
        py_px = py * drr_ap.shape[0]
        print(f"  {name:<22s} {n_PD:<10.3f} {py:<12.4f} {py_px:<10.1f}")

    print(f"\n  DRR widest row (=condyle in DRR): norm={drr_widest_row/drr_ap.shape[0]:.3f}")
    print(f"  Landmark condyle DRR-y: {0.5 + (lm['joint_center'][0] - 0.5) * SID_vox / max(D_s_ap + lm['joint_center'][1] * NA, 1e-6):.3f}")

    # 5. PD軸の向き検証: volume[0]スライス vs volume[-1]スライスの特徴
    print(f"\n=== PD axis orientation check ===")
    sl_0 = bone_mask[0]
    sl_mid = bone_mask[NP//2]
    sl_end = bone_mask[-1]
    print(f"  PD=0 (first slice): bone area = {sl_0.sum()}")
    print(f"  PD=0.5 (mid slice): bone area = {sl_mid.sum()}")
    print(f"  PD=1.0 (last slice): bone area = {sl_end.sum()}")

    # 骨面積が大きい方が関節に近い
    # 肘関節（上顆）はML幅が最大 → PD=0側とPD=1側どちらがjointに近いか
    first_quarter_area = bone_areas[:NP//4].mean()
    last_quarter_area = bone_areas[3*NP//4:].mean()
    print(f"  First quarter avg bone area: {first_quarter_area:.0f}")
    print(f"  Last quarter avg bone area:  {last_quarter_area:.0f}")

    if first_quarter_area < last_quarter_area:
        print(f"  → PD=0 side has LESS bone → likely PROXIMAL (humerus shaft, narrow)")
        print(f"  → PD=1 side has MORE bone → likely DISTAL (forearm with radius+ulna)")
        print(f"  → PD orientation: Proximal→Distal ✓")
    else:
        print(f"  → PD=0 side has MORE bone → likely DISTAL (more bone cross-section)")
        print(f"  → PD=1 side has LESS bone → likely PROXIMAL")
        print(f"  → PD orientation: Distal→Proximal ✗ (REVERSED!)")

    # 6. DRRの上部 vs 下部の特徴
    print(f"\n=== DRR top vs bottom ===")
    drr_top_quarter = drr_ap[:drr_ap.shape[0]//4]
    drr_bottom_quarter = drr_ap[3*drr_ap.shape[0]//4:]
    print(f"  DRR top quarter mean brightness: {drr_top_quarter.mean():.1f}")
    print(f"  DRR bottom quarter mean brightness: {drr_bottom_quarter.mean():.1f}")

    # 可視化: DRRに水平線でランドマーク位置と骨最大幅位置を描画
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'projection_verify')
    os.makedirs(out_dir, exist_ok=True)

    vis = cv2.cvtColor(drr_ap, cv2.COLOR_GRAY2BGR)
    H, W = drr_ap.shape

    # ランドマーク上顆レベル（赤線）
    jc_y = int((0.5 + (lm['joint_center'][0] - 0.5) * SID_vox / max(D_s_ap + lm['joint_center'][1] * NA, 1e-6)) * H)
    cv2.line(vis, (0, jc_y), (W-1, jc_y), (0, 0, 255), 1)
    cv2.putText(vis, "LM_condyle", (2, jc_y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1)

    # DRR最大幅行（緑線）
    cv2.line(vis, (0, drr_widest_row), (W-1, drr_widest_row), (0, 255, 0), 1)
    cv2.putText(vis, "DRR_widest", (2, drr_widest_row+8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1)

    # 最大輝度行（黄線）
    cv2.line(vis, (0, max_row), (W-1, max_row), (0, 255, 255), 1)

    vis_big = cv2.resize(vis, (W*4, H*4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "diagnose_AP.png"), vis_big)
    print(f"\n  Saved diagnose_AP.png")

if __name__ == "__main__":
    main()
