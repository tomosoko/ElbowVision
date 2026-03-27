#!/usr/bin/env python3
"""
measure_landmarks.py -- ファントムCTからランドマークを自動検出・精密計測し、
DEFAULT_LANDMARKS_NORMALIZED との比較を行うスクリプト。

出力:
  results/domain_gap_analysis/landmarks_measured.png  -- 可視化画像
  results/domain_gap_analysis/landmarks_measured.txt  -- テキストレポート
"""

import sys
import os
import numpy as np

# matplotlib Agg backend (before any other mpl import)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ─── elbow_synth からインポート ───────────────────────────────────────────────
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    DEFAULT_LANDMARKS_NORMALIZED,
)

# ─── 設定 ────────────────────────────────────────────────────────────────────
CT_DIR = os.path.join(PROJ_ROOT, "data", "raw_dicom", "ct")
OUT_DIR = os.path.join(PROJ_ROOT, "results", "domain_gap_analysis")
SERIES_NUM = 1
LATERALITY = "R"
TARGET_SIZE = 128
HU_MIN = -200
HU_MAX = 1000


def precise_landmark_measurement(volume: np.ndarray) -> dict:
    """
    正準ボリューム (PD, AP, ML) から精密ランドマーク計測を行う。

    auto_detect_landmarks よりきめ細かい手順:
      1. 全PDスライスでML幅を計算し、最大幅のPDレベルを特定（上顆レベル）
      2. 上顆レベルで骨マスクからML/AP方向の精密位置を特定
      3. 橈骨頭・肘頭も同様に精密計測
    """
    pd_size, ap_size, ml_size = volume.shape

    # Otsu法で骨閾値を自動決定
    flat = volume.flatten()
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    w0, mu0 = 0.0, 0.0
    mu_total = float((hist * bin_centers).sum() / total)
    best_var, best_thresh = 0.0, 0.5
    for i in range(len(hist)):
        w0 += hist[i] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (mu0 * (w0 * total - hist[i]) + bin_centers[i] * hist[i]) / (w0 * total) if w0 * total > 0 else 0
        mu1 = (float((hist[i+1:] * bin_centers[i+1:]).sum()) / (w1 * total)) if w1 * total > 0 else 0
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_thresh = var, bin_centers[i]
    bone_threshold = best_thresh
    print(f"  [精密計測] 骨閾値 (Otsu): {bone_threshold:.3f}")

    bone_mask = volume > bone_threshold

    # ── Step 1: 全PDスライスでML幅を計算 → 最大幅のPDレベル ──────────────────
    ml_widths = np.zeros(pd_size)
    bone_area = np.zeros(pd_size)
    for pd_i in range(pd_size):
        sl = bone_mask[pd_i]
        ml_proj = sl.any(axis=0)
        ml_idx = np.where(ml_proj)[0]
        if len(ml_idx) >= 2:
            ml_widths[pd_i] = ml_idx.max() - ml_idx.min()
        bone_area[pd_i] = sl.sum()

    # 中央40%に限定して上顆レベルを検出（より厳しい範囲で精度向上）
    pd_s = int(pd_size * 0.30)
    pd_e = int(pd_size * 0.70)
    epicondyle_pd_idx = pd_s + int(ml_widths[pd_s:pd_e].argmax())
    epicondyle_pd_norm = epicondyle_pd_idx / pd_size
    print(f"  [精密計測] 上顆レベル PD index: {epicondyle_pd_idx} (norm: {epicondyle_pd_norm:.3f})")
    print(f"  [精密計測] 上顆レベル ML幅: {ml_widths[epicondyle_pd_idx]:.1f} voxels")

    # ── Step 2: 上顆レベルでの精密計測 ──────────────────────────────────────────
    joint_slice = bone_mask[epicondyle_pd_idx]  # (AP, ML)
    ml_proj = joint_slice.any(axis=0)
    ap_proj = joint_slice.any(axis=1)
    ml_idx = np.where(ml_proj)[0]
    ap_idx = np.where(ap_proj)[0]

    # 外側・内側上顆: ML方向の最外端
    lat_epicondyle_ml = float(ml_idx.max()) / ml_size
    med_epicondyle_ml = float(ml_idx.min()) / ml_size

    # AP重心（上顆レベル全体）
    epicondyle_ap = float(ap_idx.mean()) / ap_size

    # 関節中心: 上顆のML中点、AP重心
    joint_center_ml = (lat_epicondyle_ml + med_epicondyle_ml) / 2

    print(f"  [精密計測] 外側上顆 ML: {lat_epicondyle_ml:.3f}")
    print(f"  [精密計測] 内側上顆 ML: {med_epicondyle_ml:.3f}")
    print(f"  [精密計測] 上顆レベル AP重心: {epicondyle_ap:.3f}")

    # ── Step 3: 肘頭（olecranon）── 上顆レベルで最後方の骨点 ─────────────────
    # AP方向で後半60%以降の骨ボクセルから重心を計算
    ol_mask = joint_slice.copy()
    ol_mask[:int(ap_size * 0.55), :] = False  # 前方を除外
    ol_yx = np.where(ol_mask)
    if len(ol_yx[0]) > 0:
        olecranon_ap = float(ol_yx[0].mean()) / ap_size
        olecranon_ml = float(ol_yx[1].mean()) / ml_size
        # 最後方点も記録
        most_posterior_ap = float(ol_yx[0].max()) / ap_size
    else:
        olecranon_ap = 0.80
        olecranon_ml = joint_center_ml
        most_posterior_ap = 0.80
    print(f"  [精密計測] 肘頭 AP重心: {olecranon_ap:.3f}, ML: {olecranon_ml:.3f}")
    print(f"  [精密計測] 最後方骨点 AP: {most_posterior_ap:.3f}")

    # ── Step 4: 橈骨頭（radial head）── 上顆レベルより少し遠位、外側前方 ────
    rh_pd_norm = min(0.95, epicondyle_pd_norm + 0.08)
    rh_pd_idx = int(pd_size * rh_pd_norm)
    rh_sl = bone_mask[rh_pd_idx]

    # 外側半かつ前半に限定
    rh_mask = rh_sl.copy()
    rh_mask[:, :int(ml_size * lat_epicondyle_ml * 0.8)] = False  # 内側を除外
    rh_mask[int(ap_size * 0.55):, :] = False  # 後方を除外
    rh_yx = np.where(rh_mask)
    if len(rh_yx[0]) > 0:
        radial_head_ap = float(rh_yx[0].mean()) / ap_size
        radial_head_ml = float(rh_yx[1].mean()) / ml_size
    else:
        radial_head_ap = 0.40
        radial_head_ml = lat_epicondyle_ml
    print(f"  [精密計測] 橈骨頭 PD: {rh_pd_norm:.3f}, AP: {radial_head_ap:.3f}, ML: {radial_head_ml:.3f}")

    # ── Step 5: 上腕骨幹部・前腕骨幹部 ──────────────────────────────────────────
    def shaft_centroid(pd_norm):
        pd_idx = int(pd_size * pd_norm)
        pd_idx = max(0, min(pd_size - 1, pd_idx))
        sl = bone_mask[pd_idx]
        yx = np.where(sl)
        if len(yx[0]) > 0:
            return float(yx[0].mean()) / ap_size, float(yx[1].mean()) / ml_size
        return 0.50, 0.50

    hum_pd_norm = max(0.05, epicondyle_pd_norm - 0.25)
    fore_pd_norm = min(0.95, epicondyle_pd_norm + 0.25)
    hum_ap, hum_ml = shaft_centroid(hum_pd_norm)
    fore_ap, fore_ml = shaft_centroid(fore_pd_norm)
    print(f"  [精密計測] 上腕骨幹部 PD: {hum_pd_norm:.3f}, AP: {hum_ap:.3f}, ML: {hum_ml:.3f}")
    print(f"  [精密計測] 前腕骨幹部 PD: {fore_pd_norm:.3f}, AP: {fore_ap:.3f}, ML: {fore_ml:.3f}")

    # ── 結果まとめ ────────────────────────────────────────────────────────────
    landmarks = {
        "humerus_shaft":      (hum_pd_norm,       hum_ap,        hum_ml),
        "lateral_epicondyle": (epicondyle_pd_norm, epicondyle_ap, lat_epicondyle_ml),
        "medial_epicondyle":  (epicondyle_pd_norm, epicondyle_ap, med_epicondyle_ml),
        "forearm_shaft":      (fore_pd_norm,       fore_ap,       fore_ml),
        "radial_head":        (rh_pd_norm,         radial_head_ap, radial_head_ml),
        "olecranon":          (epicondyle_pd_norm, olecranon_ap,  olecranon_ml),
        "joint_center":       (epicondyle_pd_norm, epicondyle_ap, joint_center_ml),
    }

    extra_info = {
        "epicondyle_pd_idx": epicondyle_pd_idx,
        "ml_widths": ml_widths,
        "bone_area": bone_area,
        "bone_threshold": bone_threshold,
        "most_posterior_ap": most_posterior_ap,
    }

    return landmarks, extra_info


def create_visualization(volume, auto_lm, precise_lm, default_lm, extra_info, out_path):
    """
    3直交断面にランドマークをオーバーレイした可視化画像を生成。
    """
    pd_size, ap_size, ml_size = volume.shape
    epic_pd = extra_info["epicondyle_pd_idx"]

    # 各ランドマークセットの色
    colors_auto = "cyan"
    colors_precise = "lime"
    colors_default = "red"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ── Row 0: 3 orthogonal slices ──────────────────────────────────────────
    # Axial slice at epicondyle level: axes[PD] -> shows (AP, ML)
    ax = axes[0, 0]
    ax.imshow(volume[epic_pd], cmap="gray", aspect="auto", origin="upper")
    ax.set_title(f"Axial (PD={epic_pd}, norm={epic_pd/pd_size:.2f})")
    ax.set_xlabel("ML (medial -> lateral)")
    ax.set_ylabel("AP (anterior -> posterior)")

    # Plot landmarks on axial slice (AP vs ML)
    for name, (pd, ap, ml) in precise_lm.items():
        pd_idx = int(pd * pd_size)
        if abs(pd_idx - epic_pd) < 5:  # only show landmarks near this PD level
            ax.plot(ml * ml_size, ap * ap_size, "o", color=colors_precise,
                    markersize=8, markeredgecolor="black", markeredgewidth=1)
            ax.annotate(name.replace("_", "\n"), (ml * ml_size, ap * ap_size),
                        fontsize=6, color=colors_precise, ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")
    for name, (pd, ap, ml) in default_lm.items():
        pd_idx = int(pd * pd_size)
        if abs(pd_idx - epic_pd) < 5:
            ax.plot(ml * ml_size, ap * ap_size, "x", color=colors_default,
                    markersize=8, markeredgewidth=2)

    # Coronal slice at AP centroid: axes[AP] -> shows (PD, ML)
    ap_mid = int(ap_size * 0.5)
    ax = axes[0, 1]
    ax.imshow(volume[:, ap_mid, :], cmap="gray", aspect="auto", origin="upper")
    ax.set_title(f"Coronal (AP={ap_mid}, norm=0.50)")
    ax.set_xlabel("ML (medial -> lateral)")
    ax.set_ylabel("PD (proximal -> distal)")

    for name, (pd, ap, ml) in precise_lm.items():
        ax.plot(ml * ml_size, pd * pd_size, "o", color=colors_precise,
                markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax.annotate(name.replace("_", "\n"), (ml * ml_size, pd * pd_size),
                    fontsize=6, color=colors_precise, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")
    for name, (pd, ap, ml) in default_lm.items():
        ax.plot(ml * ml_size, pd * pd_size, "x", color=colors_default,
                markersize=8, markeredgewidth=2)

    # Sagittal slice at ML midpoint: axes[ML] -> shows (PD, AP)
    ml_mid = int(ml_size * 0.5)
    ax = axes[0, 2]
    ax.imshow(volume[:, :, ml_mid], cmap="gray", aspect="auto", origin="upper")
    ax.set_title(f"Sagittal (ML={ml_mid}, norm=0.50)")
    ax.set_xlabel("AP (anterior -> posterior)")
    ax.set_ylabel("PD (proximal -> distal)")

    for name, (pd, ap, ml) in precise_lm.items():
        ax.plot(ap * ap_size, pd * pd_size, "o", color=colors_precise,
                markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax.annotate(name.replace("_", "\n"), (ap * ap_size, pd * pd_size),
                    fontsize=6, color=colors_precise, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")
    for name, (pd, ap, ml) in default_lm.items():
        ax.plot(ap * ap_size, pd * pd_size, "x", color=colors_default,
                markersize=8, markeredgewidth=2)

    # ── Row 1: Comparison charts ────────────────────────────────────────────
    # ML width profile along PD
    ax = axes[1, 0]
    pd_range = np.arange(pd_size) / pd_size
    ax.plot(pd_range, extra_info["ml_widths"], "b-", linewidth=1.5, label="ML width")
    ax.axvline(epic_pd / pd_size, color="red", linestyle="--", label=f"Epicondyle PD={epic_pd/pd_size:.2f}")
    ax.set_xlabel("PD (normalized)")
    ax.set_ylabel("ML width (voxels)")
    ax.set_title("ML Width Profile Along PD Axis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Comparison table: Default vs Auto vs Precise
    ax = axes[1, 1]
    ax.axis("off")
    all_names = list(precise_lm.keys())
    header = ["Landmark", "Default", "Auto", "Precise", "Diff (D-P)"]
    table_data = []
    for name in all_names:
        d = default_lm.get(name, (None, None, None))
        a = auto_lm.get(name, (None, None, None))
        p = precise_lm[name]
        d_str = f"({d[0]:.2f},{d[1]:.2f},{d[2]:.2f})" if d[0] is not None else "N/A"
        a_str = f"({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})" if a[0] is not None else "N/A"
        p_str = f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
        if d[0] is not None:
            diff = np.sqrt(sum((di - pi) ** 2 for di, pi in zip(d, p)))
            diff_str = f"{diff:.3f}"
        else:
            diff_str = "N/A"
        table_data.append([name, d_str, a_str, p_str, diff_str])

    table = ax.table(cellText=table_data, colLabels=header, loc="center",
                     cellLoc="center", colColours=["#e0e0e0"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    ax.set_title("Landmark Comparison (PD, AP, ML)", fontsize=10, pad=20)

    # Bone area profile
    ax = axes[1, 2]
    ax.plot(pd_range, extra_info["bone_area"], "g-", linewidth=1.5, label="Bone area")
    ax.axvline(epic_pd / pd_size, color="red", linestyle="--", label="Epicondyle level")
    ax.set_xlabel("PD (normalized)")
    ax.set_ylabel("Bone voxels per slice")
    ax.set_title("Bone Area Profile Along PD Axis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_precise,
               markersize=10, label="Precise measurement"),
        Line2D([0], [0], marker="x", color=colors_default,
               markersize=10, markeredgewidth=2, label="DEFAULT_LANDMARKS_NORMALIZED"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Phantom CT Landmark Measurement", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved: {out_path}")


def generate_report(auto_lm, precise_lm, default_lm, extra_info, volume_shape,
                    voxel_mm_eff, out_path):
    """テキストレポートを生成。"""
    pd_size, ap_size, ml_size = volume_shape
    lines = []
    lines.append("=" * 70)
    lines.append("Phantom CT Landmark Measurement Report")
    lines.append("=" * 70)
    lines.append(f"Volume shape (PD, AP, ML): {volume_shape}")
    lines.append(f"Voxel size (effective): {voxel_mm_eff:.3f} mm/voxel")
    lines.append(f"Bone threshold (Otsu): {extra_info['bone_threshold']:.3f}")
    lines.append(f"Epicondyle PD index: {extra_info['epicondyle_pd_idx']}")
    lines.append(f"Epicondyle PD norm: {extra_info['epicondyle_pd_idx'] / pd_size:.3f}")
    lines.append(f"ML width at epicondyle: {extra_info['ml_widths'][extra_info['epicondyle_pd_idx']]:.1f} voxels")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Auto-detected landmarks (auto_detect_landmarks):")
    lines.append("-" * 70)
    for name, (pd, ap, ml) in auto_lm.items():
        lines.append(f"  {name:25s}: PD={pd:.4f}  AP={ap:.4f}  ML={ml:.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Precise landmarks (measure_landmarks.py):")
    lines.append("-" * 70)
    for name, (pd, ap, ml) in precise_lm.items():
        lines.append(f"  {name:25s}: PD={pd:.4f}  AP={ap:.4f}  ML={ml:.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Default landmarks (DEFAULT_LANDMARKS_NORMALIZED):")
    lines.append("-" * 70)
    for name, (pd, ap, ml) in default_lm.items():
        lines.append(f"  {name:25s}: PD={pd:.4f}  AP={ap:.4f}  ML={ml:.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Difference: Default vs Precise (Euclidean distance in norm coords):")
    lines.append("-" * 70)
    for name in precise_lm:
        if name in default_lm:
            d = default_lm[name]
            p = precise_lm[name]
            dist = np.sqrt(sum((di - pi) ** 2 for di, pi in zip(d, p)))
            dist_mm = dist * max(pd_size, ap_size, ml_size) * voxel_mm_eff
            lines.append(f"  {name:25s}: {dist:.4f} (norm)  ~{dist_mm:.1f} mm")
        else:
            lines.append(f"  {name:25s}: (not in DEFAULT)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Ready to paste into DEFAULT_LANDMARKS_NORMALIZED:")
    lines.append("=" * 70)
    lines.append("DEFAULT_LANDMARKS_NORMALIZED = {")
    for name, (pd, ap, ml) in precise_lm.items():
        # Keep 2-decimal precision matching original format
        lines.append(f'    "{name}": {" " * (21 - len(name))}({pd:.2f}, {ap:.2f}, {ml:.2f}),')
    lines.append("}")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {out_path}")
    return report


def main():
    print("=" * 60)
    print("Phantom CT Landmark Measurement")
    print("=" * 60)

    # ── 1. Load CT volume ─────────────────────────────────────────────────────
    print(f"\n[1] Loading CT from: {CT_DIR}")
    print(f"    Series: {SERIES_NUM}, Laterality: {LATERALITY}")
    print(f"    HU window: {HU_MIN} ~ {HU_MAX}, Target size: {TARGET_SIZE}")

    volume, voxel_spacing, lat, voxel_mm_eff = load_ct_volume(
        CT_DIR,
        target_size=TARGET_SIZE,
        laterality=LATERALITY,
        series_num=SERIES_NUM,
        hu_min=HU_MIN,
        hu_max=HU_MAX,
    )
    print(f"    Canonical volume shape: {volume.shape}")
    print(f"    Voxel spacing (z,y,x mm): {voxel_spacing}")
    print(f"    Effective voxel size: {voxel_mm_eff:.3f} mm/voxel")

    # ── 2. Auto-detect landmarks (elbow_synth) ───────────────────────────────
    print(f"\n[2] Running auto_detect_landmarks...")
    auto_lm = auto_detect_landmarks(volume, laterality=LATERALITY)
    print(f"    Auto landmarks: {len(auto_lm)} points")

    # ── 3. Precise measurement ────────────────────────────────────────────────
    print(f"\n[3] Running precise landmark measurement...")
    precise_lm, extra_info = precise_landmark_measurement(volume)
    print(f"    Precise landmarks: {len(precise_lm)} points")

    # ── 4. Default landmarks ──────────────────────────────────────────────────
    default_lm = DEFAULT_LANDMARKS_NORMALIZED.copy()

    # ── 5. Generate outputs ───────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n[4] Generating visualization...")
    fig_path = os.path.join(OUT_DIR, "landmarks_measured.png")
    create_visualization(volume, auto_lm, precise_lm, default_lm, extra_info, fig_path)

    print(f"\n[5] Generating report...")
    txt_path = os.path.join(OUT_DIR, "landmarks_measured.txt")
    report = generate_report(auto_lm, precise_lm, default_lm, extra_info,
                             volume.shape, voxel_mm_eff, txt_path)

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print("\n" + report)

    print("\nDone.")


if __name__ == "__main__":
    main()
