"""
NCC類似度曲線の形状比較スクリプト

DRR-to-DRR（完全一致・LOO）と DRR-to-実X線の NCC曲線を並べて表示し、
ドメインギャップの「見え方」を可視化する。

論文 Figure 7（補足）: 類似度曲線の形状差異
  - DRR-to-DRR: 鋭いピーク、高い peak_ncc
  - DRR-to-実X線(標準): なだらかなピーク、低い peak_ncc
  - DRR-to-実X線(非標準): 誤った角度に鋭いピーク

使い方:
  python scripts/plot_ncc_curves.py \
    --library data/drr_library/patient008_series4_R_60to180.npz \
    --out_dir results/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))
sys.path.insert(0, str(_PROJECT_ROOT))


def compute_ncc_curve(
    query_img: np.ndarray,
    angle_to_drr: dict[float, np.ndarray],
    angle_min: float,
    angle_max: float,
    is_drr: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """
    全ライブラリ角度に対するNCC・edge_NCCスコアを計算して返す。
    is_drr=True: queryはCLAHE済みuint8（ライブラリから取得）
    is_drr=False: queryは実X線（preprocess_image で前処理）
    """
    from scripts.similarity_matching import extract_edges, ncc, preprocess_image

    if is_drr:
        xray_norm = query_img.astype(np.float32) / 255.0
    else:
        xray_norm = preprocess_image(query_img, apply_rot270=False, auto_crop=True)

    xray_edge = extract_edges(xray_norm)

    angles_sorted = sorted(angle_to_drr.keys())
    ncc_vals  = []
    encc_vals = []

    for a in angles_sorted:
        drr_norm = angle_to_drr[a].astype(np.float32) / 255.0
        drr_edge = extract_edges(drr_norm)
        ncc_vals.append(ncc(drr_norm, xray_norm))
        encc_vals.append(ncc(drr_edge, xray_edge))

    return angles_sorted, ncc_vals, encc_vals


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NCC類似度曲線の形状比較")
    parser.add_argument("--library",  required=True)
    parser.add_argument("--out_dir",  default="results/figures")
    parser.add_argument("--gt_angle", type=float, default=90.0,
                        help="DRRテスト角度（GT角度）")
    args = parser.parse_args()

    from scripts.similarity_matching import load_drr_library

    lib_path = _PROJECT_ROOT / args.library
    angles_arr, drrs, meta = load_drr_library(str(lib_path))
    angle_min = float(meta["angle_min"])
    angle_max = float(meta["angle_max"])
    angle_to_drr = {float(angles_arr[i]): drrs[i] for i in range(len(angles_arr))}
    print(f"ライブラリ: {lib_path.name} ({len(angle_to_drr)}角度)")

    # ── クエリ画像の準備 ────────────────────────────────────────────────────
    gt_angle = args.gt_angle
    # gt_angleがライブラリにない場合は最近傍にスナップ
    gt_angle_snap = min(angle_to_drr.keys(), key=lambda a: abs(a - gt_angle))
    if abs(gt_angle_snap - gt_angle) > 0.01:
        print(f"  [警告] GT角度 {gt_angle}° はライブラリに存在しません → {gt_angle_snap}° にスナップ")
    gt_angle = gt_angle_snap

    # 1) DRR from library (exact match, standard test)
    drr_query_std = angle_to_drr[gt_angle]

    # 2) DRR from library (LOO: find closest without exact match)
    loo_query = angle_to_drr[gt_angle]  # same image, LOO uses different library

    # 3) Real X-ray images
    xray_paths = {
        "Real X-ray (Standard 1)":    "data/real_xray/images/008_LAT.png",
        "Real X-ray (Standard 2)":    "data/real_xray/images/cr_008_3_52kVp.png",
        "Real X-ray (Non-standard)":  "data/real_xray/images/cr_008_2_50kVp.png",
    }

    print(f"\nNCC曲線計算中 (DRR-to-DRR GT={gt_angle}°)...")
    angles_std, ncc_std, encc_std = compute_ncc_curve(
        drr_query_std, angle_to_drr, angle_min, angle_max, is_drr=True
    )

    realxray_curves = {}
    for label, xray_path in xray_paths.items():
        full_path = _PROJECT_ROOT / xray_path
        if not full_path.exists():
            print(f"  SKIP: {full_path}")
            continue
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        print(f"  {label}...")
        a, nc, enc = compute_ncc_curve(img, angle_to_drr, angle_min, angle_max, is_drr=False)
        realxray_curves[label] = (a, nc, enc)

    # ── プロット ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    })

    n_panels = 1 + len(realxray_curves)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    # Colors
    C_NCC  = "#2196F3"
    C_ENCC = "#FF9800"
    C_GT   = "#f44336"

    def _plot_curve(ax, angles, ncc_v, encc_v, title, gt_angle=None, mark_best=True):
        ax.plot(angles, ncc_v,  "-", color=C_NCC,  linewidth=1.5, label="NCC")
        ax.plot(angles, encc_v, "-", color=C_ENCC, linewidth=1.0, alpha=0.7, label="edge-NCC")

        best_ncc_a  = angles[np.argmax(ncc_v)]
        best_encc_a = angles[np.argmax(encc_v)]
        best_comb   = (best_ncc_a + best_encc_a) / 2

        if mark_best:
            ax.axvline(best_ncc_a,  color=C_NCC,  linewidth=1.0, linestyle="--", alpha=0.7,
                       label=f"NCC peak={best_ncc_a:.1f}°")
            ax.axvline(best_encc_a, color=C_ENCC, linewidth=1.0, linestyle="--", alpha=0.7,
                       label=f"edge-NCC peak={best_encc_a:.1f}°")
            ax.axvline(best_comb,   color="black", linewidth=1.5, linestyle="-",  alpha=0.9,
                       label=f"Combined={best_comb:.1f}°")

        if gt_angle is not None:
            ax.axvline(gt_angle, color=C_GT, linewidth=1.2, linestyle=":", alpha=0.8,
                       label=f"GT={gt_angle:.1f}°")

        peak_ncc  = max(ncc_v)
        ncc_arr   = np.array(ncc_v)
        sharpness = (peak_ncc - ncc_arr.mean()) / (ncc_arr.std() + 1e-8)

        ax.set_xlabel("Angle [°]")
        ax.set_ylabel("NCC Score")
        ax.set_title(f"{title}\npeak_ncc={peak_ncc:.3f}  sharpness={sharpness:.2f}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(angle_min, angle_max)

        return best_comb, peak_ncc, sharpness

    # Panel 0: DRR-to-DRR (self-test)
    best, pncc, sharp = _plot_curve(
        axes[0], angles_std, ncc_std, encc_std,
        f"DRR-to-DRR (Standard)\nQuery: {gt_angle:.0f}° DRR",
        gt_angle=gt_angle,
    )
    print(f"\nDRR-to-DRR: combined={best:.1f}° peak_ncc={pncc:.3f} sharpness={sharp:.2f}")

    # Panels 1+: Real X-rays
    for i, (label, (a, nc, enc)) in enumerate(realxray_curves.items()):
        best, pncc, sharp = _plot_curve(
            axes[i + 1], a, nc, enc,
            label,
            gt_angle=90.0,  # All real X-rays are GT=90°
        )
        err = abs(best - 90.0)
        print(f"{label}: combined={best:.1f}° (err={err:.1f}°) peak_ncc={pncc:.3f} sharpness={sharp:.2f}")

    fig.suptitle(
        "NCC Similarity Curve Shapes: DRR-to-DRR vs DRR-to-Real X-ray\n"
        "DRR matching produces sharp peaks; real X-ray matching is flatter (domain gap)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig7_ncc_curves.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
