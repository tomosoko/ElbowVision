"""
ドメインギャップ頑健性テスト

DRRにリアルな画像劣化（ノイズ・コントラスト・ブラー・輝度シフト）を
段階的に適用し、類似度マッチングのMAEがどう変化するか測定する。

目的:
  - combined metric の頑健性マージンを定量化
  - 実X線が「どの程度の見た目の差異まで許容できるか」を予測
  - Phase 2の期待精度を劣化の程度から推定

使い方:
  python scripts/test_robustness.py \
    --library data/drr_library/patient008_series4_R_60to180.npz \
    --test_angles "90,100,110,120,130,140,150,160,170,180" \
    --out_dir results/robustness/

出力:
  results/robustness/
  ├── robustness_results.csv   — 各劣化レベルごとのMAE
  ├── robustness_plot.png      — 劣化レベル vs MAE 曲線
  └── robustness_summary.txt   — 許容限界まとめ
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))
sys.path.insert(0, str(_PROJECT_ROOT))


# ── 画像劣化関数 ─────────────────────────────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """ガウシアンノイズ (sigma: 輝度値の標準偏差, 0-255スケール)"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """ガウシアンブラー (ksize: カーネルサイズ, 奇数)"""
    ksize = max(1, ksize | 1)  # 奇数に
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def shift_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """輝度シフト (delta: -255〜255)"""
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def change_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """コントラスト変化 (alpha: 倍率, 1.0=変化なし)"""
    mean = img.astype(np.float32).mean()
    return np.clip((img.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """ガンマ補正 (gamma: 1.0=変化なし, <1=明るく, >1=暗く)"""
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def histogram_equalization(img: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE強度変化 (clip_limit: 高いほど強いコントラスト均一化)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)


# ── 劣化シナリオ定義 ─────────────────────────────────────────────────────────

PERTURBATIONS = {
    "gaussian_noise": {
        "func": add_gaussian_noise,
        "levels": [0, 5, 10, 15, 20, 25, 30, 40, 50],
        "param_name": "sigma",
        "unit": "σ (0-255)",
        "clinical_threshold": 20,  # 通常の実X線ノイズレベル
    },
    "blur": {
        "func": apply_blur,
        "levels": [1, 3, 5, 7, 9, 11, 13, 15],
        "param_name": "ksize",
        "unit": "kernel size [px]",
        "clinical_threshold": 7,
    },
    "brightness_shift": {
        "func": shift_brightness,
        "levels": [0, 10, 20, 30, 40, 50, 60, 80, 100],
        "param_name": "delta",
        "unit": "Δ brightness (0-255)",
        "clinical_threshold": 40,
    },
    "contrast_change": {
        "func": change_contrast,
        "levels": [1.0, 0.8, 0.6, 1.2, 1.4, 1.6, 2.0, 0.4],
        "param_name": "alpha",
        "unit": "contrast multiplier",
        "clinical_threshold": 1.4,
    },
    "gamma": {
        "func": apply_gamma,
        "levels": [1.0, 0.7, 0.5, 1.3, 1.6, 2.0, 2.5, 0.3],
        "param_name": "gamma",
        "unit": "gamma value",
        "clinical_threshold": 1.6,
    },
}


def run_matching_with_perturbation(
    perturbed_drr: np.ndarray,
    angle_to_drr: dict[float, np.ndarray],
    gt_angle: float,
    angle_min: float,
    angle_max: float,
    coarse_step: float = 5.0,
    fine_range: float = 10.0,
    metric: str = "combined",
) -> float:
    """
    劣化DRRをクエリとして、ライブラリとの類似度マッチングを実行。
    metric: "ncc" | "edge_ncc" | "combined" | "nmi" | "combined_nmi"
    """
    from scripts.similarity_matching import _parabolic_peak, extract_edges, ncc, nmi

    xray_norm = perturbed_drr.astype(np.float32) / 255.0
    xray_edge = extract_edges(xray_norm)

    all_scores: dict[float, dict[str, float]] = {}

    def _score(angle: float) -> dict[str, float]:
        nearest  = min(angle_to_drr.keys(), key=lambda a: abs(a - angle))
        drr_norm = angle_to_drr[nearest].astype(np.float32) / 255.0
        return {
            "ncc":      ncc(drr_norm, xray_norm),
            "edge_ncc": ncc(extract_edges(drr_norm), xray_edge),
            "nmi":      nmi(drr_norm, xray_norm),
        }

    # 粗探索
    coarse_angles = np.arange(angle_min, angle_max + coarse_step, coarse_step).tolist()
    for a in coarse_angles:
        all_scores[a] = _score(a)

    _primary = "ncc" if metric in ("combined", "combined_nmi") else metric
    coarse_best = max(coarse_angles, key=lambda a: all_scores[a][_primary])

    # 精密探索
    fine_min = max(angle_min, coarse_best - fine_range)
    fine_max = min(angle_max, coarse_best + fine_range)
    fine_angles = [a for a in np.arange(fine_min, fine_max + 1.0, 1.0).tolist()
                   if round(a, 4) not in {round(k, 4) for k in all_scores}]
    for a in fine_angles:
        all_scores[a] = _score(a)

    # ピーク決定
    if metric == "combined":
        best_ncc  = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
        best_encc = float(max(all_scores, key=lambda a: all_scores[a]["edge_ncc"]))
        pred_angle = (best_ncc + best_encc) / 2.0
    elif metric == "combined_nmi":
        best_ncc = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
        best_nmi = float(max(all_scores, key=lambda a: all_scores[a]["nmi"]))
        pred_angle = (best_ncc + best_nmi) / 2.0
    else:
        pred_angle = _parabolic_peak(all_scores, metric)

    return pred_angle


def main() -> None:
    parser = argparse.ArgumentParser(description="類似度マッチング頑健性テスト")
    parser.add_argument("--library", required=True)
    parser.add_argument("--test_angles",
                        default="90,100,110,120,130,140,150,160,170,180")
    parser.add_argument("--n_repeat", type=int, default=5,
                        help="確率的劣化（ノイズ）の繰り返し回数")
    parser.add_argument("--out_dir", default="results/robustness")
    parser.add_argument("--perturbation", default="all",
                        choices=list(PERTURBATIONS.keys()) + ["all"])
    parser.add_argument("--metric", default="combined",
                        choices=["ncc", "edge_ncc", "combined", "nmi", "combined_nmi"],
                        help="類似度メトリクス（デフォルト: combined）")
    args = parser.parse_args()

    from scripts.similarity_matching import load_drr_library

    lib_path = _PROJECT_ROOT / args.library
    out_dir  = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ライブラリロード
    angles_arr, drrs, meta = load_drr_library(str(lib_path))
    lib_step  = float(meta["angle_step"])
    angle_min = float(meta["angle_min"])
    angle_max = float(meta["angle_max"])
    angle_to_drr = {float(angles_arr[i]): drrs[i] for i in range(len(angles_arr))}
    print(f"ライブラリ: {lib_path.name} ({len(angle_to_drr)}角度)")

    # テスト角度
    test_angles_raw = [float(a.strip()) for a in args.test_angles.split(",")]
    test_angles = [min(angle_to_drr.keys(), key=lambda a: abs(a - ta)) for ta in test_angles_raw]
    print(f"テスト角度: {test_angles}")
    print(f"メトリクス: {args.metric}")

    # 対象劣化
    target_perturbations = (
        list(PERTURBATIONS.keys()) if args.perturbation == "all"
        else [args.perturbation]
    )

    all_results = []

    for pert_name in target_perturbations:
        pert = PERTURBATIONS[pert_name]
        func = pert["func"]
        levels = pert["levels"]
        print(f"\n== {pert_name} ({pert['unit']}) ==")

        level_results = []
        for level in levels:
            errors = []
            for gt_angle in test_angles:
                test_drr = angle_to_drr[gt_angle].copy()
                # 確率的劣化は複数回実行して平均
                angle_errors = []
                n_iter = args.n_repeat if pert_name == "gaussian_noise" else 1
                for _ in range(n_iter):
                    perturbed = func(test_drr, level)
                    pred = run_matching_with_perturbation(
                        perturbed, angle_to_drr, gt_angle, angle_min, angle_max,
                        metric=args.metric,
                    )
                    angle_errors.append(abs(pred - gt_angle))
                errors.append(np.mean(angle_errors))

            mae = np.mean(errors)
            max_err = np.max(errors)
            print(f"  {pert['param_name']}={level:<6} → MAE={mae:.2f}°  max={max_err:.1f}°")
            level_results.append({
                "perturbation": pert_name,
                "param_name": pert["param_name"],
                "level": level,
                "mae": round(mae, 4),
                "max_error": round(max_err, 4),
                "n_angles": len(test_angles),
            })
            all_results.append(level_results[-1])

    # CSV保存
    csv_path = out_dir / "robustness_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV保存: {csv_path}")

    # ── プロット ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
        n_plots = len(target_perturbations)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        colors = plt.cm.tab10.colors

        summary_lines = [f"Robustness Analysis — Similarity Matching ({args.metric})\n" + "="*55]

        for i, pert_name in enumerate(target_perturbations):
            pert = PERTURBATIONS[pert_name]
            subset = [r for r in all_results if r["perturbation"] == pert_name]
            levels = [r["level"] for r in subset]
            maes   = [r["mae"] for r in subset]
            maxes  = [r["max_error"] for r in subset]

            ax = axes[i]
            ax.plot(levels, maes,  "o-", color=colors[i], linewidth=2, label="MAE")
            ax.plot(levels, maxes, "s--", color=colors[i], alpha=0.5, linewidth=1, label="Max error")
            ax.axhline(3.0, color="orange", linestyle="--", linewidth=1.0, alpha=0.8,
                       label="3° clinical threshold")
            ax.axhline(8.0, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
                       label="8° LoA threshold")
            if pert["clinical_threshold"] is not None:
                ax.axvline(pert["clinical_threshold"], color="gray", linestyle=":",
                           linewidth=1.0, alpha=0.7, label=f"Clinical level")
            ax.set_xlabel(f"{pert['param_name']} [{pert['unit']}]")
            ax.set_ylabel("Error [°]")
            ax.set_title(pert_name.replace("_", " ").title())
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(10, max(maxes) * 1.2))

            # 臨床閾値超過点
            threshold_idx = next((j for j, m in enumerate(maes) if m > 3.0), None)
            if threshold_idx is not None:
                thr_level = levels[threshold_idx]
                summary_lines.append(
                    f"{pert_name:25s}: MAE>3° at {pert['param_name']}={thr_level} "
                    f"[clinical~{pert['clinical_threshold']}]"
                )
            else:
                summary_lines.append(
                    f"{pert_name:25s}: MAE<3° at ALL levels tested"
                )

        # 余白パネルを非表示
        for j in range(len(target_perturbations), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Robustness to Image Degradation — Similarity Matching ({args.metric})\n"
            f"DRR library self-test, n={len(test_angles)} angles",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / "robustness_plot.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"図保存: {out_dir / 'robustness_plot.png'}")

        # サマリー保存
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        (out_dir / "robustness_summary.txt").write_text(summary_text, encoding="utf-8")

    except Exception as e:
        print(f"図生成失敗: {e}")

    print(f"\n完了。結果: {out_dir}/")


if __name__ == "__main__":
    main()
