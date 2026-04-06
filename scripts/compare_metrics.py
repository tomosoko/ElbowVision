"""
類似度メトリクス比較スクリプト

複数の類似度メトリクス（NCC / edge_NCC / combined / NMI / combined_NMI）を
同一の実X線データセットで比較し、精度・バイアスを定量化する。

改善点: ライブラリを1回ロードし、全角度に対して全メトリクスを1パスで計算（5×高速化）

目的:
  - 論文 Supplementary: 各メトリクスのDRRバイアスを表で示す
  - combined vs combined_nmi の等価性を実証
  - 将来の多角度・多症例評価での参考

使い方:
  python scripts/compare_metrics.py \
    --library data/drr_library/patient008_series4_R_60to180.npz \
    --xray_dir data/real_xray/images \
    --gt_csv   data/real_xray/ground_truth.csv \
    --out_dir  results/metric_comparison/

出力:
  results/metric_comparison/
  ├── metric_comparison.csv   — 全メトリクス × 全X線の結果
  ├── metric_comparison.png   — バーチャート比較
  └── metric_comparison_bias.png — バイアス図（予測角 vs GT角）
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


ALL_METRICS = ["ncc", "edge_ncc", "combined", "nmi", "combined_nmi"]


def evaluate_xray_all_metrics(
    xray_img: np.ndarray,
    angle_to_drr: dict[float, np.ndarray],
    angle_min: float,
    angle_max: float,
    gt_angle: float,
    coarse_step: float = 5.0,
    fine_range: float = 10.0,
    metrics: list[str] | None = None,
) -> dict[str, object]:
    """
    1パスで全メトリクスを計算する高速評価関数。
    ライブラリを1回読んで全角度のNCC/edge_NCC/NMIスコアを計算し、
    各メトリクスのベスト角度を後処理で決定する。
    """
    from scripts.similarity_matching import (
        _parabolic_peak, extract_edges, ncc, nmi, preprocess_image,
    )
    if metrics is None:
        metrics = ALL_METRICS

    t0 = time.time()

    # X線前処理
    xray_norm = preprocess_image(xray_img, apply_rot270=False, auto_crop=True)
    xray_edge = extract_edges(xray_norm)

    # 全スコア格納: {angle: {"ncc": .., "edge_ncc": .., "nmi": ..}}
    all_scores: dict[float, dict[str, float]] = {}

    def _score_angle(a: float) -> dict[str, float]:
        nearest  = min(angle_to_drr.keys(), key=lambda x: abs(x - a))
        drr_norm = angle_to_drr[nearest].astype(np.float32) / 255.0
        return {
            "ncc":      ncc(drr_norm, xray_norm),
            "edge_ncc": ncc(extract_edges(drr_norm), xray_edge),
            "nmi":      nmi(drr_norm, xray_norm),
        }

    # 粗探索
    coarse_angles = np.arange(angle_min, angle_max + coarse_step, coarse_step).tolist()
    for a in coarse_angles:
        all_scores[a] = _score_angle(a)

    # NCC粗ベスト周辺を精密探索（全メトリクス共通）
    coarse_best_ncc = max(coarse_angles, key=lambda a: all_scores[a]["ncc"])
    fine_min = max(angle_min, coarse_best_ncc - fine_range)
    fine_max = min(angle_max, coarse_best_ncc + fine_range)
    fine_angles = [a for a in np.arange(fine_min, fine_max + 1.0, 1.0).tolist()
                   if round(a, 4) not in {round(k, 4) for k in all_scores}]
    for a in fine_angles:
        all_scores[a] = _score_angle(a)

    # edge_NCC最良が別の場所にある場合の追加精密探索
    coarse_best_encc = max(coarse_angles, key=lambda a: all_scores[a]["edge_ncc"])
    if abs(coarse_best_encc - coarse_best_ncc) > coarse_step:
        e_min = max(angle_min, coarse_best_encc - fine_range)
        e_max = min(angle_max, coarse_best_encc + fine_range)
        for a in np.arange(e_min, e_max + 1.0, 1.0).tolist():
            if round(a, 4) not in {round(k, 4) for k in all_scores}:
                all_scores[a] = _score_angle(a)

    # NMI最良が別の場所にある場合の追加精密探索
    coarse_best_nmi = max(coarse_angles, key=lambda a: all_scores[a]["nmi"])
    if abs(coarse_best_nmi - coarse_best_ncc) > coarse_step:
        n_min = max(angle_min, coarse_best_nmi - fine_range)
        n_max = min(angle_max, coarse_best_nmi + fine_range)
        for a in np.arange(n_min, n_max + 1.0, 1.0).tolist():
            if round(a, 4) not in {round(k, 4) for k in all_scores}:
                all_scores[a] = _score_angle(a)

    elapsed = time.time() - t0

    # ── 各メトリクスのベスト角度を決定 ─────────────────────────────────────
    result = {"elapsed_s": round(elapsed, 3)}

    for m in metrics:
        if m == "combined":
            best_ncc  = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
            best_encc = float(max(all_scores, key=lambda a: all_scores[a]["edge_ncc"]))
            pred = (best_ncc + best_encc) / 2.0
        elif m == "combined_nmi":
            best_ncc = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
            best_nmi = float(max(all_scores, key=lambda a: all_scores[a]["nmi"]))
            pred = (best_ncc + best_nmi) / 2.0
        else:
            pred = _parabolic_peak(all_scores, m)

        result[f"pred_{m}"] = round(pred, 2)
        result[f"err_{m}"]  = round(abs(pred - gt_angle), 2)
        result[f"bias_{m}"] = round(pred - gt_angle, 2)

    # NCC peak値（信頼度）
    nearest = min(all_scores.keys(), key=lambda a: abs(a - result.get("pred_ncc", list(all_scores.keys())[0])))
    ncc_vals = np.array([all_scores[a]["ncc"] for a in all_scores])
    result["peak_ncc"]  = round(float(all_scores[nearest]["ncc"]), 4)
    result["sharpness"] = round(float((result["peak_ncc"] - ncc_vals.mean()) / (ncc_vals.std() + 1e-8)), 3)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="類似度メトリクス比較")
    parser.add_argument("--library",     required=True)
    parser.add_argument("--xray_dir",    required=True, help="X線画像ディレクトリ")
    parser.add_argument("--gt_csv",      required=True,
                        help="GTファイル (filename,gt_angle_deg の2列CSV)")
    parser.add_argument("--out_dir",     default="results/metric_comparison")
    parser.add_argument("--coarse_step", type=float, default=5.0)
    parser.add_argument("--fine_range",  type=float, default=10.0)
    parser.add_argument("--metrics",     default="all",
                        help="カンマ区切りのメトリクス名 or 'all'")
    args = parser.parse_args()

    from scripts.similarity_matching import load_drr_library

    lib_path  = _PROJECT_ROOT / args.library
    xray_dir  = _PROJECT_ROOT / args.xray_dir
    out_dir   = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # メトリクス選択
    if args.metrics.strip().lower() == "all":
        metrics = ALL_METRICS
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    # ライブラリを1回ロード
    angles_arr, drrs, meta = load_drr_library(str(lib_path))
    angle_min = float(meta["angle_min"])
    angle_max = float(meta["angle_max"])
    angle_to_drr = {float(angles_arr[i]): drrs[i] for i in range(len(angles_arr))}
    print(f"ライブラリ: {lib_path.name} ({len(angle_to_drr)}角度)")

    # GT CSV読み込み
    gt_map: dict[str, float] = {}
    with open(_PROJECT_ROOT / args.gt_csv) as f:
        for row in csv.DictReader(f):
            fname = row.get("filename") or row.get("xray_path") or ""
            angle = float(row.get("gt_angle_deg") or row.get("gt_angle") or 0)
            if fname:
                gt_map[Path(fname).name] = angle

    if not gt_map:
        print("ERROR: GT CSVが空か形式が不正です")
        return

    print(f"対象X線: {len(gt_map)}枚  メトリクス: {metrics}\n")

    all_results = []
    for fname, gt_angle in gt_map.items():
        xray_path = xray_dir / fname
        if not xray_path.exists():
            print(f"  SKIP: {fname} が見つかりません")
            continue
        xray_img = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
        if xray_img is None:
            print(f"  SKIP: {fname} 読み込み失敗")
            continue

        print(f"  {fname} (GT={gt_angle}°) ...", end=" ", flush=True)
        try:
            row = evaluate_xray_all_metrics(
                xray_img, angle_to_drr, angle_min, angle_max, gt_angle,
                coarse_step=args.coarse_step, fine_range=args.fine_range,
                metrics=metrics,
            )
            row = {"filename": fname, "gt_angle": gt_angle, **row}
            all_results.append(row)
            parts = [f"{m}={row[f'pred_{m}']:.1f}°(err={row[f'err_{m}']:.1f}°)" for m in metrics]
            print(f"[{row['elapsed_s']}s] " + " | ".join(parts))
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_results:
        print("評価できた画像がありません")
        return

    # CSV保存
    fieldnames = ["filename", "gt_angle"]
    for m in metrics:
        fieldnames += [f"pred_{m}", f"err_{m}", f"bias_{m}"]
    fieldnames += ["peak_ncc", "sharpness", "elapsed_s"]
    csv_path = out_dir / "metric_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV保存: {csv_path}")

    # ── 統計サマリー ──────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Metric':<15} {'MAE (°)':>9} {'Bias (°)':>10} {'Max err (°)':>13} {'SD (°)':>9}")
    print("-"*72)
    for m in metrics:
        errs  = np.array([r[f"err_{m}"]  for r in all_results])
        biases = np.array([r[f"bias_{m}"] for r in all_results])
        mae  = errs.mean()
        bias = biases.mean()
        mx   = errs.max()
        sd   = biases.std(ddof=1) if len(biases) > 1 else 0.0
        print(f"{m:<15} {mae:>9.2f} {bias:>10.2f} {mx:>13.2f} {sd:>9.2f}")
    print("="*72)

    # 標準ポジショニングのみ (non_standard を除く)
    std_results = [r for r in all_results if "non_standard" not in r.get("filename", "").lower()
                   and "cr_008_2" not in r.get("filename", "")]
    if len(std_results) < len(all_results):
        print(f"\n標準ポジショニングのみ (n={len(std_results)}):")
        print(f"{'Metric':<15} {'MAE (°)':>9} {'Bias (°)':>10}")
        print("-"*40)
        for m in metrics:
            errs  = np.array([r[f"err_{m}"]  for r in std_results])
            biases = np.array([r[f"bias_{m}"] for r in std_results])
            print(f"{m:<15} {errs.mean():>9.2f} {biases.mean():>10.2f}")

    # ── プロット ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
        colors = plt.cm.tab10.colors
        labels = [r["filename"].replace(".png", "") for r in all_results]
        n_xrays = len(labels)
        x = np.arange(n_xrays)
        width = 0.14

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ── 左: per-image error grouped by metric ──────────────────────────
        ax = axes[0]
        for i, m in enumerate(metrics):
            errs = [r[f"err_{m}"] for r in all_results]
            ax.bar(x + i * width - (len(metrics)-1)*width/2, errs,
                   width, label=m, color=colors[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Absolute Error [°]")
        ax.set_title("Per-image Error by Metric")
        ax.axhline(3.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="3° threshold")
        ax.legend(fontsize=7, ncol=2)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # ── 中: MAE summary ────────────────────────────────────────────────
        ax2 = axes[1]
        maes = [np.mean([r[f"err_{m}"] for r in all_results]) for m in metrics]
        bars = ax2.bar(metrics, maes, color=colors[:len(metrics)], alpha=0.85)
        for bar, val in zip(bars, maes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f"{val:.2f}°", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("MAE [°]")
        ax2.set_title(f"MAE Comparison (n={n_xrays})")
        ax2.axhline(3.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="3° threshold")
        ax2.legend(fontsize=9)
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        ax2.tick_params(axis="x", rotation=20)

        # ── 右: Bias chart (予測角 - GT角) ────────────────────────────────
        ax3 = axes[2]
        x_m = np.arange(len(metrics))
        biases = [np.mean([r[f"bias_{m}"] for r in all_results]) for m in metrics]
        clrs   = ["#f44336" if b > 0 else "#2196F3" for b in biases]
        bars3  = ax3.bar(metrics, biases, color=clrs, alpha=0.85)
        for bar, val in zip(bars3, biases):
            ypos = val + 0.3 if val >= 0 else val - 1.5
            ax3.text(bar.get_x() + bar.get_width()/2., ypos,
                     f"{val:+.2f}°", ha="center", va="bottom", fontsize=9)
        ax3.axhline(0, color="black", linewidth=1.0)
        ax3.axhline(3.0,  color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax3.axhline(-3.0, color="orange", linestyle="--", linewidth=1, alpha=0.7,
                    label="±3° threshold")
        ax3.set_ylabel("Mean Bias (Pred - GT) [°]")
        ax3.set_title("Systematic Bias by Metric")
        ax3.legend(fontsize=9)
        ax3.yaxis.grid(True, alpha=0.3)
        ax3.set_axisbelow(True)
        ax3.tick_params(axis="x", rotation=20)

        fig.suptitle(
            "Similarity Metric Comparison — Real Phantom X-rays\n"
            f"DRR Library: {lib_path.name}  (n={n_xrays} images)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / "metric_comparison.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"図保存: {out_dir / 'metric_comparison.png'}")

    except Exception as e:
        print(f"図生成失敗: {e}")

    print(f"\n完了。結果: {out_dir}/")


if __name__ == "__main__":
    main()
