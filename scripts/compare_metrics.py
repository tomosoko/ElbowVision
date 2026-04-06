"""
類似度メトリクス比較スクリプト

複数の類似度メトリクス（NCC / edge_NCC / combined / NMI / combined_NMI）を
同一の実X線データセットで比較し、精度・バイアスを定量化する。

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
  └── metric_comparison.png   — バーチャート比較
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))
sys.path.insert(0, str(_PROJECT_ROOT))


METRICS = ["ncc", "edge_ncc", "combined", "nmi", "combined_nmi"]


def evaluate_xray(
    xray_path: str,
    library_path: str,
    gt_angle: float,
) -> dict[str, float]:
    """1枚のX線を全メトリクスで評価"""
    from scripts.similarity_matching import (
        DRRLibraryCache, extract_edges, load_drr_library,
        match_angle_from_library, ncc, nmi, preprocess_image,
    )
    import time

    xray_img = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)
    if xray_img is None:
        raise FileNotFoundError(f"X線画像が読み込めません: {xray_path}")

    results = {"filename": Path(xray_path).name, "gt_angle": gt_angle}

    for metric in METRICS:
        t0 = time.time()
        r = match_angle_from_library(library_path, xray_img, metric=metric)
        elapsed = time.time() - t0
        results[f"pred_{metric}"] = round(r.best_angle, 2)
        results[f"err_{metric}"]  = round(abs(r.best_angle - gt_angle), 2)
        results[f"t_{metric}"]    = round(elapsed, 3)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="類似度メトリクス比較")
    parser.add_argument("--library",  required=True)
    parser.add_argument("--xray_dir", required=True, help="X線画像ディレクトリ")
    parser.add_argument("--gt_csv",   required=True,
                        help="GTファイル (filename,gt_angle_deg の2列CSV)")
    parser.add_argument("--out_dir",  default="results/metric_comparison")
    parser.add_argument("--metrics",  default="all",
                        help="カンマ区切りのメトリクス名 or 'all'")
    args = parser.parse_args()

    lib_path  = str(_PROJECT_ROOT / args.library)
    xray_dir  = _PROJECT_ROOT / args.xray_dir
    out_dir   = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT CSV読み込み
    gt_map: dict[str, float] = {}
    with open(_PROJECT_ROOT / args.gt_csv) as f:
        for row in csv.DictReader(f):
            # 列名は flexible に対応
            fname = row.get("filename") or row.get("xray_path") or ""
            angle = float(row.get("gt_angle_deg") or row.get("gt_angle") or 0)
            if fname:
                gt_map[Path(fname).name] = angle

    if not gt_map:
        print("ERROR: GT CSVが空か形式が不正です")
        return

    print(f"対象X線: {len(gt_map)}枚  ライブラリ: {Path(lib_path).name}")
    print(f"メトリクス: {METRICS}\n")

    all_results = []
    for fname, gt_angle in gt_map.items():
        xray_path = xray_dir / fname
        if not xray_path.exists():
            print(f"  SKIP: {fname} が見つかりません")
            continue
        print(f"  {fname} (GT={gt_angle}°) ...")
        try:
            row = evaluate_xray(str(xray_path), lib_path, gt_angle)
            all_results.append(row)
            # サマリー表示
            parts = [f"{m}: {row[f'pred_{m}']:.1f}° (err={row[f'err_{m}']:.1f}°)" for m in METRICS]
            print("    " + " | ".join(parts))
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_results:
        print("評価できた画像がありません")
        return

    # CSV保存
    fieldnames = ["filename", "gt_angle"]
    for m in METRICS:
        fieldnames += [f"pred_{m}", f"err_{m}", f"t_{m}"]
    csv_path = out_dir / "metric_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV保存: {csv_path}")

    # ── 統計サマリー ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Metric':<15} {'MAE (°)':>10} {'Bias (°)':>10} {'Max err (°)':>12}")
    print("-"*65)
    for m in METRICS:
        errs  = [r[f"err_{m}"]               for r in all_results]
        diffs = [r[f"pred_{m}"] - r["gt_angle"] for r in all_results]
        mae  = np.mean(errs)
        bias = np.mean(diffs)
        mx   = np.max(errs)
        print(f"{m:<15} {mae:>10.2f} {bias:>10.2f} {mx:>12.2f}")
    print("="*65)

    # ── プロット ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

        labels = [r["filename"].replace(".png", "") for r in all_results]
        n = len(labels)
        x = np.arange(n)
        width = 0.15
        colors = plt.cm.tab10.colors

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Per-image error by metric
        ax = axes[0]
        for i, m in enumerate(METRICS):
            errs = [r[f"err_{m}"] for r in all_results]
            ax.bar(x + i * width - (len(METRICS)-1)*width/2, errs,
                   width, label=m, color=colors[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Absolute Error [°]")
        ax.set_title("Per-image Error by Metric")
        ax.axhline(3.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="3° threshold")
        ax.legend(fontsize=8, ncol=2)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # MAE summary bar
        ax2 = axes[1]
        maes = []
        for m in METRICS:
            errs = [r[f"err_{m}"] for r in all_results]
            maes.append(np.mean(errs))
        bars = ax2.bar(METRICS, maes, color=colors[:len(METRICS)], alpha=0.85)
        for bar, val in zip(bars, maes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                     f"{val:.2f}°", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("MAE [°]")
        ax2.set_title(f"MAE Comparison (n={n})")
        ax2.axhline(3.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="3° threshold")
        ax2.legend(fontsize=9)
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        ax2.tick_params(axis="x", rotation=20)

        fig.suptitle(
            "Similarity Metric Comparison — Real Phantom X-rays\n"
            f"DRR Library: {Path(lib_path).name}",
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
