"""
メトリクスバイアス図生成スクリプト

論文 Figure 8（技術的貢献）:
NCC(+5°)/edge-NCC(-5°) の相補的バイアスキャンセルを可視化する。

出力: results/figures/fig8_metric_bias.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="メトリクスバイアス図生成")
    parser.add_argument("--csv",
                        default="results/metric_comparison/metric_comparison.csv")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    import csv as _csv
    csv_path = _PROJECT_ROOT / args.csv
    if not csv_path.exists():
        print(f"ERROR: {csv_path} が存在しません。先に compare_metrics.py を実行してください。")
        return

    rows = []
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    # 標準ポジショニングのみ（非標準を除外）
    standard_rows = [r for r in rows if "cr_008_2" not in r["filename"]]
    all_rows      = rows

    metrics = ["ncc", "edge_ncc", "combined", "nmi", "combined_nmi"]
    metric_labels = {
        "ncc":          "NCC",
        "edge_ncc":     "Edge-NCC",
        "combined":     "Combined\n(NCC+Edge-NCC)",
        "nmi":          "NMI",
        "combined_nmi": "Combined-NMI\n(NCC+NMI)",
    }
    colors = {
        "ncc":          "#2196F3",  # blue
        "edge_ncc":     "#FF9800",  # orange
        "combined":     "#4CAF50",  # green
        "nmi":          "#9C27B0",  # purple
        "combined_nmi": "#00BCD4",  # cyan
    }

    gt = 90.0  # all standard images are GT=90°

    # Per-image bias (standard only)
    image_labels = {
        "008_LAT.png":      "LAT-1",
        "cr_008_3_52kVp.png": "LAT-2",
        "new_LAT.png":      "LAT-3",
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel A: Per-metric predicted angle (標準のみ) ──────────────────────────
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.2

    for i, row in enumerate(standard_rows):
        lbl = image_labels.get(row["filename"], row["filename"])
        preds = [float(row[f"pred_{m}"]) for m in metrics]
        ax.bar(x + (i - 1) * width, preds, width, label=lbl, alpha=0.85)

    ax.axhline(gt, color="red", linewidth=1.5, linestyle="--",
               label=f"GT = {gt:.0f}°", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m].replace("\n", " ") for m in metrics],
                       fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Predicted Angle [°]")
    ax.set_title("A  Predicted Angle per Metric\n(Standard LAT, GT=90°, n=3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(75, 105)

    # ── Panel B: Mean bias per metric ─────────────────────────────────────────
    ax = axes[1]
    mean_biases = []
    for m in metrics:
        biases = [float(r[f"bias_{m}"]) for r in standard_rows]
        mean_biases.append(np.mean(biases))

    bar_colors = [colors[m] for m in metrics]
    bars = ax.bar(range(len(metrics)), mean_biases, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(3, color="orange", linewidth=1.0, linestyle="--", alpha=0.7, label="+3° threshold")
    ax.axhline(-3, color="orange", linewidth=1.0, linestyle="--", alpha=0.7, label="-3° threshold")

    for i, (bar, bias) in enumerate(zip(bars, mean_biases)):
        ax.text(bar.get_x() + bar.get_width() / 2., bias + (0.3 if bias >= 0 else -0.5),
                f"{bias:+.1f}°", ha="center", va="bottom" if bias >= 0 else "top",
                fontsize=8.5, fontweight="bold")

    # Annotate bias cancellation
    ncc_bias  = mean_biases[metrics.index("ncc")]
    encc_bias = mean_biases[metrics.index("edge_ncc")]
    ax.annotate("",
                xy=(metrics.index("combined"), (ncc_bias + encc_bias) / 2),
                xytext=(metrics.index("ncc"), ncc_bias),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.annotate("",
                xy=(metrics.index("combined"), (ncc_bias + encc_bias) / 2),
                xytext=(metrics.index("edge_ncc"), encc_bias),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text(metrics.index("combined"), 0.5, "cancel", ha="center", fontsize=7.5,
            color="green", style="italic")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric_labels[m].replace("\n", " ") for m in metrics],
                       fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Mean Bias [°] (Pred − GT)")
    ax.set_title("B  Mean Systematic Bias per Metric\n(Standard LAT, n=3)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-8, 8)

    # ── Panel C: MAE comparison (all 4 images) ─────────────────────────────────
    ax = axes[2]
    mae_std = []
    mae_all = []
    for m in metrics:
        std_errs = [float(r[f"err_{m}"]) for r in standard_rows]
        all_errs = [float(r[f"err_{m}"]) for r in all_rows]
        mae_std.append(np.mean(std_errs))
        mae_all.append(np.mean(all_errs))

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, mae_std, w, label="Standard (n=3)", color=[colors[m] for m in metrics],
                alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, mae_all, w, label="All incl. non-std (n=4)",
                color=[colors[m] for m in metrics], alpha=0.4, edgecolor="gray", hatch="///")

    ax.axhline(3, color="orange", linewidth=1.2, linestyle="--",
               label="3° clinical threshold")

    for bar, val in zip(b1, mae_std):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.3,
                f"{val:.1f}°", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m].replace("\n", " ") for m in metrics],
                       fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("MAE [°]")
    ax.set_title("C  MAE per Metric\n(standard vs. all X-rays)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 30)

    fig.suptitle(
        "Similarity Metric Analysis: NCC–EdgeNCC Bias Cancellation\n"
        "NCC (+5°) and Edge-NCC (−5°) biases cancel in the Combined metric → 0° error",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig8_metric_bias.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path} ({out_path.stat().st_size // 1024} KB)")

    # テキストサマリー
    print("\nメトリクスバイアスサマリー (標準ポジショニング n=3):")
    print(f"{'Metric':<20} {'Bias':>8} {'MAE':>8}")
    print("-" * 38)
    for m, bias, mae in zip(metrics, mean_biases, mae_std):
        print(f"{metric_labels[m].replace(chr(10),' '):<20} {bias:>+8.2f}° {mae:>7.2f}°")


if __name__ == "__main__":
    main()
