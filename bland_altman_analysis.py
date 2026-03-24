"""
ElbowVision Bland-Altman 解析スクリプト

使い方:
  python bland_altman_analysis.py --csv results.csv

CSV形式（ヘッダー必須）:
  image_id, manual_carrying, ai_carrying,
             manual_flexion,  ai_flexion,
             manual_pronation, ai_pronation

出力:
  - validation_output/bland_altman_carrying.png
  - validation_output/bland_altman_flexion.png
  - validation_output/bland_altman_pronation.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "validation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def bland_altman_plot(manual: np.ndarray, ai: np.ndarray, angle_name: str, unit: str = "°"):
    mean = (manual + ai) / 2
    diff = ai - manual
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean, diff, color="#3b82f6", alpha=0.7, s=40, zorder=5)
    ax.axhline(mean_diff, color="#22c55e", linewidth=2, label=f"平均差: {mean_diff:.2f}{unit}")
    ax.axhline(loa_upper, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"+1.96SD: {loa_upper:.2f}{unit}")
    ax.axhline(loa_lower, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"-1.96SD: {loa_lower:.2f}{unit}")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f"平均 ({angle_name}) [{unit}]", fontsize=12)
    ax.set_ylabel(f"差（AI - 手動計測） [{unit}]", fontsize=12)
    ax.set_title(f"Bland-Altman プロット — {angle_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 統計情報
    n = len(diff)
    ci_mean = 1.96 * std_diff / np.sqrt(n)
    ax.text(
        0.02, 0.97,
        f"n={n}  |  LOA: [{loa_lower:.2f}, {loa_upper:.2f}]{unit}  |  95%CI mean: ±{ci_mean:.2f}{unit}",
        transform=ax.transAxes, fontsize=9, va="top", color="#9ca3af",
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"bland_altman_{angle_name.lower().replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  平均差: {mean_diff:.3f}{unit}")
    print(f"  LoA:    [{loa_lower:.3f}, {loa_upper:.3f}]{unit}")
    print(f"  SD:     {std_diff:.3f}{unit}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ElbowVision Bland-Altman 解析")
    parser.add_argument("--csv", required=True, help="結果CSVファイルのパス")
    args = parser.parse_args()

    import csv
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("CSVにデータがありません。")
        return

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    print(f"=== ElbowVision Bland-Altman 解析 (n={len(rows)}) ===\n")

    bland_altman_plot(col("manual_carrying"), col("ai_carrying"), "carrying_angle")
    bland_altman_plot(col("manual_flexion"),  col("ai_flexion"),  "flexion")
    bland_altman_plot(col("manual_pronation"), col("ai_pronation"), "pronation_sup")

    print("完了。")


if __name__ == "__main__":
    main()
