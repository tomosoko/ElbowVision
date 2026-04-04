"""
ConvNeXt直接推定 vs 類似度マッチング 比較図生成

results/method_comparison/comparison.json の結果から
JSRT発表・論文用の比較ビジュアルを生成する。

使い方:
  python scripts/generate_comparison_figure.py
  python scripts/generate_comparison_figure.py \
    --json results/method_comparison/comparison.json \
    --out_dir results/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Journal品質
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":   300,
    "savefig.dpi":  300,
})

LABEL_MAP = {
    "008_LAT.png":          "Std-1",
    "cr_008_2_50kVp.png":   "Non-std",
    "cr_008_3_52kVp.png":   "Std-2",
    "new_LAT.png":          "Std-3",
}

COLOR_CNX = "#2196F3"   # 青: ConvNeXt
COLOR_SIM = "#4CAF50"   # 緑: Similarity
COLOR_GT  = "#FF5722"   # 赤: GT


def load_results(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    return data["results"]


def plot_per_image_bar(results: list[dict], out_path: str) -> None:
    """各画像の絶対誤差棒グラフ（ConvNeXt vs 類似度マッチング）"""
    labels = [LABEL_MAP.get(r["filename"], r["filename"]) for r in results]
    err_cnx = [r["err_convnext"] for r in results]
    err_sim = [r["err_sim"]      for r in results]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w/2, err_cnx, w, color=COLOR_CNX, label="ConvNeXt (direct regression)")
    bars2 = ax.bar(x + w/2, err_sim, w, color=COLOR_SIM, label="Similarity Matching (combined NCC)")

    # 値ラベル
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}°", ha="center", va="bottom", fontsize=9, color=COLOR_CNX)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}°", ha="center", va="bottom", fontsize=9, color=COLOR_SIM)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Error [°]")
    ax.set_title("Per-image Error: ConvNeXt vs Similarity Matching\n(Real Phantom X-rays, GT = 90°)")
    ax.legend()
    ax.set_ylim(0, max(max(err_cnx), max(err_sim)) * 1.25)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # 非標準画像をグレーでマーク
    nonstd_idx = [i for i, r in enumerate(results) if "non_standard" in r.get("filename", "")]
    for idx in [labels.index("Non-std")] if "Non-std" in labels else []:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.08, color="gray", zorder=0)
        ax.text(idx, ax.get_ylim()[1] * 0.97, "Non-std\nposition",
                ha="center", va="top", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_mae_summary(results: list[dict], out_path: str) -> None:
    """MAE比較（標準画像のみ vs 全体）"""
    std = [r for r in results if "cr_008_2" not in r["filename"]]
    all_ = results

    groups = [
        ("Standard LAT\n(n=3)", std),
        ("All images\n(n=4)", all_),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)

    for ax, (title, subset) in zip(axes, groups):
        mae_cnx = np.mean([r["err_convnext"] for r in subset])
        mae_sim = np.mean([r["err_sim"]      for r in subset])

        bars = ax.bar(
            ["ConvNeXt\n(direct)", "Similarity\n(combined NCC)"],
            [mae_cnx, mae_sim],
            color=[COLOR_CNX, COLOR_SIM],
            width=0.5,
        )
        for bar, val in zip(bars, [mae_cnx, mae_sim]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f"{val:.1f}°", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_title(title)
        ax.set_ylabel("MAE [°]")
        ax.set_ylim(0, max(mae_cnx, mae_sim) * 1.3 + 2)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # 臨床許容ライン（例: 5°）
    for ax in axes:
        ax.axhline(5, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="5° threshold")
        ax.legend(fontsize=9)

    fig.suptitle("Mean Absolute Error Comparison\nConvNeXt (DRR-trained, no fine-tuning) vs Similarity Matching",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_prediction_scatter(results: list[dict], out_path: str) -> None:
    """GT vs Pred 散布図（両手法）"""
    gt_vals   = [r["gt_angle"] for r in results]
    pred_cnx  = [r["pred_convnext"] for r in results]
    pred_sim  = [r["pred_sim"]      for r in results]
    labels    = [LABEL_MAP.get(r["filename"], r["filename"]) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, preds, color, title in [
        (axes[0], pred_cnx, COLOR_CNX, "ConvNeXt (direct regression)"),
        (axes[1], pred_sim, COLOR_SIM, "Similarity Matching (combined NCC)"),
    ]:
        for gt, pred, lbl in zip(gt_vals, preds, labels):
            marker = "D" if lbl == "Non-std" else "o"
            ax.scatter(gt, pred, s=80, color=color, marker=marker, zorder=3)
            ax.annotate(lbl, (gt, pred), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color="gray")

        lim_min = min(min(gt_vals), min(preds)) - 5
        lim_max = max(max(gt_vals), max(preds)) + 5
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, alpha=0.5, label="Identity line")
        ax.plot([lim_min, lim_max], [lim_min + 5, lim_max + 5], "--", color="orange", linewidth=0.8, alpha=0.5)
        ax.plot([lim_min, lim_max], [lim_min - 5, lim_max - 5], "--", color="orange", linewidth=0.8, alpha=0.5,
                label="±5° band")

        mae = np.mean([abs(p - g) for g, p in zip(gt_vals, preds)])
        ax.set_title(f"{title}\nMAE = {mae:.1f}°")
        ax.set_xlabel("GT Angle [°]")
        ax.set_ylabel("Predicted Angle [°]")
        ax.legend(fontsize=8)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="比較図生成")
    parser.add_argument("--json",    default="results/method_comparison/comparison.json")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    json_path = _PROJECT_ROOT / args.json
    out_dir   = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"ERROR: {json_path} が見つかりません。compare_methods.py を先に実行してください。")
        return

    results = load_results(str(json_path))
    print(f"結果読み込み: {len(results)} 画像")

    plot_per_image_bar(results, str(out_dir / "fig5a_per_image_error.png"))
    plot_mae_summary(results,   str(out_dir / "fig5b_mae_summary.png"))
    plot_prediction_scatter(results, str(out_dir / "fig5c_prediction_scatter.png"))

    print("\n完了。生成ファイル:")
    for f in ["fig5a_per_image_error.png", "fig5b_mae_summary.png", "fig5c_prediction_scatter.png"]:
        p = out_dir / f
        if p.exists():
            print(f"  {p} ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
