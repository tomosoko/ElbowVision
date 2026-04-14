"""
ConvNeXt v5/v6 vs 類似度マッチング 比較図生成 (v6対応版)

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  python scripts/generate_comparison_figure_v6.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":     300,
    "savefig.dpi":    300,
})

LABEL_MAP = {
    "008_LAT.png":          "Std-1",
    "cr_008_2_50kVp.png":   "Non-std",
    "cr_008_3_52kVp.png":   "Std-2",
    "new_LAT.png":          "Std-3",
}

COLOR_V5  = "#9E9E9E"   # グレー: ConvNeXt v5
COLOR_V6  = "#2196F3"   # 青: ConvNeXt v6
COLOR_SIM = "#4CAF50"   # 緑: Similarity Matching
CLINICAL_THRESHOLD = 5.0


def load_results(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    return data["results"]


def generate_fig5a_v6(results: list[dict], out_path: Path) -> None:
    """各画像の絶対誤差棒グラフ (v5 vs v6 vs 類似度マッチング)"""
    labels  = [LABEL_MAP.get(r["filename"], r["filename"]) for r in results]
    err_v5  = [abs(r["err_convnext_v5"]) if r.get("err_convnext_v5") is not None else 0 for r in results]
    err_v6  = [abs(r["err_convnext_v6"]) if r.get("err_convnext_v6") is not None else float("nan") for r in results]
    err_sim = [abs(r["err_sim"])         for r in results]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w, err_v5, w, color=COLOR_V5, label="ConvNeXt v5 (single-vol DRR, MAE=16.2°)", alpha=0.85)
    bars2 = ax.bar(x,     err_v6, w, color=COLOR_V6, label="ConvNeXt v6 (3-vol DRR+aug, MAE≈10.3°)")
    bars3 = ax.bar(x + w, err_sim, w, color=COLOR_SIM, label="Similarity Matching (NCC, MAE=0.0° std)")

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.1f}°", ha="center", va="bottom", fontsize=8, color=COLOR_V5)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.1f}°", ha="center", va="bottom", fontsize=8, color=COLOR_V6)
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}°", ha="center", va="bottom", fontsize=8, color=COLOR_SIM)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Error [°]")
    ax.set_title("Per-image Error: ConvNeXt v5/v6 vs Similarity Matching\n(Real Phantom Lateral X-rays, GT = 90°)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(CLINICAL_THRESHOLD, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="5° threshold")

    max_err = max([e for e in err_v5 + err_v6 + err_sim if not np.isnan(e)] + [0])
    ax.set_ylim(0, max_err * 1.3 + 5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # 非標準画像をグレーでマーク
    if "Non-std" in labels:
        idx = labels.index("Non-std")
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.08, color="gray", zorder=0)
        ax.text(idx, ax.get_ylim()[1] * 0.97, "Non-std\nposition",
                ha="center", va="top", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def generate_fig5b_v6_mae_summary(results: list[dict], out_path: Path) -> None:
    """MAE比較サマリー (v5 vs v6 vs 類似度マッチング) - 標準画像のみ"""
    std = [r for r in results if "cr_008_2" not in r["filename"]]
    std_v6 = [r for r in std if r.get("err_convnext_v6") is not None]

    mae_v5_std = np.mean([abs(r["err_convnext_v5"]) for r in std if r.get("err_convnext_v5") is not None])
    mae_v6_std = np.mean([abs(r["err_convnext_v6"]) for r in std_v6])
    mae_sim_std = np.mean([abs(r["err_sim"]) for r in std])

    # DRR validation set MAE
    drrval_v5 = 1.412
    drrval_v6 = 0.480

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: DRR validation set
    ax = axes[0]
    vals = [drrval_v5, drrval_v6]
    colors = [COLOR_V5, COLOR_V6]
    xlabels = ["ConvNeXt v5\n(n=273 val)", "ConvNeXt v6\n(n=600 val)"]
    bars = ax.bar(xlabels, vals, color=colors, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f"{val:.3f}°", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("(A) DRR Validation Set MAE")
    ax.set_ylabel("MAE [°]")
    ax.set_ylim(0, max(vals) * 1.5)
    ax.axhline(CLINICAL_THRESHOLD, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="5° threshold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    # Improvement arrow annotation
    ax.annotate("", xy=(1, drrval_v6 + 0.05), xytext=(0, drrval_v5 - 0.05),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(0.5, (drrval_v5 + drrval_v6) / 2, "−66%", ha="center", va="center",
            fontsize=10, color="red", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.8, boxstyle="round,pad=0.2"))

    # Panel B: Real phantom X-rays (standard positioning)
    ax = axes[1]
    vals2 = [mae_v5_std, mae_v6_std, mae_sim_std]
    colors2 = [COLOR_V5, COLOR_V6, COLOR_SIM]
    xlabels2 = [f"ConvNeXt v5\n(n={len(std)})", f"ConvNeXt v6\n(n={len(std_v6)})",
                f"Similarity\nMatching (n={len(std)})"]
    bars2 = ax.bar(xlabels2, vals2, color=colors2, width=0.5)
    for bar, val in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{val:.1f}°", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("(B) Real Phantom X-rays\n(Standard Positioning, GT = 90°)")
    ax.set_ylabel("MAE [°]")
    ax.set_ylim(0, max(vals2) * 1.5)
    ax.axhline(CLINICAL_THRESHOLD, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="5° threshold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("Method Comparison: ConvNeXt (DRR-trained) vs Similarity Matching\nElbow Flexion Angle Estimation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def main() -> None:
    json_path = _PROJECT_ROOT / "results" / "method_comparison" / "comparison.json"
    out_dir   = _PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(json_path)
    print(f"結果読み込み: {len(results)} 画像")

    generate_fig5a_v6(results, out_dir / "fig5a_per_image_error_v6.png")
    generate_fig5b_v6_mae_summary(results, out_dir / "fig5b_mae_summary_v6.png")

    print("\n完了:")
    for f in ["fig5a_per_image_error_v6.png", "fig5b_mae_summary_v6.png"]:
        p = out_dir / f
        if p.exists():
            print(f"  {p} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
