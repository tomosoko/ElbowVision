"""
ElbowVision 論文用図表生成スクリプト

使い方:
  cd ~/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/generate_figures.py [--out_dir results/figures]

出力:
  results/figures/
  ├── fig1_pipeline.png          # システムパイプライン図
  ├── fig2_drr_algorithm.png     # DRR生成アルゴリズム4パネル
  ├── fig3_drr_variations.png    # DRR角度バリエーション8パネル
  └── fig4_bland_altman.png      # Bland-Altmanプロット
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# -- 論文フォーマット設定 --
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# =====================================================================
# Figure 1: System Pipeline
# =====================================================================

def generate_fig1_pipeline(out_path: str) -> None:
    """CT -> DRR -> YOLO -> ConvNeXt -> Inference のフロー図。"""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # ボックス定義: (x_center, y_center, label, color)
    boxes = [
        (1.2, 2.5, "CT Acquisition\n(Elbow Phantom)", "#E3F2FD"),
        (3.5, 2.5, "DRR Generation\n(ElbowSynth)", "#E8F5E9"),
        (5.8, 2.5, "ConvNeXt-Small\nTraining", "#F3E5F5"),
        (8.2, 2.5, "Flexion Angle\nEstimation", "#FFEBEE"),
    ]

    box_w, box_h = 1.9, 1.0

    for xc, yc, label, color in boxes:
        rect = FancyBboxPatch(
            (xc - box_w / 2, yc - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="#333333", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(xc, yc, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#333333")

    # 矢印（箱から箱へ）
    arrows = [
        ((2.15, 2.5), (2.55, 2.5)),   # CT -> DRR
        ((4.45, 2.5), (4.85, 2.5)),   # DRR -> ConvNeXt
        ((6.75, 2.5), (7.25, 2.5)),   # ConvNeXt -> Inference
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color="#555555",
                                     lw=1.5, connectionstyle="arc3,rad=0.0"))

    # 下段: データフロー注釈
    annotations = [
        (1.2, 1.5, "DICOM volume\n(extension CT)", "#78909C"),
        (3.5, 1.5, "Synthetic X-rays\n+ auto-labels (n=1,365)", "#78909C"),
        (5.8, 1.5, "ImageNet\npre-trained", "#78909C"),
        (8.2, 1.5, "MAE 1.41°\nICC 0.9989", "#2e7d32"),
    ]
    for xc, yc, text, color in annotations:
        ax.text(xc, yc, text, ha="center", va="center",
                fontsize=7, color=color, style="italic")
        ax.annotate("", xy=(xc, 1.9), xytext=(xc, 2.05),
                     arrowprops=dict(arrowstyle="-", color="#BDBDBD",
                                     lw=0.8, linestyle="--"))

    # タイトル
    ax.text(5.0, 3.3, "Fig. 1  ElbowVision System Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Figure 2: DRR Generation Algorithm (4-panel)
# =====================================================================

def generate_fig2_drr_algorithm(out_path: str) -> None:
    """DRR生成の4ステップを可視化（模式図）。"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    panel_titles = [
        "(a) CT Volume",
        "(b) 3D Rotation",
        "(c) Beer-Lambert\nProjection",
        "(d) Post-processing",
    ]

    for i, (ax, title) in enumerate(zip(axes, panel_titles)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

        if i == 0:
            # CT Volume: 3Dボックス
            _draw_3d_box(ax, 2, 2, 6, 6, "#90CAF9")
            ax.text(5, 5, "3D CT\nVolume", ha="center", va="center",
                    fontsize=8, fontweight="bold")
            ax.text(5, 1.0, "HU clipping\n[-400, 1500]", ha="center",
                    fontsize=7, color="#666666")

        elif i == 1:
            # Rotation: 回転する骨の模式図
            _draw_3d_box(ax, 2, 2, 6, 6, "#A5D6A7")
            # 回転矢印
            angle = np.linspace(0, 1.5 * np.pi, 50)
            r = 2.5
            ax.plot(5 + r * np.cos(angle), 5 + r * np.sin(angle),
                    color="#E65100", lw=1.5)
            ax.annotate("", xy=(5 + r * np.cos(1.5 * np.pi),
                               5 + r * np.sin(1.5 * np.pi)),
                        xytext=(5 + r * np.cos(1.4 * np.pi),
                                5 + r * np.sin(1.4 * np.pi)),
                        arrowprops=dict(arrowstyle="->", color="#E65100",
                                        lw=1.5))
            ax.text(5, 5, "Flexion\n+Rotation", ha="center", va="center",
                    fontsize=8, fontweight="bold")
            ax.text(5, 1.0, "affine_transform()\nSciPy", ha="center",
                    fontsize=7, color="#666666")

        elif i == 2:
            # Beer-Lambert: 透過投影
            # X線源
            ax.plot(5, 9, "r^", markersize=10)
            ax.text(5, 9.5, "X-ray source", ha="center", fontsize=7,
                    color="#D32F2F")
            # 光線
            for dx in np.linspace(-2, 2, 7):
                ax.plot([5, 5 + dx], [9, 1.5], color="#FFCDD2",
                        lw=0.5, alpha=0.7)
            # ボリューム
            rect = FancyBboxPatch((2.5, 3.5), 5, 3.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor="#A5D6A7", alpha=0.5,
                                  edgecolor="#333333", lw=0.8)
            ax.add_patch(rect)
            ax.text(5, 5.25, "Volume", ha="center", fontsize=8)
            # 検出器
            ax.plot([2, 8], [1.5, 1.5], color="#1565C0", lw=3)
            ax.text(5, 0.8, "Detector (SID=1000mm)", ha="center",
                    fontsize=7, color="#1565C0")

        elif i == 3:
            # Post-processing: DRR画像
            # 模式的なDRR画像（ノイズ+グラデーション）
            np.random.seed(42)
            img = np.random.normal(0.5, 0.15, (64, 48))
            # 骨の模式（中央に明るいライン）
            img[10:55, 20:28] += 0.3
            img[28:35, 12:36] += 0.25
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap="gray", extent=[1.5, 8.5, 1, 9],
                      aspect="auto")
            ax.text(5, 0.3, "CLAHE + resize\n+ augmentation", ha="center",
                    fontsize=7, color="#666666")

    fig.suptitle("Fig. 2  DRR Generation Algorithm", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _draw_3d_box(ax, x, y, w, h, color):
    """簡易3Dボックスを描画。"""
    d = 0.8  # 奥行きオフセット
    # 正面
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333",
                          lw=0.8, alpha=0.8)
    ax.add_patch(rect)
    # 上面
    xs = [x, x + d, x + w + d, x + w]
    ys = [y + h, y + h + d, y + h + d, y + h]
    ax.fill(xs, ys, facecolor=color, edgecolor="#333333", lw=0.8, alpha=0.5)
    # 右面
    xs2 = [x + w, x + w + d, x + w + d, x + w]
    ys2 = [y, y + d, y + h + d, y + h]
    ax.fill(xs2, ys2, facecolor=color, edgecolor="#333333", lw=0.8, alpha=0.6)


# =====================================================================
# Figure 3: DRR Variations (8 panels: 4 flexion x AP/LAT)
# =====================================================================

def generate_fig3_drr_variations(out_path: str) -> None:
    """DRR生成パラメータ変化の8パネル図。実画像があれば使用、なければダミー。"""
    import csv
    import cv2

    dataset_dir = os.path.join(PROJECT_ROOT, "data", "yolo_dataset")
    summary_csv = os.path.join(dataset_dir, "dataset_summary.csv")
    images_dir = os.path.join(dataset_dir, "images")

    target_flexions = [0, 30, 60, 90]  # 目標屈曲角
    views = ["AP", "LAT"]

    # CSVから各条件に近い画像を探す
    selected_images: dict[tuple[str, int], str | None] = {}
    for v in views:
        for f in target_flexions:
            selected_images[(v, f)] = None

    if os.path.isfile(summary_csv):
        with open(summary_csv, "r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        for v in views:
            for target_f in target_flexions:
                best_row = None
                best_dist = float("inf")
                for row in rows:
                    if row["view_type"] != v:
                        continue
                    flex = float(row["flexion_deg"])
                    # base_flexionを考慮: 実際の屈曲角 = base_flexion - rotation的な補正
                    dist = abs(flex - target_f)
                    if dist < best_dist:
                        best_dist = dist
                        best_row = row
                if best_row is not None:
                    # train/valの両方を探す
                    fname = best_row["filename"]
                    for split in ["train", "val"]:
                        p = os.path.join(images_dir, split, fname)
                        if os.path.isfile(p):
                            selected_images[(v, target_f)] = p
                            break

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))

    for row_idx, view in enumerate(views):
        for col_idx, flex in enumerate(target_flexions):
            ax = axes[row_idx, col_idx]
            img_path = selected_images.get((view, flex))

            if img_path and os.path.isfile(img_path):
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        ax.imshow(img, cmap="gray")
                    else:
                        _draw_placeholder(ax)
                except Exception:
                    _draw_placeholder(ax)
            else:
                _draw_placeholder(ax)

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Flexion {flex}$^\\circ$", fontsize=10,
                             fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{view} view", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Fig. 3  DRR Images at Various Flexion Angles",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _draw_placeholder(ax):
    """画像がない場合のプレースホルダ。"""
    ax.set_facecolor("#F5F5F5")
    ax.text(0.5, 0.5, "No image\navailable", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="#999999",
            style="italic")


# =====================================================================
# Figure 4: Bland-Altman Plot
# =====================================================================

def generate_fig4_bland_altman(out_path: str) -> None:
    """Bland-Altmanプロット。実データ(results/bland_altman/predictions.csv)を使用。"""
    import pandas as pd

    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from bland_altman import compute_bland_altman

    # 実データ読み込み
    pred_csv = os.path.join(PROJECT_ROOT, "results", "bland_altman", "predictions.csv")
    if not os.path.isfile(pred_csv):
        print(f"  WARNING: 実データが見つかりません: {pred_csv}")
        print("  先に scripts/eval_angle_estimator.py を実行してください")
        return

    df = pd.read_csv(pred_csv)
    gt_flexion   = df["gt_flexion_deg"].to_numpy(dtype=float)
    pred_flexion = df["pred_flexion_deg"].to_numpy(dtype=float)

    result = compute_bland_altman(gt_flexion, pred_flexion)
    mean_vals = (gt_flexion + pred_flexion) / 2
    diff_vals = pred_flexion - gt_flexion

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Scatter
    ax.scatter(mean_vals, diff_vals, color="#3b82f6", alpha=0.65, s=30,
               edgecolors="white", linewidths=0.4, zorder=5)

    # 平均バイアス
    ax.axhline(result.mean_diff, color="#16a34a", linewidth=2.0,
               label=f"Mean bias: {result.mean_diff:+.2f}°")

    # 95% LoA
    ax.axhline(result.loa_upper, color="#dc2626", linewidth=1.5,
               linestyle="--",
               label=f"Upper LoA: {result.loa_upper:+.2f}°")
    ax.axhline(result.loa_lower, color="#dc2626", linewidth=1.5,
               linestyle="--",
               label=f"Lower LoA: {result.loa_lower:+.2f}°")

    # ゼロライン
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5, linestyle=":")

    ax.set_xlabel("Mean of CT-derived Ground Truth and Prediction [°]")
    ax.set_ylabel("Difference (Prediction − Ground Truth) [°]")
    ax.set_title("Flexion Angle Estimation", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    # 統計サマリーテキスト
    stats_text = (
        f"n = {result.n}\n"
        f"MAE = {result.mae:.3f}°\n"
        f"RMSE = {result.rmse:.3f}°\n"
        f"ICC(3,1) = {result.icc:.4f}\n"
        f"$r^2$ = {result.r_squared:.4f}"
    )
    ax.text(0.03, 0.03, stats_text, transform=ax.transAxes,
            fontsize=8.5, va="bottom", color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.9))

    fig.suptitle(
        "Fig. 4  Bland-Altman Analysis: DRR Validation Set (n=273)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_fig4_bland_altman_v6(out_path: str) -> None:
    """Bland-Altmanプロット v6。実データ(results/bland_altman/predictions_v6.csv)を使用。"""
    import pandas as pd

    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from bland_altman import compute_bland_altman

    pred_csv = os.path.join(PROJECT_ROOT, "results", "bland_altman", "predictions_v6.csv")
    if not os.path.isfile(pred_csv):
        print(f"  WARNING: v6 predictions CSVが見つかりません: {pred_csv}")
        print("  先に scripts/eval_convnext_v6.py を実行してください")
        return

    df = pd.read_csv(pred_csv)
    gt_flexion   = df["gt_flexion_deg"].to_numpy(dtype=float)
    pred_flexion = df["pred_flexion_deg"].to_numpy(dtype=float)

    result = compute_bland_altman(gt_flexion, pred_flexion)
    mean_vals = (gt_flexion + pred_flexion) / 2
    diff_vals = pred_flexion - gt_flexion

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: GT vs Pred
    ax = axes[0]
    ax.scatter(gt_flexion, pred_flexion, color="#3b82f6", alpha=0.4, s=18,
               edgecolors="none", zorder=5)
    lim = [min(gt_flexion.min(), pred_flexion.min()) - 3,
           max(gt_flexion.max(), pred_flexion.max()) + 3]
    ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="Identity")
    ax.set_xlabel("Ground Truth Flexion [°]")
    ax.set_ylabel("Predicted Flexion [°]")
    ax.set_title("(A) GT vs Predicted", fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    stats_text = (
        f"n = {result.n}\n"
        f"MAE = {result.mae:.3f}°\n"
        f"ICC(3,1) = {result.icc:.4f}\n"
        f"$r^2$ = {result.r_squared:.4f}"
    )
    ax.text(0.04, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.9))

    # Panel B: Bland-Altman
    ax = axes[1]
    ax.scatter(mean_vals, diff_vals, color="#3b82f6", alpha=0.4, s=18,
               edgecolors="none", zorder=5)
    ax.axhline(result.mean_diff, color="#16a34a", linewidth=2.0,
               label=f"Bias: {result.mean_diff:+.3f}°")
    ax.axhline(result.loa_upper, color="#dc2626", linewidth=1.5, linestyle="--",
               label=f"Upper LoA: {result.loa_upper:+.3f}°")
    ax.axhline(result.loa_lower, color="#dc2626", linewidth=1.5, linestyle="--",
               label=f"Lower LoA: {result.loa_lower:+.3f}°")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_xlabel("Mean of GT and Prediction [°]")
    ax.set_ylabel("Difference (Pred − GT) [°]")
    ax.set_title("(B) Bland-Altman Plot", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Fig. 4  ConvNeXt-Small v6 — LAT Flexion Angle: Bland-Altman Analysis (n={result.n})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElbowVision - Generate publication-quality figures"
    )
    parser.add_argument(
        "--out_dir", default=os.path.join(PROJECT_ROOT, "results", "figures"),
        help="Output directory (default: results/figures/)",
    )
    parser.add_argument(
        "--fig", type=int, default=None,
        help="Generate only specific figure (1-5). Default: all",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    generators = {
        1: ("fig1_pipeline.png", generate_fig1_pipeline),
        2: ("fig2_drr_algorithm.png", generate_fig2_drr_algorithm),
        3: ("fig3_drr_variations.png", generate_fig3_drr_variations),
        4: ("fig4_bland_altman.png", generate_fig4_bland_altman),
        5: ("fig4_bland_altman_v6.png", generate_fig4_bland_altman_v6),
    }

    targets = [args.fig] if args.fig else sorted(generators.keys())

    for fig_num in targets:
        fname, func = generators[fig_num]
        out_path = os.path.join(args.out_dir, fname)
        print(f"\n--- Figure {fig_num}: {fname} ---")
        try:
            func(out_path)
        except Exception as e:
            print(f"ERROR generating Figure {fig_num}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
