"""
ElbowVision データセット統計可視化スクリプト

生成されたDRRデータセットの統計を可視化し、results/dataset_stats/ にPNG保存する。

使い方:
  cd /Users/kohei/Dev/vision/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/dataset_stats.py --dataset_dir data/drr_dataset_512
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset_csv(dataset_dir: str) -> pd.DataFrame:
    """dataset_summary.csv または convnext_labels.csv を読み込む。"""
    for name in ("dataset_summary.csv", "convnext_labels.csv"):
        csv_path = os.path.join(dataset_dir, name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"CSV読み込み: {csv_path} ({len(df)}件)")
            return df
    raise FileNotFoundError(
        f"dataset_summary.csv / convnext_labels.csv が {dataset_dir} に見つかりません"
    )


def plot_angle_distribution(df: pd.DataFrame, out_dir: str):
    """角度分布ヒストグラム（屈曲角・回旋誤差）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 屈曲角分布
    ax = axes[0]
    for view in ["AP", "LAT"]:
        subset = df[df["view_type"].str.upper() == view]
        if len(subset) > 0:
            ax.hist(subset["flexion_deg"], bins=30, alpha=0.6, label=f"{view} ({len(subset)}枚)")
    ax.set_xlabel("Flexion Angle (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Flexion Angle Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 回旋誤差分布
    ax = axes[1]
    for view in ["AP", "LAT"]:
        subset = df[df["view_type"].str.upper() == view]
        if len(subset) > 0:
            ax.hist(subset["rotation_error_deg"], bins=30, alpha=0.6, label=f"{view} ({len(subset)}枚)")
    ax.set_xlabel("Rotation Error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Rotation Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "angle_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {save_path}")


def plot_view_counts(df: pd.DataFrame, out_dir: str):
    """view別・split別の枚数バーチャート"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # View別
    ax = axes[0]
    view_counts = df["view_type"].str.upper().value_counts()
    colors = {"AP": "#4C72B0", "LAT": "#DD8452"}
    bars = ax.bar(view_counts.index, view_counts.values,
                  color=[colors.get(v, "#999999") for v in view_counts.index])
    for bar, count in zip(bars, view_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("View Type")
    ax.set_ylabel("Count")
    ax.set_title("Images per View Type")
    ax.grid(True, alpha=0.3, axis="y")

    # Split別
    ax = axes[1]
    if "split" in df.columns:
        split_view = df.groupby(["split", df["view_type"].str.upper()]).size().unstack(fill_value=0)
        split_view.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_xlabel("Split")
        ax.set_ylabel("Count")
        ax.set_title("Train/Val Split per View")
        ax.legend(title="View")
        ax.grid(True, alpha=0.3, axis="y")
        # 割合を表示
        for split_name in split_view.index:
            total = split_view.loc[split_name].sum()
            pct = total / len(df) * 100
            print(f"  {split_name}: {total}枚 ({pct:.1f}%)")
    else:
        ax.text(0.5, 0.5, "split列なし", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "view_split_counts.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {save_path}")


def plot_sample_grid(dataset_dir: str, df: pd.DataFrame, out_dir: str, grid_size: int = 4):
    """サンプル画像のグリッド表示"""
    import cv2

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))

    # AP/LAT からそれぞれ半分ずつサンプリング
    samples = []
    for view in ["AP", "LAT"]:
        subset = df[df["view_type"].str.upper() == view]
        if len(subset) > 0:
            n = min(grid_size * grid_size // 2, len(subset))
            samples.extend(subset.sample(n=n, random_state=42).to_dict("records"))

    # 足りない場合は全体から追加
    remaining = grid_size * grid_size - len(samples)
    if remaining > 0 and len(df) > len(samples):
        extra = df.sample(n=min(remaining, len(df)), random_state=123).to_dict("records")
        samples.extend(extra[:remaining])

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(samples):
            row = samples[idx]
            fname = row["filename"]
            split = row.get("split", "train")
            img_path = os.path.join(dataset_dir, "images", split, fname)
            if not os.path.exists(img_path):
                img_path = os.path.join(dataset_dir, "images", fname)

            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap="gray")
            view = row.get("view_type", "?")
            flex = row.get("flexion_deg", 0)
            rot = row.get("rotation_error_deg", 0)
            ax.set_title(f"{view} | flex={flex:.0f} rot={rot:.0f}", fontsize=9)

    plt.suptitle("Sample DRR Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(out_dir, "sample_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {save_path}")


def print_summary(df: pd.DataFrame):
    """統計サマリーをコンソール出力"""
    print("\n" + "=" * 50)
    print("  ElbowVision Dataset Statistics")
    print("=" * 50)
    print(f"  総画像数: {len(df)}")

    if "view_type" in df.columns:
        for view in df["view_type"].str.upper().unique():
            count = (df["view_type"].str.upper() == view).sum()
            print(f"  {view}: {count}枚")

    if "split" in df.columns:
        for split in df["split"].unique():
            count = (df["split"] == split).sum()
            pct = count / len(df) * 100
            print(f"  {split}: {count}枚 ({pct:.1f}%)")

    if "flexion_deg" in df.columns:
        print(f"\n  屈曲角: {df['flexion_deg'].min():.1f} - {df['flexion_deg'].max():.1f} deg")
        print(f"    mean={df['flexion_deg'].mean():.1f}, std={df['flexion_deg'].std():.1f}")

    if "rotation_error_deg" in df.columns:
        print(f"  回旋誤差: {df['rotation_error_deg'].min():.1f} - {df['rotation_error_deg'].max():.1f} deg")
        print(f"    mean={df['rotation_error_deg'].mean():.1f}, std={df['rotation_error_deg'].std():.1f}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="ElbowVision データセット統計可視化")
    parser.add_argument("--dataset_dir", required=True, help="データセットディレクトリ")
    parser.add_argument("--output_dir", default=None,
                        help="出力先（デフォルト: results/dataset_stats/）")
    args = parser.parse_args()

    # 出力先
    if args.output_dir:
        out_dir = args.output_dir
    else:
        project_root = Path(__file__).resolve().parent.parent
        out_dir = str(project_root / "results" / "dataset_stats")
    os.makedirs(out_dir, exist_ok=True)

    # データ読み込み
    df = load_dataset_csv(args.dataset_dir)
    print_summary(df)

    # プロット生成
    print("\nプロット生成中...")
    plot_angle_distribution(df, out_dir)
    plot_view_counts(df, out_dir)
    plot_sample_grid(args.dataset_dir, df, out_dir)

    print(f"\n完了。出力先: {out_dir}")


if __name__ == "__main__":
    main()
