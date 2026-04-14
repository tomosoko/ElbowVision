"""
YOLOv8-Pose v6 訓練曲線生成 (論文用)

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  python scripts/generate_training_curve_v6.py
"""
from __future__ import annotations

import csv
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


def load_results_csv(csv_path: Path) -> dict[str, list]:
    data: dict[str, list] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                k = k.strip()
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    return data


def generate_yolo_training_curve(out_path: Path) -> None:
    v6_csv = _PROJECT_ROOT / "runs" / "elbow_v6" / "results.csv"
    data = load_results_csv(v6_csv)

    epochs = data["epoch"]
    map50_b = data["metrics/mAP50(B)"]
    map50_p = data["metrics/mAP50(P)"]
    val_pose_loss = data["val/pose_loss"]
    train_pose_loss = data["train/pose_loss"]

    best_ep = int(np.argmax(map50_p)) + 1
    best_map = max(map50_p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: mAP50 curves
    ax = axes[0]
    ax.plot(epochs, map50_b, color="#2196F3", label="mAP50 (Box)", linewidth=1.8)
    ax.plot(epochs, map50_p, color="#4CAF50", label="mAP50 (Pose)", linewidth=1.8)
    ax.axvline(best_ep, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Best ep={best_ep}")
    ax.annotate(f"mAP50(P)={best_map:.3f}\n(ep {best_ep})",
                xy=(best_ep, best_map), xytext=(best_ep + 3, best_map - 0.03),
                fontsize=9, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title("(A) mAP50 Training Curve")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Panel B: Pose Loss curves
    ax = axes[1]
    ax.plot(epochs, train_pose_loss, color="#FF9800", label="Train pose loss", linewidth=1.8)
    ax.plot(epochs, val_pose_loss, color="#9C27B0", label="Val pose loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pose Loss")
    ax.set_title("(B) Pose Loss Curve")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("YOLOv8s-Pose v6 Training — ElbowVision\n"
                 f"mAP50(P)={best_map:.3f} at ep{best_ep} (early stop); "
                 "Train 3400/Val 600, 3-volume DRR",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def main() -> None:
    out_dir = _PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_yolo_training_curve(out_dir / "fig6_yolo_v6_training_curve.png")


if __name__ == "__main__":
    main()
