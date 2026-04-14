"""
ConvNeXt v6 評価スクリプト

elbow-api/elbow_convnext_best.pth を使って
data/yolo_dataset_v6/convnext_labels.csv のval分割全件を推論し、
GT vs Pred CSV + Bland-Altman 解析を実行する。

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  python scripts/eval_convnext_v6.py

出力:
  results/bland_altman/
  ├── predictions_v6.csv          ← GT / Pred (LAT flexion)
  ├── bland_altman_v6_flexion.png ← Bland-Altman プロット
  └── summary_v6.txt              ← Bias, LoA, ICC, MAE, RMSE (上書き)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "elbow-api" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

MODEL_PATH = PROJECT_ROOT / "elbow-api" / "elbow_convnext_best.pth"
VAL_CSV    = PROJECT_ROOT / "data" / "yolo_dataset_v6" / "convnext_labels.csv"
IMG_DIR    = PROJECT_ROOT / "data" / "yolo_dataset_v6" / "images"
OUT_DIR    = PROJECT_ROOT / "results" / "bland_altman"

# ConvNeXt 推論用 transforms（train_angle_predictor.py と同じ）
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path):
        # LAT 像のみ（屈曲角評価）
        self.df = df[df["view_type"].str.upper() == "LAT"].reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # images/val/ or images/train/ subdirectory
        split = row.get("split", "val")
        img_path = self.img_dir / split / row["filename"]
        if not img_path.exists():
            img_path = self.img_dir / row["filename"]
        img = Image.open(img_path).convert("RGB")
        img = VAL_TRANSFORMS(img)
        gt = float(row["flexion_deg"])
        return img, gt, row["filename"]


def load_model(model_path: Path, device: torch.device):
    from convnext_model import ElbowConvNeXt
    model = ElbowConvNeXt(pretrained=False)
    ckpt = torch.load(str(model_path), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def compute_bland_altman_stats(gt: np.ndarray, pred: np.ndarray) -> dict:
    from bland_altman import compute_bland_altman
    result = compute_bland_altman(gt, pred)
    return {
        "n":         result.n,
        "mean_bias": result.mean_diff,
        "sd_diff":   result.std_diff,
        "loa_upper": result.loa_upper,
        "loa_lower": result.loa_lower,
        "mae":       result.mae,
        "rmse":      result.rmse,
        "pearson_r": result.pearson_r,
        "r_squared": result.r_squared,
        "icc_31":    result.icc,
    }


def save_bland_altman_plot(gt: np.ndarray, pred: np.ndarray, stats: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi":     300,
        "savefig.dpi":    300,
    })

    mean_vals = (gt + pred) / 2
    diff_vals = pred - gt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: GT vs Pred scatter
    ax = axes[0]
    ax.scatter(gt, pred, color="#3b82f6", alpha=0.5, s=20,
               edgecolors="white", linewidths=0.3, zorder=3)
    lim = [min(gt.min(), pred.min()) - 3, max(gt.max(), pred.max()) + 3]
    ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="Identity")
    ax.set_xlabel("Ground Truth Flexion [°]")
    ax.set_ylabel("Predicted Flexion [°]")
    ax.set_title("(A) GT vs Predicted")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    stats_text = (
        f"n = {stats['n']}\n"
        f"MAE = {stats['mae']:.3f}°\n"
        f"ICC(3,1) = {stats['icc_31']:.4f}\n"
        f"$r^2$ = {stats['r_squared']:.4f}"
    )
    ax.text(0.04, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.9))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel B: Bland-Altman
    ax = axes[1]
    ax.scatter(mean_vals, diff_vals, color="#3b82f6", alpha=0.5, s=20,
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.axhline(stats["mean_bias"], color="#16a34a", linewidth=2.0,
               label=f"Bias: {stats['mean_bias']:+.3f}°")
    ax.axhline(stats["loa_upper"], color="#dc2626", linewidth=1.5, linestyle="--",
               label=f"Upper LoA: {stats['loa_upper']:+.3f}°")
    ax.axhline(stats["loa_lower"], color="#dc2626", linewidth=1.5, linestyle="--",
               label=f"Lower LoA: {stats['loa_lower']:+.3f}°")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_xlabel("Mean of GT and Prediction [°]")
    ax.set_ylabel("Difference (Pred − GT) [°]")
    ax.set_title("(B) Bland-Altman Plot")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"ConvNeXt-Small v6 — LAT Flexion Angle Estimation\n"
        f"DRR Validation Set (n={stats['n']})",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # デバイス
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # データ読み込み
    df = pd.read_csv(VAL_CSV)
    df_val = df[df["split"] == "val"]
    print(f"Val total: {len(df_val)} → LAT only: {(df_val['view_type'].str.upper() == 'LAT').sum()}")

    dataset = ValDataset(df_val, IMG_DIR)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Evaluating {len(dataset)} LAT val images...")

    # モデルロード
    model = load_model(MODEL_PATH, device)

    all_preds = []
    all_gts   = []
    all_fnames = []

    with torch.no_grad():
        for imgs, gts, fnames in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            # preds[:, 1] = flexion_deg
            all_preds.extend(preds[:, 1].tolist())
            all_gts.extend(gts.numpy().tolist())
            all_fnames.extend(list(fnames))

    gt_arr   = np.array(all_gts)
    pred_arr = np.array(all_preds)

    # CSV保存
    csv_path = OUT_DIR / "predictions_v6.csv"
    pd.DataFrame({
        "filename":       all_fnames,
        "gt_flexion_deg": gt_arr,
        "pred_flexion_deg": pred_arr,
    }).to_csv(str(csv_path), index=False)
    print(f"CSV保存: {csv_path} ({len(all_fnames)} 行)")

    # 統計
    stats = compute_bland_altman_stats(gt_arr, pred_arr)

    # サマリーテキスト
    summary_path = OUT_DIR / "summary_v6.txt"
    lines = [
        "=" * 60,
        "ElbowVision Bland-Altman Analysis Summary — ConvNeXt v6",
        "=" * 60,
        "",
        f"--- flexion_deg (n={stats['n']}) ---",
        f"  Mean Bias      : {stats['mean_bias']:.3f} deg",
        f"  SD of Diff     : {stats['sd_diff']:.3f} deg",
        f"  95% LoA        : [{stats['loa_lower']:.3f}, {stats['loa_upper']:.3f}] deg",
        f"  MAE            : {stats['mae']:.3f} deg",
        f"  RMSE           : {stats['rmse']:.3f} deg",
        f"  Pearson r      : {stats['pearson_r']:.4f}",
        f"  r^2            : {stats['r_squared']:.4f}",
        f"  ICC(3,1)       : {stats['icc_31']:.4f}",
        "  --- Clinical Threshold ---",
        f"  Bias <=+/-3.0 deg : {'PASS' if abs(stats['mean_bias']) <= 3.0 else 'FAIL'} ({stats['mean_bias']:.3f})",
        f"  LoA  <=+/-8.0 deg : {'PASS' if max(abs(stats['loa_lower']), abs(stats['loa_upper'])) <= 8.0 else 'FAIL'} (max={max(abs(stats['loa_lower']), abs(stats['loa_upper'])):.3f})",
    ]
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text)
    print(text)

    # 図生成
    plot_path = OUT_DIR / "bland_altman_v6_flexion.png"
    save_bland_altman_plot(gt_arr, pred_arr, stats, plot_path)

    print(f"\n完了:")
    print(f"  CSV:   {csv_path}")
    print(f"  Plot:  {plot_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
