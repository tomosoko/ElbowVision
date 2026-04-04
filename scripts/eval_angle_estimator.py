"""
ElbowVision ConvNeXt 角度推定モデル評価スクリプト

runs/angle_estimator/best.pth を使って data/angle_dataset/val.csv 全273枚を推論し、
GT vs Pred CSV を生成後、Bland-Altman 解析を実行する。

使い方:
  cd ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/eval_angle_estimator.py

出力:
  results/bland_altman/
  ├── predictions.csv          ← GT / Pred 全273行
  ├── bland_altman_flexion.png ← Bland-Altman プロット
  └── summary.txt              ← Bias, LoA, ICC, MAE, RMSE, r
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── パス設定 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(PROJECT_ROOT, "runs", "angle_estimator", "best.pth")
VAL_CSV     = os.path.join(PROJECT_ROOT, "data", "angle_dataset", "val.csv")
IMG_DIR     = os.path.join(PROJECT_ROOT, "data", "angle_dataset", "images")
OUT_DIR     = os.path.join(PROJECT_ROOT, "results", "bland_altman")

# 訓練時の正規化パラメータ（memory: norm = (angle - 90) / 90）
ANGLE_MIN = 90.0
ANGLE_MAX = 180.0

# ── モデル定義 ────────────────────────────────────────────────────────────────


class AngleEstimator(nn.Module):
    """ConvNeXt-Small バックボーン + 単一出力（屈曲角）"""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.convnext_small(weights=None)
        in_features = backbone.classifier[2].in_features  # 768
        backbone.classifier[2] = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)       # [B, 1]
        return out.squeeze(1)        # [B]


# ── 前処理 ────────────────────────────────────────────────────────────────────

# ConvNeXt-Small ImageNet 標準前処理
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(img_path: str) -> torch.Tensor:
    """グレースケール PNG を RGB に変換して前処理テンソルを返す"""
    img = Image.open(img_path).convert("RGB")
    return _transform(img)


# ── 推論 ──────────────────────────────────────────────────────────────────────


class ValDataset(Dataset):
    """angle_dataset/val.csv + images/ からデータを読み込む Dataset"""

    def __init__(self, val_csv: str, img_dir: str,
                 transform: transforms.Compose) -> None:
        self.df = pd.read_csv(val_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        gt = float(row["angle_deg"])
        return tensor, gt, row["filename"]


def run_inference(model: nn.Module,
                  val_csv: str,
                  img_dir: str,
                  device: torch.device,
                  batch_size: int = 32) -> pd.DataFrame:
    """val.csv の全画像を推論し GT / Pred DataFrame を返す"""
    dataset = ValDataset(val_csv, img_dir, _transform)

    # MPSではspawnベースのnum_workersは不安定なので0
    # CPU推論時は4並列で高速化可能
    num_workers = 0 if device.type == "mps" else 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    pred_angles: list[float] = []
    gt_angles: list[float] = []
    filenames: list[str] = []

    total = len(dataset)
    processed = 0
    for tensors, gts, fnames in loader:
        tensors = tensors.to(device)
        with torch.no_grad():
            out = model(tensors)  # [B] - 正規化空間

        # 逆正規化: pred_angle = out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
        pred = out.cpu().numpy() * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
        pred_angles.extend(pred.tolist())
        gt_angles.extend(gts.tolist())
        filenames.extend(list(fnames))

        processed += len(tensors)
        print(f"  推論中... {processed}/{total}", end="\r")

    print()

    result_df = pd.DataFrame({
        "filename":         filenames,
        "gt_flexion_deg":   gt_angles,
        "pred_flexion_deg": pred_angles,
    })
    return result_df


# ── メイン ────────────────────────────────────────────────────────────────────


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # デバイス選択
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"デバイス: {device}")

    # モデルロード
    print(f"モデルロード: {os.path.relpath(MODEL_PATH, PROJECT_ROOT)}")
    model = AngleEstimator()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    print("  → ロード完了")

    # 推論
    print(f"\n推論開始: {os.path.relpath(VAL_CSV, PROJECT_ROOT)} ({VAL_CSV})")
    result_df = run_inference(model, VAL_CSV, IMG_DIR, device)

    # 基本統計
    gt   = result_df["gt_flexion_deg"].to_numpy()
    pred = result_df["pred_flexion_deg"].to_numpy()
    errors = np.abs(gt - pred)
    print(f"\n【推論結果サマリー】")
    print(f"  n        = {len(gt)}")
    print(f"  MAE      = {errors.mean():.3f} °")
    print(f"  RMSE     = {np.sqrt((errors**2).mean()):.3f} °")
    print(f"  <5°      = {(errors < 5).mean() * 100:.1f}%")
    print(f"  <10°     = {(errors < 10).mean() * 100:.1f}%")

    # CSV 保存
    pred_csv = os.path.join(OUT_DIR, "predictions.csv")
    result_df.to_csv(pred_csv, index=False)
    print(f"\n予測CSV保存: {os.path.relpath(pred_csv, PROJECT_ROOT)}")

    # Bland-Altman 解析
    print("\nBland-Altman 解析を実行中...")
    ba_script = os.path.join(PROJECT_ROOT, "scripts", "bland_altman.py")
    if not os.path.exists(ba_script):
        print(f"  ERROR: bland_altman.py が見つかりません: {ba_script}")
        sys.exit(1)

    # bland_altman.py は gt_flexion_deg / pred_flexion_deg 列をそのまま使う
    sys.path.insert(0, os.path.dirname(ba_script))
    from bland_altman import run_analysis
    run_analysis(pred_csv, OUT_DIR)

    print(f"\n完了。結果: {os.path.relpath(OUT_DIR, PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
