"""
ElbowVision ConvNeXt-Small 訓練スクリプト

OsteoVision の train_angle_predictor.py を肘用に移植。

【使い方】
  # 訓練（アノテーション後）
  cd /Users/kohei/ElbowVision_Dev/elbow-api
  source venv/bin/activate
  python training/train_angle_predictor.py \\
    --csv  ../data/yolo_dataset/dataset_summary.csv \\
    --imgs ../data/images/

  # 保存先: elbow-api/elbow_convnext_best.pth（API が自動で読み込む）

【CSV形式】
  filename, view_type, carrying_angle, flexion, pronation_sup, varus_valgus
  AP_001.png, AP, 10.5, 0.0, 2.1, 1.3
  LAT_001.png, LAT, 0.0, 85.0, 1.5, 0.8

【訓練戦略】
  - ImageNet事前学習 ConvNeXt-Small をファインチューニング
  - AP/LAT で有効な出力チャンネルを損失マスクで制御
    AP  → carrying_angle のみL1損失を計算
    LAT → flexion のみL1損失を計算
    pronation_sup・varus_valgus は両方で学習
  - Early stopping (patience=20)
  - 最良モデルを elbow_convnext_best.pth に保存
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 同ディレクトリの convnext_model を参照
sys.path.insert(0, os.path.dirname(__file__))
from convnext_model import ElbowConvNeXt


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ElbowDataset(Dataset):
    """
    CSV + 画像ディレクトリからデータを読み込む Dataset。
    AP/LAT の view_type を取得し、損失マスクに使う。
    """
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 角度ラベル: [carrying_angle, flexion, pronation_sup, varus_valgus]
        angles = torch.tensor([
            float(row.get("carrying_angle", 0.0)),
            float(row.get("flexion", 0.0)),
            float(row.get("pronation_sup", 0.0)),
            float(row.get("varus_valgus", 0.0)),
        ], dtype=torch.float32)

        # 損失マスク: AP/LAT で有効なチャンネルだけ1
        view = str(row.get("view_type", "AP")).upper()
        mask = torch.tensor([
            1.0 if view == "AP"  else 0.0,  # carrying_angle
            1.0 if view == "LAT" else 0.0,  # flexion
            1.0,                             # pronation_sup (両方有効)
            1.0,                             # varus_valgus  (両方有効)
        ], dtype=torch.float32)

        return image, angles, mask


# ─── 訓練ループ ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(args.csv)
    val_mask = df.get("split", pd.Series(["train"] * len(df))) == "val"
    df_train = df[~val_mask]
    df_val   = df[val_mask]
    print(f"train: {len(df_train)}枚 / val: {len(df_val)}枚")

    df_train.to_csv("/tmp/_train_split.csv", index=False)
    df_val.to_csv(  "/tmp/_val_split.csv",   index=False)

    train_ds = ElbowDataset("/tmp/_train_split.csv", args.imgs, transform)
    val_ds   = ElbowDataset("/tmp/_val_split.csv",   args.imgs, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    model = ElbowConvNeXt(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_count = 0
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "elbow_convnext_best.pth")

    for epoch in range(1, args.epochs + 1):
        # ── 訓練 ──
        model.train()
        train_loss = 0.0
        for images, targets, masks in train_loader:
            images, targets, masks = images.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = (torch.abs(preds - targets) * masks).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── 検証 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, masks in val_loader:
                images, targets, masks = images.to(device), targets.to(device), masks.to(device)
                preds = model(images)
                val_loss += (torch.abs(preds - targets) * masks).mean().item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.3f}°  val={val_loss:.3f}°")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved ({best_val_loss:.3f}°) → {save_path}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n訓練完了。最良val loss: {best_val_loss:.3f}°")
    print(f"保存先: {save_path}")


# ─── エントリポイント ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ElbowVision ConvNeXt訓練")
    parser.add_argument("--csv",      required=True, help="データセットCSVパス")
    parser.add_argument("--imgs",     required=True, help="画像ディレクトリ")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
