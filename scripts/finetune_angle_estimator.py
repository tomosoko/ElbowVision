"""
ConvNeXt 屈曲角推定モデル Fine-tuning（実X線データ対応）

Phase 2（30+枚の実X線 + GT角度）向けの fine-tuning パイプライン。
DRR訓練済みモデルのドメインギャップを実X線でのfine-tuningで縮小する。

戦略:
  - backbone の最終2ブロックのみ学習可能、それ以前は凍結（feature extraction保持）
  - 低学習率（DRR学習率の1/10 = 3e-6）で catastrophic forgetting を防止
  - 20倍データ拡張（rot±15, flip, CLAHE強度変化, ノイズ, brightness/contrast）
  - GT角度CSVをそのまま読む（ground_truth.csv と同じ形式）

使い方:
  # Phase 2データ取得後
  python scripts/finetune_angle_estimator.py \
    --real_xray_dir  data/real_xray/images/ \
    --gt_csv         data/real_xray/ground_truth.csv \
    --base_model     runs/angle_estimator/best.pth \
    --out_dir        runs/angle_estimator_finetuned/ \
    --epochs         100 --lr 3e-6 --batch_size 8

GT CSV形式（角度つき）:
  filename, gt_angle, note
  008_LAT.png, 90.0, standard_LAT
  img_120deg.png, 120.0, standard_LAT
  ...
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── モデル定義（eval_angle_estimator.py と共通） ────────────────────────────

def build_model():
    import torch.nn as nn
    import torchvision.models as tvm

    class AngleEstimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tvm.convnext_small(weights=None)
            self.backbone.classifier[2] = nn.Linear(768, 1)

        def forward(self, x):
            return self.backbone(x).squeeze(-1)

    return AngleEstimator()


def freeze_backbone_partial(model, unfreeze_last_n_stages: int = 2):
    """
    ConvNeXt-Small は features[0..7] の 8ブロック構成。
    最終 unfreeze_last_n_stages ブロックのみ学習可能にする。
    """
    import torch.nn as nn

    # まず全パラメータを凍結
    for p in model.parameters():
        p.requires_grad_(False)

    # classifier（出力層）は常に学習可能
    for p in model.backbone.classifier.parameters():
        p.requires_grad_(True)

    # 最終Nブロックを解凍
    features = list(model.backbone.features.children())
    n_blocks  = len(features)
    start_idx = max(0, n_blocks - unfreeze_last_n_stages)
    for i in range(start_idx, n_blocks):
        for p in features[i].parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  学習可能パラメータ: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    return model


# ── データセット ────────────────────────────────────────────────────────────

class RealXrayDataset:
    """実X線画像 + GT角度のデータセット（augmentation込み）"""

    ANGLE_MIN = 90.0
    ANGLE_MAX = 180.0

    def __init__(self, gt_csv: str, img_dir: str, augment: bool = True, aug_factor: int = 20):
        self.img_dir = Path(img_dir)
        self.augment = augment
        self.aug_factor = aug_factor

        # CSV読み込み
        self.samples: list[tuple[str, float]] = []
        with open(gt_csv) as f:
            for row in csv.DictReader(f):
                fname = row["filename"]
                angle = float(row["gt_angle"])
                if (self.img_dir / fname).exists():
                    self.samples.append((fname, angle))
                else:
                    print(f"  [SKIP] {fname} (not found)")

        if not self.samples:
            raise FileNotFoundError(f"GT CSVに有効な画像がありません: {gt_csv}")

        print(f"データセット: {len(self.samples)}枚（augment={augment}, ×{aug_factor}）")

    def __len__(self):
        return len(self.samples) * (self.aug_factor if self.augment else 1)

    def _load_and_preprocess(self, img_path: Path) -> np.ndarray:
        """読み込み → グレー → 256x256 → CLAHE → rot90CW"""
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(str(img_path))
        # 骨領域クロップ（背景除去）
        mask  = gray > 15
        rows  = np.any(mask, axis=1); cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            if ((rmax-rmin)*(cmax-cmin)) / (gray.shape[0]*gray.shape[1]) < 0.7:
                pad = int((rmax-rmin)*0.1)
                gray = gray[max(0,rmin-pad):rmax+pad, max(0,cmin-pad):cmax+pad]
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        return gray

    def _augment(self, gray: np.ndarray) -> np.ndarray:
        """ランダム拡張（回転±15°, flipLR, ノイズ, brightness/contrast）"""
        # 回転
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (256, 256), borderValue=0)

        # 水平フリップ（左右対称）
        if np.random.rand() < 0.5:
            gray = cv2.flip(gray, 1)

        # ガウシアンノイズ
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, np.random.uniform(0, 10), gray.shape).astype(np.float32)
            gray  = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 輝度・コントラスト変化
        alpha = np.random.uniform(0.8, 1.2)   # contrast
        beta  = np.random.uniform(-15, 15)     # brightness
        gray  = np.clip(gray.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return gray

    def __getitem__(self, idx: int):
        import torch
        import torchvision.transforms.functional as TF

        base_idx = idx % len(self.samples)
        fname, angle = self.samples[base_idx]
        gray = self._load_and_preprocess(self.img_dir / fname)

        if self.augment and idx >= len(self.samples):
            gray = self._augment(gray)

        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tensor   = TF.to_tensor(gray_rgb)
        tensor   = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor   = TF.resize(tensor, [224, 224])

        norm_angle = (angle - self.ANGLE_MIN) / (self.ANGLE_MAX - self.ANGLE_MIN)
        return tensor, torch.tensor(norm_angle, dtype=torch.float32), fname


# ── 訓練ループ ────────────────────────────────────────────────────────────────

def train(
    gt_csv: str,
    img_dir: str,
    base_model_path: str,
    out_dir: str,
    epochs: int = 100,
    lr: float = 3e-6,
    batch_size: int = 8,
    aug_factor: int = 20,
    unfreeze_stages: int = 2,
    patience: int = 20,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # モデルロード
    print(f"ベースモデル: {base_model_path}")
    model = build_model()
    ckpt = torch.load(base_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = freeze_backbone_partial(model, unfreeze_stages)
    model = model.to(device)

    # データ
    dataset = RealXrayDataset(gt_csv, img_dir, augment=True, aug_factor=aug_factor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    best_loss = float("inf")
    no_improve = 0

    print(f"\n訓練開始: epochs={epochs}, lr={lr}, batch={batch_size}")
    print(f"  データ: {len(dataset)}枚 ({len(dataset.samples)}実X線 × {aug_factor}aug)")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for tensors, labels, _ in loader:
            tensors = tensors.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            out  = model(tensors)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        epoch_loss = np.mean(losses)
        print(f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.5f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), out_path / "best_finetuned.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping (patience={patience})")
                break

    # 最終評価
    print(f"\n訓練完了。best_loss={best_loss:.5f}")
    print(f"保存: {out_path / 'best_finetuned.pth'}")

    # 評価: 実X線での推論
    model.load_state_dict(torch.load(out_path / "best_finetuned.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    eval_dataset = RealXrayDataset(gt_csv, img_dir, augment=False)
    print("\n評価（fine-tuned model）:")
    errors = []
    for tensors, labels, fnames in DataLoader(eval_dataset, batch_size=1):
        with torch.no_grad():
            out = model(tensors.to(device))
        pred_deg = float(out.item()) * 90 + 90
        gt_deg   = float(labels.item()) * 90 + 90
        err = abs(pred_deg - gt_deg)
        errors.append(err)
        print(f"  {fnames[0]:30} GT={gt_deg:.1f}° Pred={pred_deg:.1f}° Err={err:.1f}°")
    print(f"  MAE = {np.mean(errors):.2f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt 角度推定 Fine-tuning（実X線）")
    parser.add_argument("--real_xray_dir", default="data/real_xray/images/")
    parser.add_argument("--gt_csv",        default="data/real_xray/ground_truth.csv")
    parser.add_argument("--base_model",    default="runs/angle_estimator/best.pth")
    parser.add_argument("--out_dir",       default="runs/angle_estimator_finetuned")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=3e-6,
                        help="学習率（DRR訓練の1/10 = 3e-6 推奨）")
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--aug_factor",    type=int,   default=20,
                        help="データ拡張倍率（n_images × aug_factor = total）")
    parser.add_argument("--unfreeze_stages", type=int, default=2,
                        help="解凍するConvNeXtステージ数（1=最終のみ, 2=最終2ブロック）")
    parser.add_argument("--patience",      type=int,   default=20)
    args = parser.parse_args()

    train(
        gt_csv         = str(_PROJECT_ROOT / args.gt_csv),
        img_dir        = str(_PROJECT_ROOT / args.real_xray_dir),
        base_model_path= str(_PROJECT_ROOT / args.base_model),
        out_dir        = str(_PROJECT_ROOT / args.out_dir),
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        aug_factor     = args.aug_factor,
        unfreeze_stages= args.unfreeze_stages,
        patience       = args.patience,
    )


if __name__ == "__main__":
    main()
