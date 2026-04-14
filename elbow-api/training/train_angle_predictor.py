"""
ElbowVision ConvNeXt-Small 訓練スクリプト

OsteoVision の train_angle_predictor.py を肘用に移植。

【使い方】
  # 訓練（アノテーション後）
  cd ~/develop/research/ElbowVision/elbow-api
  source venv/bin/activate
  python training/train_angle_predictor.py \
    --csv  ../data/yolo_dataset_v2/convnext_labels.csv \
    --imgs ../data/yolo_dataset_v2/images/

  # 保存先: elbow-api/elbow_convnext_best.pth（API が自動で読み込む）

【CSV形式（convnext_labels.csv — elbow_synth.py が自動生成）】
  filename, split, view_type, rotation_error_deg, flexion_deg, carrying_angle
  elbow_00000.png, val, AP, -18.38, 170.27, 0.0
  elbow_00050.png, train, LAT, 5.12, 87.3, 0.0

  ※ dataset_summary.csv も使用可能（必要な列が含まれていれば OK）

【画像パスの解決】
  --imgs は images/ ディレクトリを指定する。
  CSVの split 列（train/val）を使い、images/{split}/{filename} を自動解決する。

【出力（convnext_model.py の OUTPUT_DIM=2 と一致）】
  index 0 -- rotation_error_deg: 理想位からの前腕回旋ズレ（deg）
  index 1 -- flexion_deg       : 肘屈曲角（deg）

【訓練戦略】
  - ImageNet事前学習 ConvNeXt-Small をファインチューニング
  - AP/LAT で有効な出力チャンネルを損失マスクで制御
    AP  -> rotation_error_deg のみL1損失を計算（flexion_deg はマスク=0）
    LAT -> flexion_deg のみL1損失を計算（rotation_error_deg はマスク=0）
  - 学習率ウォームアップ（最初の5エポック、線形）
  - Mixed precision training (torch.amp) 対応
  - TensorBoard ログ出力
  - Early stopping (patience=20)
  - 最良モデルを elbow_convnext_best.pth に保存
  - 訓練完了後にloss曲線をPNG保存
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 同ディレクトリの convnext_model を参照
sys.path.insert(0, os.path.dirname(__file__))
from convnext_model import ElbowConvNeXt


# --- Dataset -----------------------------------------------------------------


class ElbowDataset(Dataset):
    """
    CSV + 画像ディレクトリからデータを読み込む Dataset。
    AP/LAT の view_type を取得し、損失マスクに使う。

    画像パスは images/{split}/{filename} の構造を想定。
    split列がない場合やフラットな構造の場合はフォールバックする。
    """

    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_img_path(self, row) -> str:
        """画像パスを解決する。split付きサブディレクトリを優先的に探す。"""
        filename = str(row["filename"])
        split = str(row.get("split", "train"))

        # 1. images/{split}/{filename} （elbow_synth.py の出力構造）
        path = os.path.join(self.img_dir, split, filename)
        if os.path.exists(path):
            return path

        # 2. images/{filename} （フラット構造）
        path = os.path.join(self.img_dir, filename)
        if os.path.exists(path):
            return path

        # 3. そのまま返す（FileNotFoundError は DataLoader で検出）
        return os.path.join(self.img_dir, split, filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_img_path(row)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ポジショニングラベル: [rotation_error_deg, flexion_deg]
        #   rotation_error_deg : 理想位からの回旋ズレ量（AP像のみ有効）
        #   flexion_deg        : 肘屈曲角（LAT像のみ有効）
        view = str(row.get("view_type", "AP")).upper()

        angles = torch.tensor(
            [
                float(row.get("rotation_error_deg", 0.0)),
                float(row.get("flexion_deg", 90.0)),
            ],
            dtype=torch.float32,
        )

        # 損失マスク: AP/LAT で有効なチャンネルだけ1
        mask = torch.tensor(
            [
                1.0 if view == "AP" else 0.0,  # rotation_error_deg: AP像のみ
                1.0 if view == "LAT" else 0.0,  # flexion_deg: LAT像のみ
            ],
            dtype=torch.float32,
        )

        return image, angles, mask


# --- 学習率スケジューラ（ウォームアップ付きCosine） ---------------------------


class WarmupCosineScheduler:
    """最初の warmup_epochs で線形ウォームアップし、以降は Cosine Annealing。"""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1)
        )
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            # 線形ウォームアップ: lr = base_lr * (step / warmup_epochs)
            scale = self._step_count / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# --- 訓練ループ --------------------------------------------------------------


def train(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Mixed precision: CUDA では float16、MPS/CPU では bfloat16（MPS対応）
    use_amp = args.amp and (device.type in ("cuda", "mps"))
    if device.type == "cuda":
        amp_dtype = torch.float16
    elif device.type == "mps":
        amp_dtype = torch.float32  # MPS は autocast 非対応のため実質無効
        use_amp = False
    else:
        amp_dtype = torch.float32
        use_amp = False

    if use_amp:
        print(f"Mixed precision: ON (dtype={amp_dtype})")
    else:
        print("Mixed precision: OFF")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),        # ±10° 回転 (v4改善)
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # v4改善
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    df = pd.read_csv(args.csv)

    # CSV列の存在チェック
    required_cols = {"filename", "view_type", "rotation_error_deg", "flexion_deg"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSVに必須列が不足: {missing}")
        print(f"  CSVの列: {list(df.columns)}")
        print(
            "  elbow_synth.py が生成する convnext_labels.csv を使用してください。"
        )
        sys.exit(1)

    val_mask = df.get("split", pd.Series(["train"] * len(df))) == "val"
    df_train = df[~val_mask]
    df_val = df[val_mask]
    print(f"train: {len(df_train)}枚 / val: {len(df_val)}枚")

    # AP/LAT分布を表示
    for split_name, split_df in [("train", df_train), ("val", df_val)]:
        views = split_df.get("view_type", pd.Series())
        if len(views) > 0:
            ap_count = (views.str.upper() == "AP").sum()
            lat_count = (views.str.upper() == "LAT").sum()
            print(f"  {split_name}: AP={ap_count}, LAT={lat_count}")

    # プロセス固有のtempファイルを使用（並行実行時の競合を防ぐ）
    _pid = os.getpid()
    _tmp_train = os.path.join(tempfile.gettempdir(), f"_elbow_train_{_pid}.csv")
    _tmp_val   = os.path.join(tempfile.gettempdir(), f"_elbow_val_{_pid}.csv")
    df_train.to_csv(_tmp_train, index=False)
    df_val.to_csv(_tmp_val, index=False)

    train_ds = ElbowDataset(_tmp_train, args.imgs, transform)
    val_ds = ElbowDataset(_tmp_val, args.imgs, val_transform)

    num_workers = getattr(args, 'num_workers', 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers
    )

    model = ElbowConvNeXt(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs
    )

    # GradScaler は CUDA のみ
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # TensorBoard
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "runs",
        f"convnext_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard: {log_dir}")
        print(f"  tensorboard --logdir {os.path.dirname(log_dir)}")
    except ImportError:
        writer = None
        print("TensorBoard: tensorboard未インストール（pip install tensorboard）")

    best_val_loss = float("inf")
    patience_count = 0
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "elbow_convnext_best.pth"
    )

    # 損失関数（ループ外で1度だけ生成）
    huber = nn.HuberLoss(delta=10.0, reduction='none')

    # 記録用
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(1, args.epochs + 1):
        # -- 訓練 --
        model.train()
        train_loss = 0.0
        for images, targets, masks in train_loader:
            images, targets, masks = (
                images.to(device),
                targets.to(device),
                masks.to(device),
            )
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    preds = model(images)
                    loss = (huber(preds, targets) * masks).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss = (huber(preds, targets) * masks).mean()
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # -- 検証 --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, masks in val_loader:
                images, targets, masks = (
                    images.to(device),
                    targets.to(device),
                    masks.to(device),
                )
                preds = model(images)
                val_loss += (huber(preds, targets) * masks).mean().item()
        val_loss /= max(len(val_loader), 1)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 記録
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)

        # TensorBoard
        if writer:
            writer.add_scalars(
                "Loss", {"train": train_loss, "val": val_loss}, epoch
            )
            writer.add_scalar("LearningRate", current_lr, epoch)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.3f} deg  val={val_loss:.3f} deg  "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(
                f"  -> Best model saved ({best_val_loss:.3f} deg) -> {save_path}"
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if writer:
        writer.close()

    # 一時CSVを削除
    for tmp in (_tmp_train, _tmp_val):
        try:
            os.remove(tmp)
        except OSError:
            pass

    # -- Loss曲線をPNG保存 --
    plot_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "training_loss_curve.png",
    )
    _save_loss_plot(train_losses, val_losses, learning_rates, plot_path)

    print(f"\n訓練完了。最良val loss: {best_val_loss:.3f} deg")
    print(f"保存先: {save_path}")
    print(f"Loss曲線: {plot_path}")
    if writer:
        print(f"TensorBoard: tensorboard --logdir {os.path.dirname(log_dir)}")


def _save_loss_plot(
    train_losses: list,
    val_losses: list,
    learning_rates: list,
    save_path: str,
):
    """訓練・検証のloss曲線と学習率をPNGに保存。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    epochs = range(1, len(train_losses) + 1)

    # Loss 曲線
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
    ax1.scatter([best_epoch], [best_val], color="green", s=80, zorder=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MAE, deg)")
    ax1.set_title(f"ElbowVision ConvNeXt Training  |  Best val={best_val:.3f} deg @ epoch {best_epoch}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 学習率
    ax2.plot(epochs, learning_rates, "g-", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss曲線を保存: {save_path}")


# --- エントリポイント ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ElbowVision ConvNeXt訓練")
    parser.add_argument("--csv", required=True, help="データセットCSVパス（convnext_labels.csv推奨）")
    parser.add_argument("--imgs", required=True, help="画像ディレクトリ（images/を指定）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--warmup", type=int, default=5, help="ウォームアップエポック数")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Mixed precision training (CUDA時のみ有効)",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Mixed precision無効化")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
