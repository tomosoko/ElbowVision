"""ElbowVision ConvNeXt-Small — M4 Pro最適化ローカル訓練スクリプト

YOLO訓練完了後に実行する Phase 2 訓練。
64GB RAM + MPS GPU をフル活用。

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python elbow-train/train_convnext_m4pro.py

  # YOLO完了後に自動で続けて実行する場合:
  python elbow-train/train_m4pro.py && python elbow-train/train_convnext_m4pro.py
"""

import os
import sys
import time

# --- パス設定 -----------------------------------------------------------------
# このスクリプトは ElbowVision/ から実行される想定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "yolo_dataset_v6", "convnext_labels.csv")
IMGS_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset_v6", "images")
SAVE_PATH = os.path.join(BASE_DIR, "elbow-api", "elbow_convnext_best.pth")

# train_angle_predictor.py を import するためにパスを追加
TRAINING_DIR = os.path.join(BASE_DIR, "elbow-api", "training")
sys.path.insert(0, TRAINING_DIR)

# --- 訓練パラメータ（M4 Pro 64GB最適化・v4改善版）----------------------------
EPOCHS = 150          # v4: 大データセット対応
BATCH_SIZE = 64       # 64GB RAM活用
LR = 5e-5             # v4: より細かい収束
PATIENCE = 30         # v4: patience拡大
WARMUP = 5


def main():
    # 事前チェック
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSVファイルが見つかりません: {CSV_PATH}")
        print("  先に elbow_synth.py で DRR + ラベル生成を実行してください。")
        sys.exit(1)

    if not os.path.isdir(IMGS_DIR):
        print(f"ERROR: 画像ディレクトリが見つかりません: {IMGS_DIR}")
        sys.exit(1)

    import torch
    import pandas as pd

    # データセット確認
    df = pd.read_csv(CSV_PATH)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()

    print("=" * 60)
    print("  ElbowVision ConvNeXt-Small — M4 Pro最適化訓練")
    print(f"  MPS GPU: {torch.backends.mps.is_available()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CSV: {CSV_PATH}")
    print(f"  Images: {IMGS_DIR}")
    print(f"  Train: {n_train} / Val: {n_val}")
    print(f"  Batch: {BATCH_SIZE} / Epochs: {EPOCHS} / Patience: {PATIENCE}")
    print(f"  Save: {SAVE_PATH}")
    print("=" * 60)

    # train_angle_predictor.py の train() を argparse.Namespace で呼ぶ
    from argparse import Namespace
    from train_angle_predictor import train

    args = Namespace(
        csv=CSV_PATH,
        imgs=IMGS_DIR,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        warmup=WARMUP,
        num_workers=4,    # v4改善: DataLoader並列化
        amp=True,         # MPS では内部で自動無効化される
    )

    start = time.time()
    train(args)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  ConvNeXt訓練完了: {elapsed:.0f}秒 ({elapsed / 60:.1f}分)")
    print(f"  モデル保存先: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
