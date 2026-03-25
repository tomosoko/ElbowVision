#!/bin/bash
# ============================================================
# ElbowVision - Mac Mini M4 Pro セットアップ & DRR本番生成
# ============================================================
# Mac Mini到着後にこのスクリプトを実行すると：
#   1. Python仮想環境の構築
#   2. 依存パッケージのインストール
#   3. 高解像度DRR生成（target_size=512）
#   4. SSDへの保存
#
# 使い方:
#   cd ~/Dev/vision/ElbowVision
#   bash scripts/setup_mac_mini.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="$HOME/Dev/vision/ElbowVision"
VENV_DIR="$PROJECT_DIR/elbow-api/venv"
CT_DIR="$PROJECT_DIR/data/raw_dicom/ct/"

# SSD出力先（接続されていなければローカルにフォールバック）
SSD_PATH="/Volumes/SSD-PGU3/Research_data/ElbowVision"
LOCAL_PATH="$PROJECT_DIR/data/drr_output"

echo "=========================================="
echo "  ElbowVision セットアップ (Mac Mini M4 Pro)"
echo "=========================================="

# ── 1. Python仮想環境 ──
echo ""
echo "[1/4] Python仮想環境の確認・構築..."
if [ ! -d "$VENV_DIR" ]; then
    echo "  仮想環境が見つかりません。作成します..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Python: $(python3 --version)"
echo "  venv: $VENV_DIR"

# ── 2. 依存パッケージ ──
echo ""
echo "[2/4] 依存パッケージのインストール..."
pip install --quiet --upgrade pip
pip install --quiet \
    numpy scipy pydicom opencv-python-headless \
    scikit-image pillow matplotlib \
    ultralytics torch torchvision \
    pingouin statsmodels scikit-learn \
    pytorch-grad-cam albumentations

echo "  インストール完了"

# ── 3. CTデータ確認 ──
echo ""
echo "[3/4] CTデータの確認..."
if [ -d "$CT_DIR" ]; then
    CT_COUNT=$(ls "$CT_DIR"/*.dcm 2>/dev/null | wc -l | tr -d ' ')
    echo "  CT DICOMファイル: ${CT_COUNT}枚"
else
    echo "  警告: CTデータが見つかりません: $CT_DIR"
    echo "  旧マシンからコピーしてください"
    exit 1
fi

# ── 4. DRR本番生成 ──
echo ""
echo "[4/4] 高解像度DRR生成..."

# SSD確認
if [ -d "/Volumes/SSD-PGU3" ]; then
    OUT_DIR="$SSD_PATH/drr_dataset_512"
    echo "  SSD検出: $SSD_PATH"
else
    OUT_DIR="$LOCAL_PATH/drr_dataset_512"
    echo "  SSD未接続 → ローカルに出力: $OUT_DIR"
fi

mkdir -p "$OUT_DIR"

echo ""
echo "  生成パラメータ:"
echo "    target_size: 512"
echo "    AP: 240枚 / LAT: 720枚"
echo "    domain_aug: 有効"
echo "    出力先: $OUT_DIR"
echo ""
echo "  開始..."

cd "$PROJECT_DIR"
python elbow-train/elbow_synth.py \
    --ct_dir "$CT_DIR" \
    --out_dir "$OUT_DIR" \
    --laterality L \
    --target_size 512 \
    --domain_aug \
    --n_ap 240 \
    --n_lat 720

echo ""
echo "=========================================="
echo "  完了"
echo "  出力: $OUT_DIR"
echo "  画像数: $(ls "$OUT_DIR"/images/train/*.png 2>/dev/null | wc -l | tr -d ' ') (train) + $(ls "$OUT_DIR"/images/val/*.png 2>/dev/null | wc -l | tr -d ' ') (val)"
echo "=========================================="
