#!/usr/bin/env bash
# train_pipeline.sh — YOLO + ConvNeXt 連続訓練パイプライン
#
# 使い方:
#   cd ~/develop/Dev/vision/ElbowVision
#   source elbow-api/venv/bin/activate
#   bash scripts/train_pipeline.sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

echo "============================================================"
echo "  ElbowVision Training Pipeline (v3 dataset)"
echo "  Base: $BASE_DIR"
echo "============================================================"

# --- Phase 1: YOLO訓練 ---
YOLO_MARKER="runs/elbow_m4pro_s/weights/best.pt"

if [ -f "$YOLO_MARKER" ]; then
    echo ""
    echo "[Phase 1] YOLO訓練済み: $YOLO_MARKER が存在します。スキップ。"
else
    echo ""
    echo "[Phase 1] YOLO訓練を開始..."
    python elbow-train/train_m4pro.py

    if [ ! -f "$YOLO_MARKER" ]; then
        echo "ERROR: YOLO訓練が完了しましたが best.pt が見つかりません。"
        echo "  runs/ ディレクトリを確認してください。"
        exit 1
    fi
    echo "[Phase 1] YOLO訓練完了: $YOLO_MARKER"
fi

# --- Phase 2: ConvNeXt訓練 ---
CONVNEXT_CSV="$BASE_DIR/data/yolo_dataset_v3/convnext_labels.csv"

if [ ! -f "$CONVNEXT_CSV" ]; then
    echo "ERROR: ConvNeXtラベルCSVが見つかりません: $CONVNEXT_CSV"
    echo "  先に elbow_synth.py でDRR + ラベル生成を実行してください。"
    exit 1
fi

echo ""
echo "[Phase 2] ConvNeXt訓練を開始..."
python elbow-train/train_convnext_m4pro.py

echo ""
echo "============================================================"
echo "  パイプライン完了"
echo "============================================================"
