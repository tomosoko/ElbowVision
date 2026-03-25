#!/bin/bash
# =============================================================
# ElbowVision フルパイプライン実行スクリプト
# Mac Mini M4 Pro 64GB 用
# =============================================================
# 1. configs/dataset_config.yaml からDRR生成
# 2. データセット統計の表示・可視化
# 3. YOLOv8-pose訓練
# 4. ConvNeXt訓練
# 5. Bland-Altman検証
# 6. 結果サマリー出力
#
# 使い方:
#   cd ~/Dev/vision/ElbowVision
#   bash scripts/generate_dataset.sh [--step N] [--config PATH]
#
# オプション:
#   --step N     : 指定ステップから開始 (1-6)
#   --config PATH: YAML設定ファイル (デフォルト: configs/dataset_config.yaml)
#   --dry-run    : コマンドを表示するだけで実行しない
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/elbow-api/venv"
CONFIG_FILE="$PROJECT_DIR/configs/dataset_config.yaml"
TRAIN_CONFIG="$PROJECT_DIR/configs/training_config.yaml"
START_STEP=1
DRY_RUN=false

# --- 引数パース ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)   START_STEP="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "不明な引数: $1"; exit 1 ;;
    esac
done

# --- YAML読み取りヘルパー（Python使用）---
read_yaml() {
    local key="$1"
    local file="$2"
    python3 -c "
import yaml
with open('$file') as f:
    cfg = yaml.safe_load(f)
val = cfg.get('$key', '')
print(val if val is not None else '')
"
}

# --- 環境チェック ---
echo "=========================================="
echo "  ElbowVision フルパイプライン"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# venv確認
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: 仮想環境が見つかりません: $VENV_DIR"
    echo "  先に scripts/setup_mac_mini.sh を実行してください"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python3 --version)"
echo "venv: $VENV_DIR"
echo "Config: $CONFIG_FILE"

# PyYAML確認
python3 -c "import yaml" 2>/dev/null || {
    echo "PyYAML をインストール中..."
    pip install --quiet pyyaml
}

# 設定読み込み
DATASET_DIR=$(read_yaml "out_dir" "$CONFIG_FILE")
# 相対パスなら PROJECT_DIR を基準に
if [[ ! "$DATASET_DIR" = /* ]]; then
    DATASET_DIR="$PROJECT_DIR/$DATASET_DIR"
fi
echo "Dataset: $DATASET_DIR"
echo ""

run_cmd() {
    echo "  >> $*"
    if [ "$DRY_RUN" = false ]; then
        eval "$@"
    fi
}

STEP_START_TIME=$(date +%s)

# =============================================================
# Step 1: DRR生成
# =============================================================
if [ "$START_STEP" -le 1 ]; then
    echo "=========================================="
    echo "  [Step 1/6] DRR生成"
    echo "=========================================="
    STEP_START=$(date +%s)

    run_cmd python3 "$PROJECT_DIR/elbow-train/elbow_synth.py" \
        --config "$CONFIG_FILE"

    STEP_END=$(date +%s)
    echo "  所要時間: $((STEP_END - STEP_START))秒"
    echo ""
fi

# =============================================================
# Step 2: データセット統計
# =============================================================
if [ "$START_STEP" -le 2 ]; then
    echo "=========================================="
    echo "  [Step 2/6] データセット統計"
    echo "=========================================="

    run_cmd python3 "$PROJECT_DIR/scripts/dataset_stats.py" \
        --dataset_dir "$DATASET_DIR"

    echo ""
fi

# =============================================================
# Step 3: YOLOv8-pose訓練
# =============================================================
if [ "$START_STEP" -le 3 ]; then
    echo "=========================================="
    echo "  [Step 3/6] YOLOv8-pose訓練"
    echo "=========================================="
    STEP_START=$(date +%s)

    YOLO_YAML="$DATASET_DIR/dataset.yaml"
    if [ ! -f "$YOLO_YAML" ]; then
        echo "ERROR: dataset.yaml が見つかりません: $YOLO_YAML"
        echo "  Step 1 (DRR生成) を先に実行してください"
        exit 1
    fi

    # training_config.yaml からYOLOパラメータを読み取り
    YOLO_EPOCHS=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('yolo', {}).get('epochs', 200))
")
    YOLO_IMGSZ=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('yolo', {}).get('imgsz', 512))
")
    YOLO_BATCH=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('yolo', {}).get('batch', 16))
")
    YOLO_PATIENCE=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('yolo', {}).get('patience', 30))
")
    YOLO_NAME=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('yolo', {}).get('name', 'elbowvision_pose_v2'))
")

    run_cmd python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.train(
    data='$YOLO_YAML',
    epochs=$YOLO_EPOCHS,
    imgsz=$YOLO_IMGSZ,
    batch=$YOLO_BATCH,
    device='mps',
    name='$YOLO_NAME',
    pose=1.5,
    patience=$YOLO_PATIENCE,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
)
print('YOLOv8-pose 訓練完了')
"

    STEP_END=$(date +%s)
    echo "  所要時間: $((STEP_END - STEP_START))秒"
    echo ""
fi

# =============================================================
# Step 4: ConvNeXt訓練
# =============================================================
if [ "$START_STEP" -le 4 ]; then
    echo "=========================================="
    echo "  [Step 4/6] ConvNeXt訓練"
    echo "=========================================="
    STEP_START=$(date +%s)

    CONVNEXT_CSV="$DATASET_DIR/convnext_labels.csv"
    CONVNEXT_IMGS="$DATASET_DIR/images/"

    if [ ! -f "$CONVNEXT_CSV" ]; then
        echo "ERROR: convnext_labels.csv が見つかりません: $CONVNEXT_CSV"
        exit 1
    fi

    # training_config.yaml からConvNeXtパラメータを読み取り
    CONVNEXT_EPOCHS=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('convnext', {}).get('epochs', 100))
")
    CONVNEXT_BATCH=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('convnext', {}).get('batch_size', 32))
")
    CONVNEXT_LR=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('convnext', {}).get('lr', 0.0003))
")
    CONVNEXT_WARMUP=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('convnext', {}).get('warmup_epochs', 5))
")
    CONVNEXT_PATIENCE=$(python3 -c "
import yaml
with open('$TRAIN_CONFIG') as f: cfg = yaml.safe_load(f)
print(cfg.get('convnext', {}).get('patience', 20))
")

    run_cmd python3 "$PROJECT_DIR/elbow-api/training/train_angle_predictor.py" \
        --csv "$CONVNEXT_CSV" \
        --imgs "$CONVNEXT_IMGS" \
        --epochs "$CONVNEXT_EPOCHS" \
        --batch "$CONVNEXT_BATCH" \
        --lr "$CONVNEXT_LR" \
        --warmup "$CONVNEXT_WARMUP" \
        --patience "$CONVNEXT_PATIENCE"

    STEP_END=$(date +%s)
    echo "  所要時間: $((STEP_END - STEP_START))秒"
    echo ""
fi

# =============================================================
# Step 5: Bland-Altman検証
# =============================================================
if [ "$START_STEP" -le 5 ]; then
    echo "=========================================="
    echo "  [Step 5/6] Bland-Altman検証"
    echo "=========================================="

    BA_SCRIPT="$PROJECT_DIR/scripts/bland_altman.py"
    if [ -f "$BA_SCRIPT" ]; then
        run_cmd python3 "$BA_SCRIPT" \
            --dataset_dir "$DATASET_DIR" \
            --yolo_model "$PROJECT_DIR/runs/pose/$YOLO_NAME/weights/best.pt" \
            --convnext_model "$PROJECT_DIR/elbow-api/elbow_convnext_best.pth"
    else
        echo "  SKIP: bland_altman.py が見つかりません"
        echo "  別途 Bland-Altman検証を実施してください"
    fi
    echo ""
fi

# =============================================================
# Step 6: 結果サマリー
# =============================================================
if [ "$START_STEP" -le 6 ]; then
    echo "=========================================="
    echo "  [Step 6/6] 結果サマリー"
    echo "=========================================="

    TOTAL_END=$(date +%s)
    TOTAL_TIME=$((TOTAL_END - STEP_START_TIME))

    echo ""
    echo "  総所要時間: ${TOTAL_TIME}秒 ($((TOTAL_TIME / 60))分)"
    echo ""
    echo "  --- 生成ファイル ---"

    # データセット
    if [ -d "$DATASET_DIR" ]; then
        TRAIN_COUNT=$(ls "$DATASET_DIR"/images/train/*.png 2>/dev/null | wc -l | tr -d ' ')
        VAL_COUNT=$(ls "$DATASET_DIR"/images/val/*.png 2>/dev/null | wc -l | tr -d ' ')
        echo "  データセット: $DATASET_DIR"
        echo "    train: ${TRAIN_COUNT}枚 / val: ${VAL_COUNT}枚"
    fi

    # YOLOモデル
    YOLO_BEST="$PROJECT_DIR/runs/pose/${YOLO_NAME:-elbowvision_pose_v2}/weights/best.pt"
    if [ -f "$YOLO_BEST" ]; then
        echo "  YOLOモデル: $YOLO_BEST"
    fi

    # ConvNeXtモデル
    CONVNEXT_BEST="$PROJECT_DIR/elbow-api/elbow_convnext_best.pth"
    if [ -f "$CONVNEXT_BEST" ]; then
        echo "  ConvNeXtモデル: $CONVNEXT_BEST"
    fi

    # 統計プロット
    STATS_DIR="$PROJECT_DIR/results/dataset_stats"
    if [ -d "$STATS_DIR" ]; then
        echo "  統計プロット: $STATS_DIR/"
    fi

    echo ""
    echo "=========================================="
    echo "  パイプライン完了"
    echo "=========================================="
fi
