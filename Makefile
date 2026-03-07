## ElbowVision — 実験用コマンド集
## 使い方: make <ターゲット名>

PYTHON     = python3
CT_IN      = data/raw_dicom/ct/CT_001
DRR_OUT    = data/ct_drr/CT_001
VAL_RESULT = validation_output/ai_results.csv
API_URL    = http://localhost:8000

# ─── セットアップ ──────────────────────────────────────────────────────────────

## 依存パッケージのインストール
setup:
	pip3 install -r elbow-api/requirements.txt

## data/ ディレクトリ構造を作成
init-dirs:
	mkdir -p data/raw_dicom/ct
	mkdir -p data/ct_drr
	mkdir -p data/images/train data/images/val
	mkdir -p data/labels/train data/labels/val
	mkdir -p validation_output
	@echo "ディレクトリ作成完了"

# ─── DRR生成（CT → 模擬X線） ──────────────────────────────────────────────────

## CTからDRR生成（data/raw_dicom/ct/CT_001/ にCT DICOMを入れてから実行）
drr:
	$(PYTHON) elbow-train/ct_reorient.py \
		--input   $(CT_IN) \
		--output  $(DRR_OUT)

## 向き確認のみ（DRR生成なし、ターミナルにpreviewを表示）
drr-preview:
	$(PYTHON) elbow-train/ct_reorient.py \
		--input   $(CT_IN) \
		--preview

## 回転バリエーション付きDRR生成（データ拡張）
drr-aug:
	$(PYTHON) elbow-train/ct_reorient.py \
		--input     $(CT_IN) \
		--output    $(DRR_OUT) \
		--rotations 0,15,30,45,-15,-30,-45

# ─── DRR + YOLOラベル自動生成（LabelStudio不要） ────────────────────────────

## 合成ファントムで動作確認（実CTなしでテスト）
factory-test:
	$(PYTHON) elbow-train/yolo_pose_factory.py --synthetic

## 実CTからDRR + YOLOラベルを自動生成
factory:
	$(PYTHON) elbow-train/yolo_pose_factory.py \
		--ct_dir $(CT_IN) \
		--out_dir data/yolo_dataset

# ─── サーバー起動 ─────────────────────────────────────────────────────────────

## APIサーバー起動（ポート8000）
api:
	cd elbow-api && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## フロントエンド起動（ポート3000）
frontend:
	cd elbow-frontend && npm run dev

# ─── 訓練 ────────────────────────────────────────────────────────────────────

## YOLOv8-poseを訓練（アノテーション完了後に実行）
train:
	$(PYTHON) elbow-train/train_yolo_pose.py

# ─── 解析・検証 ──────────────────────────────────────────────────────────────

## val画像を一括解析してCSV出力（make api を先に別ターミナルで起動すること）
analyze:
	$(PYTHON) elbow-train/batch_analyze.py \
		--input  data/images/val \
		--output $(VAL_RESULT) \
		--api    $(API_URL)

## Bland-Altman 精度検証（CSVに手動計測値を追記してから実行）
validate:
	$(PYTHON) bland_altman_analysis.py

# ─── APIヘルスチェック ────────────────────────────────────────────────────────

## APIが起動しているか確認
check:
	curl -s $(API_URL)/api/health | python3 -m json.tool

# ─── ヘルプ ──────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "=== ElbowVision 実験フロー ==="
	@echo ""
	@echo "  [1] make init-dirs   # フォルダ作成（初回のみ）"
	@echo "  [2] make drr-preview # CT向き確認"
	@echo "  [3] make drr         # CT → DRR生成（模擬X線）"
	@echo "  [4] make drr-aug     # 回転バリエーション付きDRR生成（データ拡張）"
	@echo "  [4] make factory-test # 合成ファントムで動作確認（実CTなし）"
	@echo "  [5] make factory     # CT → DRR + YOLOラベル自動生成（アノテーション不要）"
	@echo "  [6] make train       # YOLOv8-pose訓練"
	@echo "  [6] make api         # APIサーバー起動（別ターミナルで）"
	@echo "  [7] make analyze     # val画像を一括解析 → CSV出力"
	@echo "  [8] make validate    # Bland-Altman検証"
	@echo ""
	@echo "  make check           # APIの死活確認"
	@echo "  make setup           # pip依存パッケージインストール"
	@echo ""

.PHONY: setup init-dirs drr drr-preview drr-aug factory factory-test api frontend train analyze validate check help
