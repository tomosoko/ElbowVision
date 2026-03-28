@~/.claude/CLAUDE.md

# ElbowVision

肘X線ポジショニングAIガイドシステム（YOLOv8-pose + FastAPI + Next.js）

## 現在のフェーズ: プレ研究（ファントム実証）

**目的**: ファントム（骨等価樹脂模型）を使い「AIが肘X線のキーポイントを検出できる」ことを実証する。
精度の極限追求ではなく**動作実証**がゴール。手順はdocs/07_プレ研究手順.mdに従う。
CT → DRR自動生成 → 訓練 → ポジショニング補正アドバイス

## 主要ファイル

| パス | 役割 |
|---|---|
| `elbow-api/main.py` | FastAPI（角度推定・ポジショニング補正） |
| `elbow-api/training/train_angle_predictor.py` | ConvNeXt訓練 |
| `elbow-frontend/src/app/page.tsx` | Next.jsメインUI |
| `elbow-train/elbow_synth.py` | CT→DRR自動生成★ |
| `elbow-train/train_yolo_pose.py` | YOLOv8-pose訓練 |
| `configs/dataset_config.yaml` | DRR生成パラメータ（マルチシリーズ対応） |
| `docs/` | 各種手順書（00〜06） |

## 起動

```bash
# API（ポート8000）
cd elbow-api && source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# フロントエンド（ポート3000）
cd elbow-frontend && npm run dev
```

## DRR生成

```bash
# 単一シリーズ
python elbow-train/elbow_synth.py \
  --ct_dir data/raw_dicom/ct/ \
  --out_dir data/yolo_dataset/ \
  --laterality R

# マルチシリーズ（3ボリューム: 180°/135°/90°）
python elbow-train/elbow_synth.py \
  --ct_dir "data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000" \
  --out_dir data/yolo_dataset/ \
  --laterality R \
  --series_nums "4,8,12" \
  --base_flexions "180,135,90" \
  --hu_min 50 --hu_max 800 \
  --domain_aug
```

## CTデータ

- `data/raw_dicom/ct/`: 単一シリーズ（180スライス）
- `data/raw_dicom/ct_volume/`: 3ボリューム（FC85骨カーネル Series 4/8/12）
  - Series 4: 301sl, 伸展180°（AP像用）
  - Series 8: 312sl, 135°
  - Series 12: 331sl, 屈曲90°（LAT像用）

## 技術ノート

- 骨閾値: 2段階Otsu（ファントム骨素材対応）
- 回転中心: 滑車/小頭中心（上顆中点より前方）
- キーポイント6点: humerus_shaft, lateral/medial_epicondyle, forearm_shaft, radial_head, olecranon
- HUウィンドウ: ファントム最適 hu_min=50, hu_max=800
