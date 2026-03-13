@~/.claude/CLAUDE.md

# ElbowVision

肘X線ポジショニングAIガイドシステム（YOLOv8-pose + FastAPI + Next.js）
CT → DRR自動生成 → 訓練 → ポジショニング補正アドバイス

## 主要ファイル

| パス | 役割 |
|---|---|
| `elbow-api/main.py` | FastAPI（角度推定・ポジショニング補正） |
| `elbow-api/training/train_angle_predictor.py` | ConvNeXt訓練 |
| `elbow-frontend/src/app/page.tsx` | Next.jsメインUI |
| `elbow-train/elbow_synth.py` | CT→DRR自動生成★ |
| `elbow-train/train_yolo_pose.py` | YOLOv8-pose訓練 |
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
python elbow-train/elbow_synth.py \
  --ct_dir data/raw_dicom/ct/ \
  --out_dir data/yolo_dataset/ \
  --laterality R   # R=右腕 / L=左腕（省略時はDICOMタグ自動検出）
```

## 次のステップ

1. ファントムCT撮影 → `data/raw_dicom/ct/`
2. `elbow_synth.py` でDRR生成
3. 実X線撮影 → `dicom_to_png.py` でPNG変換
4. YOLOv8-pose訓練 → ConvNeXt訓練
5. Bland-Altman検証（docs/06）
