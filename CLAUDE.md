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

## 訓練結果記録

### ConvNeXt-Small v6（2026-04-13〜14）
```
データ:   data/yolo_dataset_v6/  Train 3400 / Val 600
設定:     batch=64, epochs=150(patience=30), MPS GPU
結果:     Best val error = 0.103 deg (ep125, 150ep完走, 437min)
モデル:   elbow-api/elbow_convnext_best.pth
```

### YOLOv8-Pose v6（2026-04-13）
```
mAP50 = 0.995  (elbow_v612, early stop ep93)
重み:   elbow-api/models/yolo_pose_best.pt  ← v6コピー済み
```

### 実X線推論結果（2026-04-14）
```
AP像 (3枚): rotation_err ≈ +22.9°, flexion ≈ 5.6° (AP不使用)
LAT像(3枚): rotation_err ≈ -9.9°,  flexion ≈ 79.7° ← 90°ボリューム撮影と一致
※ 推論コマンド:
   cd elbow-api
   venv/bin/python -c "sys.path.insert(0,'training'); ... ElbowConvNeXt ..."
   ※ main.py はモデルパスが elbow-api/ 相対なので elbow-api/ から実行必須
```

## テスト（278件全通過）
```bash
cd /Users/kohei/develop/research/ElbowVision
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pytest -q
```
| テストファイル | 件数 | 対象 |
|---|---|---|
| `tests/test_bland_altman.py` | 17 | Bland-Altman解析 |
| `tests/test_ct_reorient.py` | 47 | CT再方向付け + _axis_arrow + parse_coords |
| `tests/test_elbow_synth.py` | 24 | DRR生成・回転行列・transform_landmarks_canonical |
| `tests/test_angle_functions.py` | 13 | carrying/flexion angle計算 |
| `tests/test_make_yolo_label.py` | 15 | YOLOラベル生成・透視投影 |
| `tests/test_warmup_scheduler.py` | 8 | WarmupCosineScheduler |
| `tests/test_elbow_dataset.py` | 9 | ElbowDataset._resolve_img_path + __getitem__ |
| `tests/test_inference.py` | 18 | 推論パイプライン・CSV出力 |
| `tests/test_dicom_to_png.py` | 17 | DICOM→PNG変換 |
| `tests/test_eval_realxray_batch.py` | 16 | Phase 2 Bland-Altman (ICC・MAE・LoA・PASS/FAIL) |
| `elbow-api/tests/test_api.py` | 65 | FastAPI エンドポイント（view_type・ConvNeXt second_opinion含む） |

## 技術ノート

- 骨閾値: 2段階Otsu（ファントム骨素材対応）
- 回転中心: 滑車/小頭中心（上顆中点より前方）
- キーポイント6点: humerus_shaft, lateral/medial_epicondyle, forearm_shaft, radial_head, olecranon
- HUウィンドウ: ファントム最適 hu_min=50, hu_max=800
