# OSS導入推奨（参考）

調査日: 2026-03-25

## 即効性大

| 導入OSS | 対象箇所 | 効果 | 工数 |
|---|---|---|---|
| TorchDRR (`torch-drr`) | elbow_synth.py | DRR生成22倍高速化 | 1-2日 |
| MONAI (`monai[all]`) | elbow_synth.py / ct_reorient.py | DICOM処理統一化、ct_reorient.py不要化 | 2-3日 |
| pytorch-grad-cam | ConvNeXt Grad-CAM実装 | 手作り実装→標準ライブラリ化 | 3時間 |
| pingouin | bland_altman_analysis.py | 統計計算70行→5行 | 1時間 |

## 中期的

| 導入OSS | 対象箇所 | 効果 | 工数 |
|---|---|---|---|
| Detectron2 | YOLOv8の代替/補完 | 複数肘対応、多視点推定 | 1-2日 |
| timm | ConvNeXtバックボーン | 事前学習モデル充実 | 半日 |

## 変更不要
- ultralytics YOLOv8-pose / ConvNeXt: 現行で十分
- scipy.spatial.transform.Rotation: 最適
- FastAPI: 最適

## 備考
- OsteoVisionと同時にTorchDRR/MONAI導入推奨（共通コード）
