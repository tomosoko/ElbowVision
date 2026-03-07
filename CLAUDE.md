# ElbowVision - ファイルマップ

肘ファントムX線から関節角度をAIで推定するシステム（YOLOv8-pose + FastAPI + Next.js）
OsteoVision（膝）のインフラをそのまま流用して横展開。

**優位点：** 実X線ファントム使用 → ドメインギャップなし・倫理委員会不要

---

## どこに何があるか

### ルート直下
| ファイル | 何のファイルか |
|---|---|
| `bland_altman_analysis.py` | 精度検証スクリプト（OsteoVisionから流用） |
| `docker-compose.yml` | Docker一括起動設定 |
| `.gitignore` | Gitの除外設定 |

### docs/ — 手順書・ドキュメント
| ファイル | 何のファイルか |
|---|---|
| `00_全体フロー.md` | 開発の全体フロー概要 |
| `01_環境構築手順.md` | 開発環境のセットアップ手順 |
| `02_アノテーション手順.md` | LabelStudioでのキーポイント付け方 |
| `03_LabelStudio設定.md` | LabelStudioの設定方法 |
| `04_モデル訓練手順.md` | YOLOv8-poseの訓練手順 |
| `05_システム起動手順.md` | APIとフロントエンドの起動手順 |
| `06_精度検証手順.md` | Bland-Altman等の検証手順 |

### elbow-api/ — FastAPIバックエンド
| ファイル/フォルダ | 何のファイルか |
|---|---|
| `main.py` | APIのメイン（角度推定エンドポイント） |
| `requirements.txt` | Pythonパッケージ一覧 |
| `Dockerfile` | Docker設定 |
| `tests/test_api.py` | APIのテストコード |
| `tests/conftest.py` | テスト共通設定 |

### elbow-frontend/ — Next.jsフロントエンド
| ファイル/フォルダ | 何のファイルか |
|---|---|
| `src/app/page.tsx` | メインページ（画像アップロード・結果表示） |
| `src/app/layout.tsx` | レイアウト設定 |
| `src/app/globals.css` | グローバルスタイル |
| `package.json` | npmパッケージ一覧 |
| `next.config.ts` | Next.js設定 |
| `Dockerfile` | Docker設定 |

### elbow-api/training/ — ConvNeXtセカンドオピニオン訓練（OsteoVisionのtraining/に相当）
| ファイル | 何のファイルか |
|---|---|
| `convnext_model.py` | ElbowConvNeXt モデル定義（API・訓練スクリプトで共有） |
| `train_angle_predictor.py` | ConvNeXt訓練スクリプト（AP/LAT損失マスク付き） |

### elbow-train/ — YOLOv8訓練関連スクリプト
| ファイル | 何のファイルか |
|---|---|
| `dicom_to_png.py` | DICOMファイルをPNG画像に変換する |
| `batch_analyze.py` | 複数画像をまとめて解析する |
| `train_yolo_pose.py` | YOLOv8-poseを訓練するメインスクリプト |
| `dataset.yaml` | YOLOデータセット設定 |
| `annotation_guide.md` | アノテーション作業ガイド |
| `phantom_shooting_protocol.md` | ファントム撮影プロトコル |

### data/ — 訓練データ
| フォルダ | 何のフォルダか |
|---|---|
| `data/images/train/` | 訓練用画像（ここに撮影画像を入れる） |
| `data/images/val/` | 検証用画像 |
| `data/labels/train/` | 訓練用ラベル（アノテーション結果） |
| `data/labels/val/` | 検証用ラベル |

---

## 起動コマンド

```bash
# APIサーバー起動（ポート8000）
cd /Users/kohei/ElbowVision_Dev/elbow-api
uvicorn main:app --host 0.0.0.0 --port 8000

# フロントエンド起動（ポート3000）
cd /Users/kohei/ElbowVision_Dev/elbow-frontend
npm run dev

# Docker一括起動
cd /Users/kohei/ElbowVision_Dev
docker-compose up
```

---

## 測定対象角度
- **外反角（Carrying angle）**：AP像で計測、正常5〜15°
- **屈曲角**：側面像で計測
- **橈骨頭の傾き**：脱臼・骨折後評価（将来拡張）

## AI推論パイプライン
1. **YOLOv8-pose**（プライマリ）→ キーポイント検出 → 幾何学的角度計算
2. **Classical CV**（フォールバック）→ CLAHE + Otsu + 連結成分
3. **エッジバリデーション**（全モデル共通）→ Canny + HoughLinesP で骨軸を独立検出 → プライマリ角度との差分でconfidence判定
4. **ConvNeXt-Small**（セカンドオピニオン）→ 角度直接回帰 + Grad-CAM XAI

---

## 現在の状況（2026-03-07時点）

**完了済み**
- プロジェクト全体の骨格（API・フロントエンド・訓練スクリプト）
- 手順書（docs/）
- 訓練スクリプト群（elbow-train/）

**完了済み（追加）**
- ConvNeXt セカンドオピニオン + Grad-CAM XAI
- エッジバリデーション（Canny + Hough → 骨軸角度で自動検証）
- フロントエンド：セカンドオピニオン紫表示・エッジ検証バナー・Grad-CAMタブ

**次にやること**
1. 肘ファントムを職場でAP・側面撮影
2. DICOMをPNGに変換（dicom_to_png.py）
3. LabelStudioでキーポイントアノテーション（docs/02参照）
4. YOLOv8-pose訓練（train_yolo_pose.py）
5. ConvNeXt訓練（elbow-api/training/train_angle_predictor.py）
