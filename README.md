# ElbowVision

肘ファントムX線画像から肘関節角度をAIで推定するシステム。

## 概要

- **対象角度**: 外反角（Carrying angle）・屈曲角
- **モデル**: YOLOv8-pose（キーポイント検出）
- **特徴**: 実X線ファントム使用のためドメインギャップなし・倫理委員会不要

## 構成

```
ElbowVision/
├── elbow-api/       # FastAPI バックエンド
├── elbow-frontend/  # Next.js フロントエンド
├── elbow-train/     # YOLOv8 学習スクリプト
├── data/            # データセット
├── docs/            # ドキュメント
└── docker-compose.yml
```

## セットアップ

```bash
docker-compose up --build
```

- API: http://localhost:8000
- フロントエンド: http://localhost:3000

## 学習

```bash
cd elbow-train
python train.py
```

## 関連

- [OsteoVision](../OsteoVision_Dev/) — 膝関節角度推定（姉妹プロジェクト）
