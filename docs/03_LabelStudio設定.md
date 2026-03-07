# 03 LabelStudio 設定手順

## インストール

```bash
cd /Users/kohei/ElbowVision_Dev
python3 -m venv venv_labelstudio
source venv_labelstudio/bin/activate
pip install label-studio
```

---

## 起動

```bash
source /Users/kohei/ElbowVision_Dev/venv_labelstudio/bin/activate
label-studio start
```

→ ブラウザで `http://localhost:8080` が自動的に開く

---

## 初回セットアップ

1. メールアドレスとパスワードを入力してアカウント作成
   （ローカルのみ使用するので何でもよい）
2. 「Create Project」

---

## プロジェクト設定

### General タブ
- Project Name: `ElbowVision_Annotation`

### Data Import タブ
画像をインポートする方法（どちらでも可）:

**方法A: ドラッグ&ドロップ**
- `data/images/train/` と `data/images/val/` の画像を全選択してドロップ

**方法B: フォルダ指定（推奨）**
1. 「Add Storage」→「Local files」
2. Absolute local path: `/Users/kohei/ElbowVision_Dev/data/images`
3. 「Add Storage」→「Sync Storage」

### Labeling Setup タブ

「Custom template」→「Code」タブに以下を貼り付け:

```xml
<View>
  <Header value="肘関節 キーポイントアノテーション"/>
  <Image name="image" value="$image"
         zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <KeyPointLabels name="kp" toName="image" opacity="0.9" strokeWidth="3">
    <Label value="humerus_shaft"      background="#3b82f6"
           hint="上腕骨幹部（近位端中央）"/>
    <Label value="lateral_epicondyle" background="#a855f7"
           hint="外側上顆（最突出点）"/>
    <Label value="medial_epicondyle"  background="#ec4899"
           hint="内側上顆（最突出点）"/>
    <Label value="forearm_shaft"      background="#22c55e"
           hint="前腕骨幹部（遠位端中央）"/>
  </KeyPointLabels>
</View>
```

「Save」をクリック。

---

## アノテーション作業の手順

### 1枚あたりの流れ（目安: 2〜3分/枚）

```
1. 画像を開く
2. マウスホイールでズームイン（骨端部が大きく見える程度）
3. 左パネルから「humerus_shaft」を選択
4. 上腕骨幹部の近位端にクリック → 点が打たれる
5. 「lateral_epicondyle」を選択 → 外側上顆にクリック
6. 「medial_epicondyle」を選択 → 内側上顆にクリック
7. 「forearm_shaft」を選択 → 前腕骨幹部の遠位端にクリック
8. 「Submit」をクリック（必須！これで保存される）
```

### ショートカットキー（作業効率化）

| キー | 操作 |
|-----|------|
| `W` | 次の画像へ |
| `S` | Submit（保存） |
| `1〜4` | ラベル1〜4を選択 |
| `Ctrl + Z` | 直前の操作を取り消し |
| `スペース` | パン（画像を動かす）モード切替 |
| `Ctrl + ホイール` | ズームイン/アウト |

---

## エクスポート手順

全画像のアノテーションが完了したら:

1. プロジェクト一覧 → `ElbowVision_Annotation`
2. 右上「Export」ボタン
3. 形式: **YOLO with Keypoints** を選択

   > 「YOLO」だけの選択肢はバウンディングボックスのみ。
   > **「YOLO with Keypoints」** を必ず選ぶこと。

4. 「Export」→ ZIPファイルがダウンロードされる

---

## エクスポート後の配置

```bash
# ダウンロードしたZIPを解凍
cd ~/Downloads
unzip labelstudio_export_*.zip -d elbow_labels/

# ラベルファイルをプロジェクトに配置
cp elbow_labels/labels/train/*.txt /Users/kohei/ElbowVision_Dev/data/labels/train/
cp elbow_labels/labels/val/*.txt   /Users/kohei/ElbowVision_Dev/data/labels/val/

# 枚数確認
ls /Users/kohei/ElbowVision_Dev/data/labels/train/ | wc -l
ls /Users/kohei/ElbowVision_Dev/data/labels/val/   | wc -l
```

画像ファイルとラベルファイルの数が一致していればOK。

---

## ラベルファイルの形式確認

```bash
cat /Users/kohei/ElbowVision_Dev/data/labels/train/AP_B_001.txt
```

正しい形式（1行）:
```
0  0.500 0.500 0.820 0.900  0.210 0.120 2  0.450 0.480 2  0.550 0.480 2  0.500 0.850 2
│  │─── bounding box ────│  │──kp0──│ v  │──kp1──│ v  │──kp2──│ v  │──kp3──│ v
クラス                       humerus  L.epic      M.epic      forearm
```

- 数値はすべて 0〜1 で正規化された座標
- `v` は visibility: `2` = 可視（通常はすべて2）

---

## トラブル対応

### Local storage が使えない場合（macOS Sequoia）

```bash
# 環境変数で許可する
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/kohei
label-studio start
```

### アノテーションがエクスポートされない

- 各画像で「Submit」を押したか確認（「Skip」では保存されない）
- Project Settings →「Annotations」→ Submitted annotations の数を確認

### ポート8080 が使用中

```bash
label-studio start --port 8081
```
