# 新人向けスライド — Mac Mini作業手順

## 目的

ElbowVisionの新人向け紹介スライドを完成させる。
具体的には以下を実行:

1. 3ボリュームCT（ct_180/ct_135/ct_90）からDRR GIFアニメーションを生成
2. 画像をBase64埋め込みした自己完結HTMLスライドを生成

## 前提

- ElbowVisionリポジトリが `~/develop/research/ElbowVision/` にある
- `data/raw_dicom/` に `ct_180/`, `ct_135/`, `ct_90/`（または `ct_volume/`）が存在
- `elbow-api/venv/` にPython環境がある

## 手順

### Step 1: git pull

```bash
cd ~/develop/research/ElbowVision
git pull
```

### Step 2: DRR GIFアニメーション生成

3軸（回旋・屈曲・視点切り替え）で変化するDRRアニメーションを生成する。

```bash
cd ~/develop/research/ElbowVision
~/develop/research/ElbowVision/elbow-api/venv/bin/python scripts/generate_drr_gif.py
```

このスクリプトは以下の4つのGIFを出力する:
- `slides/drr_rotation.gif` — 回旋（前腕回内/回外）のみ
- `slides/drr_flexion.gif` — 屈曲（曲げ伸ばし）のみ
- `slides/drr_combined.gif` — 回旋+屈曲を同時に変化
- `slides/drr_animation.gif` — 3つ横並び（スライド埋め込み用）

もし3ボリュームCTがあるなら、`drr_motion_series.py` も実行して屈伸のフリップブック画像を生成:

```bash
~/develop/research/ElbowVision/elbow-api/venv/bin/python scripts/drr_motion_series.py
```

### Step 3: スライドHTML生成（画像Base64埋め込み）

```bash
cd ~/develop/research/ElbowVision

# 画像をBase64埋め込みしたMDを生成
python3 -c "
import re, base64, os, mimetypes
slides_dir = 'slides'
with open(os.path.join(slides_dir, 'newcomer_intro.md')) as f:
    content = f.read()
def replace_img(m):
    alt, src = m.group(1), m.group(2)
    abs_path = os.path.normpath(os.path.join(slides_dir, src))
    if not os.path.exists(abs_path):
        print(f'MISSING: {abs_path}')
        return m.group(0)
    mime = mimetypes.guess_type(abs_path)[0] or 'image/png'
    with open(abs_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f'Embedded: {abs_path}')
    return f'![{alt}](data:{mime};base64,{b64})'
new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, content)
with open(os.path.join(slides_dir, 'newcomer_intro_embedded.md'), 'w') as f:
    f.write(new_content)
print('Done')
"

# MarpでHTML変換
npx marp slides/newcomer_intro_embedded.md --html -o slides/newcomer_intro.html
```

### Step 4: 確認 & コミット

```bash
# ブラウザで確認
open slides/newcomer_intro.html

# コミット & プッシュ（HTMLは.gitignoreで除外済み）
git add slides/ scripts/generate_drr_gif.py
git commit -m "feat: 3軸DRR GIFアニメーション追加（Mac Mini生成）"
git push
```

## スライド構成（14枚）

1. タイトル: ElbowVision
2. 目次
3. 肘X線撮影の「あるある」
4. ElbowVisionのコンセプト（推論結果画像付き）
5. 実際の推論結果（AP/LAT並列表示）
6. AIの学習データどうする？
7. CTからX線を「作る」（パイプライン図）
8. DRRって何？（Beer-Lambert法則）
9. 角度を変えると...（DRRバリエーション図）
10. このアプローチの何がすごいか（比較表）
11. 使っている技術
12. 研究のロードマップ
13. まとめ
14. DRR生成デモ（GIFアニメーション）

## 注意

- `slides/newcomer_intro.html` と `slides/newcomer_intro_embedded.md` は `.gitignore` に登録済み（サイズが大きいため）
- スライドのソースは `slides/newcomer_intro.md`（画像は相対パス参照）
