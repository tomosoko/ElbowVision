# Phase 2 実施プロトコル

**目的:** 新ファントム + 実X線30枚以上での多角度Bland-Altman検証  
**期間目標:** 2026年8〜10月（JSRT秋季前）  
**パイプライン状態:** ✅ 全スクリプト実装済み・動作確認済み

---

## Step 1: ファントム準備

### 1.1 必要なファントム仕様

| 項目 | 要件 |
|------|------|
| 材質 | 骨等価樹脂（HU 300〜800 相当） |
| 関節構造 | 肘関節可動式（固定ネジで任意角度ロック可能） |
| 角度範囲 | 90°〜180°（10°刻みで固定できること） |
| 代替案 | 既存ファントム patient008 の継続使用（多角度撮影） |

### 1.2 CT撮影（伸展位 180°）

```
CT装置: Canon Aquilion（または相当機種）
管電圧:  100 kVp
管電流:  50〜80 mAs
スキャン: Helical
FOV:     180 mm（上肢小FOV）
Matrix:  512 × 512
再構成:  1.0 mm厚 / 0.8 mm間隔（オーバーラップ）
カーネル: FC43（骨高精細）
体位:    肘伸展位（屈曲0°、laterality=R で撮影）
DICOM保存: 必須（anonymized）
```

---

## Step 2: DRRライブラリ構築

```bash
# 患者登録 + DRRライブラリ自動構築
python scripts/add_patient.py \
  --patient_id P009 \
  --ct_dir "data/raw_dicom/ct_volume/患者ディレクトリ名" \
  --laterality R \
  --series_num 4 \
  --hu_min 50 --hu_max 800 \
  --angle_min 60 --angle_max 180 --angle_step 1

# 生成確認
ls data/drr_library/P009_*.npz
python scripts/self_test_library.py \
  --library data/drr_library/P009_series4_R_60to180.npz \
  --test_angles "90,100,110,120,130,140,150,160,170,180" \
  --loo
# 期待値: MAE < 1°（LOO）
```

---

## Step 3: 実X線撮影プロトコル

### 3.1 角度設定と撮影計画

```bash
# 撮影プロトコルシート生成（印刷して撮影室で使用）
python scripts/generate_shooting_protocol.py \
  --patient_id P009 \
  --angle_start 90 --angle_end 180 --angle_step 10 \
  --n_repeat 3 \
  --out_dir results/shooting_protocol/
```

### 3.2 撮影手順（各角度）

1. **ゴニオメーターで角度確認**（GT角度として記録）
2. **固定ネジで関節ロック**（撮影中に動かないこと）
3. **LAT撮影（側面像）** — 内外側上顆が重なるポジショニング
4. **DICOM保存**（ファイル名に角度を含める例: `P009_LAT_090deg.dcm`）
5. **DICOMをPNGに変換**（または直接DICOM対応スクリプトを使用）

### 3.3 必要枚数の目安

| 目的 | 枚数 | 条件 |
|------|------|------|
| Bland-Altman 最低限 | 30枚 | 10角度 × 3回撮影 |
| 統計的に十分 | 50枚以上 | 10角度 × 5回 or 詳細ステップ |
| LOA CI信頼性向上 | 100枚 | 多角度 or 複数患者 |

### 3.4 GT角度記録フォーマット

```
data/real_xray/patients_phase2.csv に追記:

patient_id, xray_path, gt_angle_deg, laterality, series_num, hu_min, hu_max, library_path, note
P009, data/real_xray/images/P009_LAT_090.png, 90.0, R, 4, 50, 800, data/drr_library/P009_series4_R_60to180.npz, standard_LAT
P009, data/real_xray/images/P009_LAT_100.png, 100.0, R, 4, 50, 800, data/drr_library/P009_series4_R_60to180.npz, standard_LAT
...
```

---

## Step 4: バッチ評価

```bash
# 全X線に対して類似度マッチングを実行 + Bland-Altman解析
python scripts/eval_realxray_batch.py \
  --patient_list data/real_xray/patients_phase2.csv \
  --out_dir results/phase2_eval/ \
  --method similarity

# 出力確認
cat results/phase2_eval/summary.txt
# 期待値（Phase 2目標）:
#   MAE < 5°
#   Bias ≤ ±3° → PASS
#   LoA ≤ ±8° → PASS
#   ICC(3,1) ≥ 0.90 → PASS
```

---

## Step 5: 結果解析・論文更新

### 5.1 メトリクス比較（オプション）

```bash
python scripts/compare_metrics.py \
  --library data/drr_library/P009_series4_R_60to180.npz \
  --xray_dir data/real_xray/images/ \
  --gt_csv data/real_xray/patients_phase2.csv \
  --out_dir results/metric_comparison_phase2/
```

### 5.2 LaTeXテーブル更新

```bash
# Phase 2結果CSVができたら再実行
python scripts/generate_paper_latex.py --out_dir results/paper_latex/
```

### 5.3 論文更新箇所

| セクション | 更新内容 |
|-----------|---------|
| 3.3 Results (Real X-ray) | Phase 2 Bland-Altman数値で埋める |
| Abstract | n=X (実X線枚数) を更新 |
| Discussion 4.1 | Phase 2 MAEをDRR val (1.41°) と比較 |
| Conclusion | Phase 2成果を追記 |
| Table 1b | LoA実データで更新 |

---

## Step 6: 精度不足時のドメイン適応

Phase 2 MAE > 10° の場合のみ実施:

```bash
# ConvNeXt fine-tuning（Phase 2実X線で）
python scripts/finetune_angle_estimator.py \
  --real_xray_dir data/real_xray/images/ \
  --gt_csv data/real_xray/patients_phase2.csv \
  --base_model runs/angle_estimator/best.pth \
  --out_dir runs/angle_estimator_finetuned/ \
  --epochs 100 --lr 3e-6 --batch_size 8
```

---

## チェックリスト（逐次チェック）

### Phase 2 開始前
- [ ] ファントム準備完了（角度固定機構確認）
- [ ] CT撮影完了（伸展位 DICOM保存済み）
- [ ] `add_patient.py` でDRRライブラリ構築完了
- [ ] LOO自己テスト MAE < 1° 確認

### 撮影後
- [ ] 全角度（90-180°, 10°刻み）× 3回以上撮影完了
- [ ] GT角度CSVに記録完了（`patients_phase2.csv`）
- [ ] 画像品質確認（内外側上顆重なり OK）

### 解析後
- [ ] `eval_realxray_batch.py` 実行完了
- [ ] Bland-Altman: Bias ≤ ±3° → PASS
- [ ] Bland-Altman: LoA ≤ ±8° → PASS
- [ ] Bland-Altman: ICC ≥ 0.90 → PASS
- [ ] 論文 Results 3.3 を実データで更新
- [ ] LaTeXテーブル再生成

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| 推定角度が実際と大きくずれる（>15°） | lateralityミスマッチ | `--laterality R` を確認（左腕CTでもR） |
| NCC peak値が低い（<0.3） | ポジショニング非標準 / 画像品質 | 撮り直し・auto_crop確認 |
| 処理時間が40秒以上 | DRRキャッシュ未使用 | `--library` オプション確認 |
| Bland-Altman: LoA が広い（>±10°） | 特定角度での系統誤差 | 角度別誤差分布を確認 → 問題角度を除外or再撮影 |
| LOO MAE > 5° | DRRライブラリ品質問題 | `--hu_min/--hu_max` 調整、CTカーネル確認 |

---

*作成日: 2026-04-06  
Phase 2 実施時に更新すること*
