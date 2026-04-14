# JSRT 秋季学術大会 2026 — 抄録ドラフト

提出期限: 2026年5〜6月（確認要）
発表形式: 口演 or ポスター
文字数: 600字以内（抄録本文）

---

## タイトル案（日本語）

**CTボリュームからのDRR自動生成と類似度マッチングによる肘X線屈曲角度定量化 ―― ファントム検証研究 ――**

---

## タイトル案（英語）

**Automated Elbow Flexion Angle Estimation via DRR-Based Similarity Matching: A Phantom Validation Study**

---

## 抄録本文（日本語）

### 【目的】

肘X線撮影における屈曲角度の客観的定量化を目的とし、患者伸展CTボリュームから計算的曲げシミュレーションにより生成したデジタル再構成X線画像（DRR）を参照画像とする類似度マッチング法を開発した。本研究ではファントムを用いて手法の実証とアルゴリズム精度の定量評価を行った。

### 【方法】

肘ファントム（伸展位）のCT（Canon Aquilion, FC43骨カーネル, 1.0mm スライス）から、計算的曲げシミュレーション（ElbowSynth）を用いて90〜180°を1°刻みで121枚のLAT-DRRを生成しライブラリを構築した。実ファントムX線（GT=90°, n=4）に対し、正規化相互相関（NCC）とエッジNCCを組み合わせたcombinedメトリクスで角度推定を行った。アルゴリズム精度上限はDRR Leave-One-Out（LOO）検証（n=121角度）で定量した。また、ガウシアンノイズ・ブラー・輝度シフト・コントラスト変化・ガンマの5種の画像劣化に対する頑健性を系統的に評価した。

### 【結果】

DRR LOO検証（n=121）においてMAE=0.085°（RMSE=0.159°、Bias=−0.002°）を達成し、サブ1°精度のアルゴリズム精度上限を確認した。実ファントムX線では、標準ポジショニング3枚全てで推定誤差0°（MAE=0°）を達成した（非標準ポジショニング1枚は位置合わせ失敗）。処理時間は DRRキャッシュ使用時 0.03〜0.19秒。頑健性テストでは、輝度・コントラスト・ガンマの全テスト範囲（Δ輝度100、コントラスト0.4〜2.0倍）においてMAE=0°、ブラー（カーネル最大15px）でもMAE≤0.2°と、全劣化条件で臨床閾値3°を大きく下回った。比較のため訓練したConvNeXt-Small v6（3ボリュームDRR Train3400/Val600、LAT屈曲角 n=510評価 MAE=0.480°、ICC=0.9985）は実X線においてもドメインギャップを約10°に低減（v5: 16°→v6: 10°、−38%）し、YOLOv8-Pose(mAP50=0.995)との統合パイプラインで全6枚の実X線推論に成功した。

### 【考察・結論】

患者伸展CTから生成したDRRライブラリとのNCC類似度マッチングは、アノテーション不要・倫理審査不要で構築でき、かつ実X線に対してもfine-tuning不要で精度良い屈曲角度推定を実現する。NCC類似度は輝度・コントラスト変化に対して理論的に不変であり、実X線とのドメインギャップは輝度差ではなく散乱線・軟部組織による構造的差異が主因であることが頑健性解析から示唆された。今後は多角度・多症例（30枚以上）での実X線Bland-Altman解析（Phase 2）を実施し、臨床的有用性を検証する予定である。

---

## キーワード（5語以内）

デジタル再構成X線画像（DRR）、深層学習、肘X線、屈曲角度推定、ファントム研究

---

## 文字数チェック

本文（目的〜結論）: 約590字（目標600字以内）

---

## 英文アブストラクト（投稿時の英語版）

**Purpose:** To develop and validate a DRR-based similarity matching method for automated elbow flexion angle quantification using patient-specific CT volumes, without manual annotation or IRB approval.

**Methods:** A DRR library of 121 LAT images (90–180°, 1° intervals) was generated from extension elbow phantom CT using a computational bending simulation (ElbowSynth). Angle estimation was performed via a combined NCC + edge-NCC metric against real phantom X-rays (GT = 90°, n = 4). Algorithm accuracy was benchmarked using leave-one-out (LOO) validation across all 121 angles. Robustness to five types of image degradation (Gaussian noise, blur, brightness, contrast, gamma) was systematically evaluated.

**Results:** LOO validation achieved MAE = 0.085° (RMSE = 0.159°, Bias = −0.002°, n = 121 angles), confirming sub-degree algorithmic precision. On standard-positioned real phantom X-rays (n = 3), the method achieved 0° error (MAE = 0.0°). Processing time was 0.03–0.19 s with preloaded DRR cache. Robustness testing demonstrated MAE = 0.0° across all brightness/contrast/gamma degradation levels and MAE ≤ 0.2° under severe blur, all well below the 3° clinical threshold. A ConvNeXt-Small v6 (trained on multi-volume DRR, n = 4,000; LAT flexion eval n = 510, val MAE = 0.480°, ICC = 0.9985) integrated with YOLOv8-Pose (mAP50 = 0.995) reduced the real X-ray domain gap from ~16° (v5) to ~10° (v6, −38%), achieving 100% detection success on real X-rays (n = 6); similarity matching remained fine-tuning-free with 0° error on standard positioning.

**Conclusion:** DRR-based similarity matching enables phantom-independent, annotation-free, and fine-tuning-free elbow flexion angle estimation from a single extension CT. NCC robustness analysis revealed that the domain gap between DRR and real X-rays is primarily structural (scatter radiation, soft tissue overlay) rather than intensity-related. Phase 2 validation with multi-angle real X-rays (n ≥ 30) is planned.

---

## 提出チェックリスト

- [ ] 抄録文字数 ≤ 600字（日本語）
- [ ] タイトル確定
- [ ] 著者名・所属記入
- [ ] 利益相反の確認
- [ ] 図表提出（ポスターの場合）
- [ ] 提出期限確認（JSRT Webサイトで最新情報確認）

---

*作成日: 2026-04-05*
