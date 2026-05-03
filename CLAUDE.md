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
         → Bland-Altman (n=510 LAT): MAE=0.467°, ICC=0.9988, Bias=-0.176°
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

## テスト（1745件全通過）
```bash
cd /Users/kohei/develop/research/ElbowVision
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pytest -q
```
| テストファイル | 件数 | 対象 |
|---|---|---|
| `tests/test_bland_altman.py` | 18 | Bland-Altman解析 |
| `tests/test_ct_reorient.py` | 47 | CT再方向付け + _axis_arrow + parse_coords |
| `tests/test_elbow_synth.py` | 24 | DRR生成・回転行列・transform_landmarks_canonical |
| `tests/test_angle_functions.py` | 13 | carrying/flexion angle計算 |
| `tests/test_make_yolo_label.py` | 15 | YOLOラベル生成・透視投影 |
| `tests/test_warmup_scheduler.py` | 8 | WarmupCosineScheduler |
| `tests/test_elbow_dataset.py` | 9 | ElbowDataset._resolve_img_path + __getitem__ |
| `tests/test_inference.py` | 18 | 推論パイプライン・CSV出力 |
| `tests/test_dicom_to_png.py` | 16 | DICOM→PNG変換 |
| `tests/test_eval_realxray_batch.py` | 17 | Phase 2 Bland-Altman (ICC・MAE・LoA・PASS/FAIL) |
| `tests/test_drr_utils.py` | 26 | drr_utils.py 純関数（SSIM・Dice・histogram_match等） |
| `elbow-train/tests/test_synth.py` | 29 | elbow_synth.py DRR生成・回転行列・ランドマーク投影 |
| `elbow-api/tests/test_api.py` | 67 | FastAPI エンドポイント（view_type・ConvNeXt second_opinion含む） |
| `elbow-api/tests/test_inference_positioning.py` | 49 | estimate_positioning_correction・epicondyle比率閾値 |
| `elbow-api/tests/test_inference_cv.py` | 42 | detect_bone_landmarks_classical・GradCAM overlay等 |
| `elbow-api/tests/test_validate_angle.py` | 28 | validate_angle_with_edges（Canny+Hough信頼度判定） |
| `elbow-api/tests/test_gradcam.py` | 23 | GradCAM クラス（generate・正規化・フック・target_idx） |
| `tests/test_phantom_seg_quality.py` | 65 | phantom_seg_quality.py 純関数（Dice・HD95・セグ手法・モンテカルロ評価） |
| `elbow-api/tests/test_convnext_model.py` | 31 | ElbowConvNeXt（定数・init・head構造・forward shape・勾配・モード切替） |
| `tests/test_phantom_radiomics_stability.py` | 51 | phantom_radiomics_stability.py 純関数（to_uint8_roi・GLCM・ROI抽出・条件生成・ICC） |
| `tests/test_similarity_matching.py` | 70 | similarity_matching.py 純関数（ncc・ssim・nmi・compute_similarity・crop_to_bone・preprocess_image・extract_edges・_parabolic_peak・MatchResult・DRRLibraryCache初期化） |
| `tests/test_calibrate_confidence.py` | 58 | calibrate_confidence.py 純関数（load_results・compute_threshold_metrics・棄却率・True Reject Rate・False Pass Rate・MAE計算） |
| `tests/test_compare_metrics.py` | 86 | compare_metrics.py 純関数（ALL_METRICS定数・evaluate_xray_all_metrics・戻り値構造・err/bias不変条件・メトリクス制限・決定論性） |
| `tests/test_drr_param_sweep.py` | 42 | drr_param_sweep.py 純関数（compute_ssim・hist_intersection・edge_ratio・evaluate_drr） |
| `tests/test_analyze_joint_motion.py` | 41 | analyze_joint_motion.py 純関数（calc_flexion_angle・bone_outline_2d） |
| `tests/test_compare_methods.py` | 48 | compare_methods.py 純関数（ncc・edges・preprocess_drr・crop_xray・preprocess_xray） |
| `tests/test_eval_angle_estimator.py` | 41 | eval_angle_estimator.py（ANGLE_MIN/MAX定数・逆正規化式・AngleEstimator init・forward shape・勾配・モード切替） |
| `tests/test_flexion_bone_seg.py` | 50 | flexion_bone_seg.py 純関数（otsu_threshold・two_stage_otsu・rotation_matrix_z・bone_dice・build_forearm_weight_3d） |
| `tests/test_flexion_2d_warp.py` | 46 | flexion_2d_warp.py 純関数（KP_ORDER定数・project_landmarks_2d・ssim_gray・draw_landmarks・tps_warp） |
| `tests/test_dataset_stats.py` | 37 | dataset_stats.py（print_summary・plot_angle_distribution・plot_view_counts・plot_sample_grid） |
| `tests/test_experiment_flexion_accuracy.py` | 54 | experiment_flexion_accuracy.py 純関数（compute_dice・compute_ssim・landmark_error_mm・to_mm・procrustes_rotation・nearest_volume_for_angle・rotation_matrix_custom_axis・rotate_volume_data_driven） |
| `tests/test_approachB_landmark_model.py` | 52 | approachB_landmark_model.py 純関数（定数・to_mm・build_humerus_frame・transform_to_frame・hf_to_norm・predict_hf） |
| `tests/test_bland_altman_analysis.py` | 34 | bland_altman_analysis.py（bland_altman_plot・ファイル出力・統計印字・エッジケース・LoA/SD数式） |
| `tests/test_build_drr_library.py` | 53 | build_drr_library.py（TARGET_SIZE定数・preprocess_drr・build_library npz出力・angles/drrs/meta検証・elbow_synthモック） |
| `tests/test_compare_drr_real.py` | 50 | compare_drr_real.py 純関数（extract_bone_region・compute_edge・compute_ssim・compute_fft_profile・histogram_intersection・compute_contrast_ratio） |
| `tests/test_pipeline_synth_to_yolo.py` | 52 | pipeline_synth_to_yolo.py 純関数（定数・build_bone_split・draw_keypoints・synthesize_lat_drr・gamma補正） |
| `tests/test_analyze_ct_hu.py` | 40 | analyze_ct_hu.py（analyze_hu返値構造・パーセンタイル精度・非空気フィルタリング・HU素材分類・matplotlibコール検証） |
| `tests/test_generate_paper_latex.py` | 58 | generate_paper_latex.py（_latex_escape・_table_wrap・全6テーブル生成関数・fallback/skip動作） |
| `tests/test_batch_multi_patient.py` | 46 | batch_multi_patient.py（定数・PatientConfig dataclass・_apply_domain_aug・build_pooled_dataset） |
| `tests/test_analyze_flexion.py` | 57 | analyze_flexion.py（定数・SERIESdict・compare_landmarks・compute_metrics SSIM/Dice・write_report 評価閾値） |
| `tests/test_test_robustness.py` | 52 | test_robustness.py（画像劣化純関数6種・PERTURBATIONS dict構造・identity level・clipping・monotonicity） |
| `tests/test_measure_landmarks.py` | 56 | measure_landmarks.py（定数・precise_landmark_measurement・座標範囲・解剖学的整合性・上顆検出・対称性・エッジケース・generate_report） |
| `tests/test_ct_to_xray_improved.py` | 50 | ct_to_xray_improved.py（to_mm・to_voxel・build_humerus_frame・transform_to_frame・procrustes_align・compute_rotation_axis_from_arc・build_rotation_around_axis） |

## 技術ノート

- 骨閾値: 2段階Otsu（ファントム骨素材対応）
- 回転中心: 滑車/小頭中心（上顆中点より前方）
- キーポイント6点: humerus_shaft, lateral/medial_epicondyle, forearm_shaft, radial_head, olecranon
- HUウィンドウ: ファントム最適 hu_min=50, hu_max=800
