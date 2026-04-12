"""
ファントムCTボリュームを使ったRadiomics特徴量安定性解析

CT再構成条件（ノイズレベル・平滑化・シャープニング）の違いが
Radiomics特徴量にどう影響するかをシミュレーションで評価する。

安定な特徴量 (ICC > 0.75) は撮影条件が変わっても信頼できる。

使い方:
  python3 phantom_radiomics_stability.py
"""

import os
import glob
import json
import numpy as np
import pydicom
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.feature import graycomatrix, graycoprops
from itertools import product

# ─── 設定 ────────────────────────────────────────────────────────────────────
DICOM_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_dicom", "ct")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results", "radiomics_stability")
os.makedirs(OUT_DIR, exist_ok=True)

ROI_SIZE  = 32   # ROIサイズ (voxel)
N_SLICES  = 10   # 解析に使うスライス数（中央から前後）

# ─── DICOM読み込み ────────────────────────────────────────────────────────────
def load_volume(dicom_dir):
    files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope     = float(getattr(slices[0], "RescaleSlope",     1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    vol = vol * slope + intercept
    print(f"Volume: {vol.shape}, HU: {vol.min():.0f}~{vol.max():.0f}")
    return vol

# ─── ROI自動配置 ──────────────────────────────────────────────────────────────
def extract_roi_slices(volume, n_slices=N_SLICES, roi_size=ROI_SIZE):
    """
    中央付近の複数スライスからアクリル（均一）領域のROIを切り出す
    HU値が中間帯（アクリル）の最も均一な領域を選ぶ
    """
    mid_z = volume.shape[0] // 2
    z_indices = range(mid_z - n_slices // 2, mid_z + n_slices // 2 + 1)
    rois = []
    positions = []

    for z in z_indices:
        sl = volume[z]
        # アクリル相当のHU帯 (-200 ~ 500) でマスク
        mask = (sl > -200) & (sl < 500)
        if not np.any(mask):
            continue
        # 最も均一な領域を探す (std最小)
        best_std = 1e9
        best_roi = None
        best_pos = None
        h, w = sl.shape
        step = roi_size // 2
        for y in range(0, h - roi_size, step):
            for x in range(0, w - roi_size, step):
                patch = sl[y:y+roi_size, x:x+roi_size]
                m = mask[y:y+roi_size, x:x+roi_size]
                if m.mean() < 0.8:  # 80%以上マスク内
                    continue
                std = patch.std()
                if std < best_std:
                    best_std = std
                    best_roi = patch.copy()
                    best_pos = (z, y, x)
        if best_roi is not None:
            rois.append(best_roi)
            positions.append(best_pos)

    print(f"ROI抽出: {len(rois)}個 (各{roi_size}x{roi_size}px)")
    return rois, positions

# ─── 条件シミュレーション ─────────────────────────────────────────────────────
def apply_conditions(volume):
    """
    CT再構成条件を模擬した複数バリエーションを生成

    Returns:
        dict: {condition_name: volume}
    """
    conditions = {}

    # ① ベースライン（実CTに近いPoissonノイズ付き標準条件）
    base = volume + np.random.normal(0, 8, volume.shape).astype(np.float32)
    conditions["original"] = base

    # ② ガウスノイズ（低線量シミュレーション）
    for sigma in [5, 15, 30]:
        noisy = volume + np.random.normal(0, sigma, volume.shape).astype(np.float32)
        conditions[f"noise_σ{sigma}"] = noisy

    # ③ Gaussianぼかし（軟部組織カーネル相当）
    for s in [0.5, 1.0, 2.0]:
        blurred = gaussian_filter(volume.astype(np.float32), sigma=s)
        conditions[f"blur_σ{s}"] = blurred

    # ④ シャープニング（骨カーネル相当）
    for alpha in [0.5, 1.0, 2.0]:
        blurred = gaussian_filter(volume.astype(np.float32), sigma=1.0)
        sharp = volume + alpha * (volume - blurred)
        conditions[f"sharp_α{alpha}"] = sharp.astype(np.float32)

    # ⑤ ノイズ + ぼかし（低線量 + 軟部カーネル）
    noisy15 = volume + np.random.normal(0, 15, volume.shape).astype(np.float32)
    conditions["noise15+blur1"] = gaussian_filter(noisy15, sigma=1.0)

    print(f"生成条件数: {len(conditions)}")
    return conditions

# ─── Radiomics特徴量抽出 ─────────────────────────────────────────────────────
def to_uint8_roi(patch):
    """ROIをuint8（0-255）に正規化"""
    p_min, p_max = patch.min(), patch.max()
    if p_max - p_min < 1e-6:
        return np.zeros_like(patch, dtype=np.uint8)
    return ((patch - p_min) / (p_max - p_min) * 255).astype(np.uint8)


def extract_glcm_features(patch_uint8):
    """GLCM（グレーレベル共起行列）から5つの一次統計＋テクスチャ特徴量を抽出"""
    # GLCM: 4方向・距離1
    distances  = [1]
    angles     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(patch_uint8, distances=distances, angles=angles,
                         levels=256, symmetric=True, normed=True)

    features = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        features[prop] = float(vals.mean())

    # 一次統計
    flat = patch_uint8.flatten().astype(np.float64)
    features["mean"]    = float(flat.mean())
    features["std"]     = float(flat.std())
    features["entropy"] = float(-np.sum(
        [p * np.log2(p + 1e-12) for p in np.bincount(patch_uint8.flatten()) / patch_uint8.size]
    ))
    return features


def extract_features_from_conditions(conditions, rois_original, positions):
    """
    各条件×各ROIの特徴量を抽出

    Returns:
        dict: {feature_name: {condition_name: [values per ROI]}}
    """
    feature_names = ["contrast", "dissimilarity", "homogeneity",
                     "energy", "correlation", "mean", "std", "entropy"]
    data = {f: {c: [] for c in conditions} for f in feature_names}

    for cond_name, vol in conditions.items():
        for _, (z, y, x) in zip(rois_original, positions):
            roi_size = rois_original[0].shape[0]
            patch = vol[z, y:y+roi_size, x:x+roi_size]
            patch_u8 = to_uint8_roi(patch)
            feats = extract_glcm_features(patch_u8)
            for fname in feature_names:
                data[fname][cond_name].append(feats[fname])

    return data

# ─── ICC計算 ─────────────────────────────────────────────────────────────────
def calculate_icc(values_by_condition):
    """
    ICC(2,1) を計算: 条件間でどれだけ一致しているか
    ICC > 0.75 → 良好な安定性
    ICC > 0.90 → 優秀な安定性

    values_by_condition: {condition: [v1, v2, ..., vN]}
    """
    try:
        import pingouin as pg
        import pandas as pd

        rows = []
        for cond, vals in values_by_condition.items():
            for i, v in enumerate(vals):
                rows.append({"roi": i, "condition": cond, "value": v})
        df = pd.DataFrame(rows)
        icc_result = pg.intraclass_corr(
            data=df, targets="roi", raters="condition", ratings="value"
        )
        # ICC(2,1) = Two-way mixed, single measures
        icc21 = icc_result[icc_result["Type"] == "ICC2"]["ICC"].values
        if len(icc21) > 0:
            return float(icc21[0])
    except Exception:
        pass

    # フォールバック: 手動計算
    all_vals = np.array(list(values_by_condition.values()))  # (n_cond, n_roi)
    n_cond, n_roi = all_vals.shape
    grand_mean = all_vals.mean()
    ss_total   = ((all_vals - grand_mean) ** 2).sum()
    roi_means  = all_vals.mean(axis=0)
    ss_between = n_cond * ((roi_means - grand_mean) ** 2).sum()
    ms_between = ss_between / (n_roi - 1)
    ms_error   = (ss_total - ss_between) / (n_roi * (n_cond - 1))
    if ms_between + (n_cond - 1) * ms_error < 1e-12:
        return 0.0
    icc = (ms_between - ms_error) / (ms_between + (n_cond - 1) * ms_error)
    return float(icc)

# ─── 可視化 ──────────────────────────────────────────────────────────────────
def plot_stability_chart(stability, conditions, feat_data, out_path):
    """安定性チャート: CV棒グラフ + 条件別相対変化ヒートマップ"""
    feat_names = list(stability.keys())
    cond_names = [c for c in conditions if c != "original"]
    cv_vals    = [stability[f]["cv_pct"] for f in feat_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ① CV棒グラフ
    colors = ["#2ecc71" if v < 5 else "#e67e22" if v < 20 else "#e74c3c"
              for v in cv_vals]
    axes[0].barh(feat_names, cv_vals, color=colors)
    axes[0].axvline(5,  color="green",  linestyle="--", linewidth=1, label="安定 (<5%)")
    axes[0].axvline(20, color="orange", linestyle="--", linewidth=1, label="不安定 (≥20%)")
    axes[0].set_xlabel("変動係数 CV (%)")
    axes[0].set_title("Radiomics特徴量の安定性 (CV)")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    for i, (f, v) in enumerate(zip(feat_names, cv_vals)):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # ② 条件別相対変化ヒートマップ
    matrix = np.zeros((len(feat_names), len(cond_names)))
    for i, f in enumerate(feat_names):
        orig_mean = stability[f]["orig_mean"]
        for j, c in enumerate(cond_names):
            cond_mean = stability[f]["cond_means"][c]
            matrix[i, j] = abs(cond_mean - orig_mean)

    vmax = np.percentile(matrix, 95) if matrix.max() > 0 else 1.0
    im = axes[1].imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)
    axes[1].set_xticks(range(len(cond_names)))
    axes[1].set_xticklabels(cond_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(feat_names)))
    axes[1].set_yticklabels(feat_names, fontsize=8)
    axes[1].set_title("条件別 相対変化率 (%) vs original")
    plt.colorbar(im, ax=axes[1], label="|変化率| (%)")

    for i in range(len(feat_names)):
        for j in range(len(cond_names)):
            axes[1].text(j, i, f"{matrix[i,j]:.2f}",
                          ha="center", va="center", fontsize=5, color="black")

    plt.suptitle("CT再構成条件変化に対するRadiomics特徴量の安定性\n(アクリルファントムボリューム)",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"図保存: {out_path}")


def plot_feature_distribution(feat_data, feature_name, out_path):
    """特定の特徴量の条件間分布をバイオリンプロット"""
    cond_names = list(feat_data.keys())
    vals       = [feat_data[c] for c in cond_names]

    fig, ax = plt.subplots(figsize=(14, 4))
    parts = ax.violinplot(vals, showmedians=True)
    ax.set_xticks(range(1, len(cond_names) + 1))
    ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(feature_name)
    ax.set_title(f"'{feature_name}' の条件間分布")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roi_preview(volume, positions, roi_size, out_path):
    """ROI配置の確認画像"""
    mid_z = volume.shape[0] // 2
    sl = volume[mid_z]
    # 正規化
    sl_vis = np.clip(sl, -200, 500)
    sl_vis = ((sl_vis - sl_vis.min()) / (sl_vis.max() - sl_vis.min()) * 255).astype(np.uint8)
    sl_bgr = cv2.cvtColor(sl_vis, cv2.COLOR_GRAY2BGR)
    for z, y, x in positions:
        if z == mid_z:
            cv2.rectangle(sl_bgr, (x, y), (x+roi_size, y+roi_size), (0, 255, 80), 2)
    cv2.imwrite(out_path, sl_bgr)
    print(f"ROIプレビュー: {out_path}")

# ─── メイン ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Radiomics特徴量安定性解析")
    print("=" * 60)

    np.random.seed(42)

    # 1. ボリューム読み込み
    print("\n[1/5] DICOMボリューム読み込み...")
    volume = load_volume(DICOM_DIR)

    # 2. ROI抽出
    print("\n[2/5] ROI自動抽出...")
    rois, positions = extract_roi_slices(volume)
    if not rois:
        print("ERROR: ROIが見つかりません")
        return

    plot_roi_preview(volume, positions, ROI_SIZE,
                      os.path.join(OUT_DIR, "roi_preview.png"))

    # 3. 条件シミュレーション
    print("\n[3/5] CT条件シミュレーション...")
    conditions = apply_conditions(volume)

    # 4. 特徴量抽出
    print("\n[4/5] Radiomics特徴量抽出...")
    feat_data = extract_features_from_conditions(conditions, rois, positions)

    # 5. 安定性評価 (CV & 相対変化率)
    print("\n[5/5] 安定性評価...")
    stability = {}
    cond_names = list(conditions.keys())
    orig_name  = "original"

    for feat_name, cond_dict in feat_data.items():
        orig_vals  = np.array(cond_dict[orig_name])
        orig_mean  = orig_vals.mean()

        # 各条件での平均値
        cond_means = {c: np.array(v).mean() for c, v in cond_dict.items()}

        # 全条件込みのCV (変動係数)
        all_vals = np.concatenate(list(cond_dict.values()))
        cv = all_vals.std() / (abs(all_vals.mean()) + 1e-9) * 100

        # 最大絶対変化率 (originalからの最大ズレ、絶対値ベース)
        feat_range = max(abs(v) for v in cond_means.values()) - min(abs(v) for v in cond_means.values())
        max_abs_change = max(
            abs(cond_means[c] - orig_mean)
            for c in cond_names if c != orig_name
        )

        stability[feat_name] = {
            "cv_pct":          round(cv, 2),
            "max_abs_change":  round(max_abs_change, 4),
            "orig_mean":       round(orig_mean, 4),
            "cond_means":      {c: round(v, 4) for c, v in cond_means.items()},
        }

        grade = "🟢安定" if cv < 5 else "🟠やや不安定" if cv < 20 else "🔴不安定"
        print(f"  {feat_name:15s}: CV={cv:5.1f}%  最大絶対変化={max_abs_change:.4f}  {grade}")

    # 結果保存
    with open(os.path.join(OUT_DIR, "stability_results.json"), "w") as f:
        json.dump(stability, f, indent=2, ensure_ascii=False)

    # 可視化
    plot_stability_chart(stability, conditions, feat_data,
                         os.path.join(OUT_DIR, "stability_chart.png"))

    # サマリー
    print(f"\n{'='*60}")
    stable = [f for f, v in stability.items() if v["cv_pct"] < 5]
    unstable = [f for f, v in stability.items() if v["cv_pct"] >= 20]
    print(f"安定特徴量 (CV<5%):  {stable}")
    print(f"不安定特徴量 (CV≥20%): {unstable}")
    print(f"結果ディレクトリ: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
