"""
DRRパラメータ探索: 複数のHUウィンドウ・後処理設定でDRRを生成し、
実X線との類似度を定量比較するスクリプト

M4 Pro 64GBで並列実行可能
"""
import os
import sys
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# elbow_synth を import するためパスを追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume, auto_detect_landmarks,
    rotate_volume_and_landmarks, generate_drr,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "results/domain_gap_analysis/param_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

REAL_XRAY = {
    "AP":  os.path.join(PROJECT_ROOT, "data/real_xray/images/008_AP.png"),
    "LAT": os.path.join(PROJECT_ROOT, "data/real_xray/images/008_LAT.png"),
}
CT_DIR = os.path.join(PROJECT_ROOT, "data/raw_dicom/ct")


def load_real_gray(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def compute_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
    sigma1_sq = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def hist_intersection(img1, img2):
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).ravel()
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).ravel()
    h1n, h2n = h1 / (h1.sum() + 1e-8), h2 / (h2.sum() + 1e-8)
    return float(np.minimum(h1n, h2n).sum())


def edge_ratio(img1, img2):
    """img2 / img1 のエッジ密度比"""
    def edge_density(img):
        e = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0.5), 50, 150)
        return e.sum() / (255.0 * e.size)
    d1, d2 = edge_density(img1), edge_density(img2)
    return d2 / max(d1, 1e-8)


def generate_drr_with_postprocess(volume, landmarks, voxel_mm, axis,
                                   gamma, clahe_clip, target_max_atten,
                                   scatter_frac, output_size=256):
    """generate_drr のカスタム後処理版"""
    from scipy.ndimage import map_coordinates as _mc

    NP, NA, NM = volume.shape
    if axis == "AP":
        H, W, D = NP, NM, NA
    else:
        H, W, D = NP, NA, NM

    sid_mm = 1000.0
    SID_vox = sid_mm / voxel_mm
    D_s = max(SID_vox - D, 1.0)

    hh, ww = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32), indexing='ij')
    d_vals = np.arange(D, dtype=np.float32)
    inv_m = (D_s + d_vals) / SID_vox
    inv_m3 = inv_m[None, None, :]
    h_vol = H / 2.0 + (hh[:, :, None] - H / 2.0) * inv_m3
    w_vol = W / 2.0 + (ww[:, :, None] - W / 2.0) * inv_m3
    d_vol = np.tile(d_vals[None, None, :], (H, W, 1))

    if axis == "AP":
        coords = np.stack([h_vol.ravel(), d_vol.ravel(), w_vol.ravel()], axis=0)
    else:
        coords = np.stack([h_vol.ravel(), w_vol.ravel(), d_vol.ravel()], axis=0)

    samples = _mc(volume, coords, order=1, mode='constant', cval=0.0)
    mu_volume = samples.reshape(H, W, D)

    # Beer-Lambert
    line_integral = mu_volume.sum(axis=2)
    li_max = line_integral.max() + 1e-8
    line_integral = line_integral * (target_max_atten / li_max)
    transmission = np.exp(-line_integral)

    # 散乱線
    intensity = (1.0 - scatter_frac) * transmission + scatter_frac

    # ヒール効果
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32), indexing='ij')
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    max_dist_sq = cy ** 2 + cx ** 2 + 1e-8
    heel_effect = 1.0 - 0.05 * (dist_sq / max_dist_sq)
    intensity = intensity * heel_effect

    # 反転（骨=白）
    img = 1.0 - intensity
    bg_mask = img < 0.02

    # ウィンドウ
    fg = img[~bg_mask]
    if len(fg) > 0:
        p_low = np.percentile(fg, 2)
        p_high = np.percentile(fg, 99.5)
    else:
        p_low, p_high = 0.0, 1.0
    img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
    img[bg_mask] = 0.0

    # ガンマ補正
    img = np.power(img + 1e-8, gamma)
    img[bg_mask] = 0.0

    # uint8 変換
    img_u8 = (img * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)

    # リサイズ
    img_u8 = cv2.resize(img_u8, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return img_u8


def evaluate_drr(drr, real):
    """DRR 1枚を実X線と比較"""
    ssim = compute_ssim(drr, real)
    hist_sim = hist_intersection(drr, real)
    e_ratio = edge_ratio(real, drr)
    return {"ssim": ssim, "hist": hist_sim, "edge_ratio": e_ratio}


def run_volume_group(hu_min, hu_max, target_size, postprocess_configs):
    """
    同一CTボリューム設定で複数の後処理パラメータを一括評価。
    ボリューム読み込みは1回だけ。
    """
    print(f"\n  Loading volume: hu=[{hu_min},{hu_max}] size={target_size}")
    try:
        vol, _, lat, vox_mm = load_ct_volume(
            CT_DIR, laterality='R', series_num=1,
            hu_min=hu_min, hu_max=hu_max, target_size=target_size,
        )
        lm = auto_detect_landmarks(vol, laterality=lat)
    except Exception as e:
        print(f"  ERROR loading volume: {e}")
        return [{"config": (hu_min, hu_max, target_size, *pc),
                 "label": "error", "error": str(e), "score": -1}
                for pc in postprocess_configs]

    # AP/LAT ボリューム回転も1回だけ
    rotated = {}
    for view in ["AP", "LAT"]:
        flex = 170.0 if view == "AP" else 90.0
        rot_vol, rot_lm = rotate_volume_and_landmarks(vol, lm, 0.0, flex, base_flexion=90.0)
        rotated[view] = (rot_vol, rot_lm)

    real_imgs = {v: load_real_gray(REAL_XRAY[v], 256) for v in ["AP", "LAT"]}

    results_list = []
    for gamma, clahe_clip, target_max_atten, scatter_frac in postprocess_configs:
        label = (f"hu{hu_min}_{hu_max}_sz{target_size}_g{gamma:.1f}"
                 f"_cl{clahe_clip:.1f}_att{target_max_atten:.1f}_sc{scatter_frac:.2f}")
        config = (hu_min, hu_max, target_size, gamma, clahe_clip, target_max_atten, scatter_frac)

        view_results = {}
        for view in ["AP", "LAT"]:
            rot_vol, rot_lm = rotated[view]
            drr = generate_drr_with_postprocess(
                rot_vol, rot_lm, vox_mm, view,
                gamma=gamma, clahe_clip=clahe_clip,
                target_max_atten=target_max_atten,
                scatter_frac=scatter_frac, output_size=256,
            )
            cv2.imwrite(os.path.join(OUT_DIR, f"drr_{view}_{label}.png"), drr)
            view_results[view] = evaluate_drr(drr, real_imgs[view])

        score = sum(r["ssim"] + r["hist"] + min(r["edge_ratio"], 1.0)
                     for r in view_results.values()) / 2.0

        results_list.append({
            "config": config,
            "label": label,
            "results": view_results,
            "score": score,
        })
        print(f"    {label}: score={score:.4f} "
              f"(AP: SSIM={view_results['AP']['ssim']:.3f} "
              f"LAT: SSIM={view_results['LAT']['ssim']:.3f})")

    return results_list


def main():
    print("DRR パラメータ探索（2フェーズ: ボリューム設定 → 後処理）")
    print("=" * 60)

    t0 = time.time()
    results_all = []

    # ── Phase 1: HUウィンドウ + target_size（後処理はデフォルト固定）──
    # ボリューム読み込み回数を最小化: (hu_min, hu_max, target_size) ごとに1回
    hu_mins = [-200, 0, 50, 100]
    hu_maxs = [800, 1000]
    target_sizes = [128, 256]
    default_pp = [(0.7, 2.0, 4.0, 0.03)]  # (gamma, clahe, atten, scatter)

    vol_combos = [(hmin, hmax, ts)
                  for hmin in hu_mins for hmax in hu_maxs for ts in target_sizes]
    print(f"Phase 1: {len(vol_combos)} volume configs × 1 postprocess = {len(vol_combos)} runs")

    for hmin, hmax, ts in vol_combos:
        batch = run_volume_group(hmin, hmax, ts, default_pp)
        results_all.extend(batch)

    # Phase 1 ベスト
    valid = [r for r in results_all if r["score"] > 0]
    if not valid:
        print("ERROR: 全設定でエラー")
        return
    best_p1 = max(valid, key=lambda r: r["score"])
    best_hu_min, best_hu_max, best_ts = best_p1["config"][:3]
    print(f"\n{'='*60}")
    print(f"Phase 1 Best: hu=[{best_hu_min},{best_hu_max}] size={best_ts} "
          f"score={best_p1['score']:.4f}")
    print(f"{'='*60}")

    # ── Phase 2: 後処理パラメータ探索（ベストボリューム設定で）──
    gammas = [0.4, 0.5, 0.7, 1.0, 1.2]
    clahe_clips = [1.0, 1.5, 2.0, 3.0]
    attens = [2.5, 3.0, 4.0, 5.0, 6.0]
    scatters = [0.03, 0.08, 0.15]

    pp_combos = [(g, cl, att, sc)
                 for g in gammas for cl in clahe_clips
                 for att in attens for sc in scatters]
    print(f"\nPhase 2: {len(pp_combos)} postprocess configs (1 volume load)")

    batch = run_volume_group(best_hu_min, best_hu_max, best_ts, pp_combos)
    results_all.extend(batch)

    # ── 最終ランキング ──
    valid = [r for r in results_all if r["score"] > 0]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  Top 15 Configurations (elapsed: {time.time()-t0:.0f}s)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Score':>6} {'hu_min':>6} {'hu_max':>6} {'size':>4} "
          f"{'gamma':>5} {'clahe':>5} {'atten':>5} {'scat':>5}  "
          f"{'AP_SSIM':>7} {'AP_Hist':>7} {'LAT_SSIM':>8} {'LAT_Hist':>8}")
    print("-" * 100)

    for i, r in enumerate(valid[:15]):
        c = r["config"]
        ap = r["results"]["AP"]
        lat = r["results"]["LAT"]
        print(f"{i+1:>4} {r['score']:>6.3f} {c[0]:>6} {c[1]:>6} {c[2]:>4} "
              f"{c[3]:>5.1f} {c[4]:>5.1f} {c[5]:>5.1f} {c[6]:>5.2f}  "
              f"{ap['ssim']:>7.4f} {ap['hist']:>7.4f} {lat['ssim']:>8.4f} {lat['hist']:>8.4f}")

    # ベスト設定を保存
    best = valid[0]
    best_config_path = os.path.join(OUT_DIR, "best_config.txt")
    with open(best_config_path, "w") as f:
        c = best["config"]
        f.write(f"hu_min={c[0]}\n")
        f.write(f"hu_max={c[1]}\n")
        f.write(f"target_size={c[2]}\n")
        f.write(f"gamma={c[3]}\n")
        f.write(f"clahe_clip={c[4]}\n")
        f.write(f"target_max_attenuation={c[5]}\n")
        f.write(f"scatter_fraction={c[6]}\n")
        f.write(f"score={best['score']}\n")
    print(f"\nベスト設定を保存: {best_config_path}")

    # ベスト設定のDRR vs Real 比較画像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Best Config: hu=[{best['config'][0]},{best['config'][1]}] "
                 f"size={best['config'][2]} γ={best['config'][3]} "
                 f"CLAHE={best['config'][4]} atten={best['config'][5]}\n"
                 f"Score={best['score']:.4f}", fontsize=12)
    for i, view in enumerate(["AP", "LAT"]):
        real = load_real_gray(REAL_XRAY[view], 256)
        drr = cv2.imread(os.path.join(OUT_DIR, f"drr_{view}_{best['label']}.png"),
                         cv2.IMREAD_GRAYSCALE)
        diff = np.abs(real.astype(float) - drr.astype(float))

        axes[i, 0].imshow(real, cmap="gray")
        axes[i, 0].set_title(f"Real {view}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(drr, cmap="gray")
        rv = best["results"][view]
        axes[i, 1].set_title(f"DRR {view} (SSIM={rv['ssim']:.3f})")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(diff, cmap="hot")
        axes[i, 2].set_title(f"Diff (Hist={rv['hist']:.3f})")
        axes[i, 2].axis("off")

    fig.savefig(os.path.join(OUT_DIR, "best_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("比較画像保存完了")


if __name__ == "__main__":
    main()
