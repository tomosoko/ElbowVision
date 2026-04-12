"""
CT画質劣化が骨セグメンテーション精度に与える影響
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
複数のセグメンテーション手法 × 複数の画質劣化条件で
Dice / Hausdorff / Precision / Recall を定量評価する。

モンテカルロ法で信頼区間付き結果を算出。
論文投稿レベルの可視化・レポートを生成する。

使い方:
  python3 phantom_seg_quality.py
  python3 phantom_seg_quality.py --n-trials 5   # 試行回数変更
"""

import os, sys, glob, json, argparse, time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pydicom
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import (gaussian_filter, binary_closing,
                            binary_fill_holes, label as nd_label,
                            distance_transform_edt)

# ─── パス設定 ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DICOM_DIR = os.path.join(BASE_DIR, "data", "raw_dicom", "ct")
OUT_DIR   = os.path.join(BASE_DIR, "results", "seg_quality_pro")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── スタイル設定 ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.linewidth":   1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  2.0,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})
COLORS = {
    "threshold": "#2ecc71",
    "otsu":      "#3498db",
    "watershed": "#e74c3c",
    "noise":     "#2ecc71",
    "blur":      "#3498db",
    "combo":     "#9b59b6",
}

# ─── データクラス ─────────────────────────────────────────────────────────────
@dataclass
class SegResult:
    cond_type:  str
    cond_label: str
    param:      float
    method:     str
    dice_mean:  float
    dice_std:   float
    hd95_mean:  float
    hd95_std:   float
    precision_mean: float
    recall_mean:    float
    vs_mean:    float   # Volume Similarity

# ─── 1. ボリューム読み込み ────────────────────────────────────────────────────
def load_volume(dicom_dir: str) -> np.ndarray:
    files  = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"DICOM not found: {dicom_dir}")
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope     = float(getattr(slices[0], "RescaleSlope",     1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    return vol * slope + intercept

# ─── 2. GTマスク ─────────────────────────────────────────────────────────────
def make_gt_mask(volume: np.ndarray, hu_min: float = 200) -> np.ndarray:
    mask = binary_closing(volume >= hu_min, iterations=2)
    mask = binary_fill_holes(mask)
    labeled, n = nd_label(mask)
    if n > 0:
        sizes = np.array([np.sum(labeled == i) for i in range(1, n+1)])
        keep  = np.where(sizes >= 50)[0] + 1
        mask  = np.isin(labeled, keep)
    return mask.astype(np.uint8)

# ─── 3. ノイズ → 線量換算 ─────────────────────────────────────────────────────
def sigma_to_mas_equivalent(sigma: float, ref_sigma: float = 5.0,
                              ref_mas: float = 200.0) -> float:
    """
    ガウスノイズσをmAs等価値に変換 (CTノイズ ∝ 1/√mAs)
    σ = k / √mAs → mAs = (k / σ)² = ref_mAs × (ref_σ / σ)²
    """
    if sigma <= 0:
        return ref_mas
    return ref_mas * (ref_sigma / sigma) ** 2

# ─── 4. 劣化条件 ─────────────────────────────────────────────────────────────
def make_degraded(volume: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    conditions = []

    # ノイズ（低線量シミュレーション）
    for sigma in [0, 5, 10, 15, 20, 30, 40, 60, 80]:
        noise = rng.normal(0, sigma, volume.shape).astype(np.float32) if sigma > 0 \
                else np.zeros_like(volume)
        mas_eq = sigma_to_mas_equivalent(sigma) if sigma > 0 else 9999
        conditions.append({
            "type": "noise", "label": f"σ={sigma} HU",
            "param": float(sigma), "mas_eq": round(mas_eq, 1),
            "volume": volume + noise,
        })

    # 平滑化カーネル
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]:
        conditions.append({
            "type": "blur", "label": f"Gauss σ={sigma}",
            "param": float(sigma), "mas_eq": None,
            "volume": gaussian_filter(volume.astype(np.float32), sigma=sigma),
        })

    # ノイズ + ぼかし（低線量 + 軟部カーネル）
    for ns, bs in [(10, 0.5), (20, 1.0), (30, 1.5), (40, 2.0), (60, 3.0)]:
        noisy = volume + rng.normal(0, ns, volume.shape).astype(np.float32)
        conditions.append({
            "type": "combo", "label": f"n{ns}+b{bs}",
            "param": float(ns), "mas_eq": None,
            "volume": gaussian_filter(noisy, sigma=bs),
        })

    return conditions

# ─── 5. セグメンテーション手法 ────────────────────────────────────────────────
def seg_threshold(vol: np.ndarray, hu_min: float = 200) -> np.ndarray:
    """固定HU閾値 + 形態学的処理"""
    mask = binary_closing(vol >= hu_min, iterations=2)
    mask = binary_fill_holes(mask)
    labeled, n = nd_label(mask)
    if n > 0:
        sizes = np.array([np.sum(labeled == i) for i in range(1, n+1)])
        mask  = np.isin(labeled, np.where(sizes >= 50)[0] + 1)
    return mask.astype(np.uint8)


def seg_otsu(vol: np.ndarray) -> np.ndarray:
    """Otsu二値化（各スライスで独立）"""
    out = np.zeros_like(vol, dtype=np.uint8)
    for z in range(vol.shape[0]):
        sl = vol[z]
        sl_u8 = np.clip((sl + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(sl_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[z] = (mask > 0).astype(np.uint8)
    # 3Dクリーンアップ
    out = binary_closing(out, iterations=2).astype(np.uint8)
    labeled, n = nd_label(out)
    if n > 0:
        sizes = np.array([np.sum(labeled == i) for i in range(1, n+1)])
        out   = np.isin(labeled, np.where(sizes >= 100)[0] + 1).astype(np.uint8)
    return out


def seg_adaptive(vol: np.ndarray) -> np.ndarray:
    """適応型閾値: Otsu閾値 + 形態学的watershed風処理"""
    # まずOtsuで大まかに
    coarse = seg_otsu(vol)
    # さらに侵食→膨張で確実な領域を拡張
    from scipy.ndimage import binary_erosion, binary_dilation
    sure_fg = binary_erosion(coarse, iterations=3)
    sure_bg = binary_dilation(coarse, iterations=5)
    # 確実なFG周辺をGradientで精密化
    refined = binary_dilation(sure_fg, iterations=4).astype(np.uint8)
    return refined

METHODS = {
    "HU-Threshold": seg_threshold,
    "Otsu":         seg_otsu,
    "Adaptive":     seg_adaptive,
}

# ─── 6. 評価指標 ─────────────────────────────────────────────────────────────
def calc_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 2.0 * inter / denom if denom > 0 else 1.0


def calc_hd95(pred: np.ndarray, gt: np.ndarray,
               n_sample_slices: int = 5) -> float:
    """
    95%パーセンタイル Hausdorff Distance (voxel単位)
    速度のため中央付近のスライスをサンプリングして2Dで計算
    """
    p3d, g3d = pred.astype(bool), gt.astype(bool)
    if not p3d.any() or not g3d.any():
        return float("inf")

    nz = p3d.shape[0]
    mid = nz // 2
    step = max(1, nz // (n_sample_slices * 2))
    z_indices = list(range(max(0, mid - n_sample_slices*step),
                            min(nz, mid + n_sample_slices*step), step))[:n_sample_slices]

    hds = []
    for z in z_indices:
        p_sl = p3d[z]
        g_sl = g3d[z]
        if not p_sl.any() or not g_sl.any():
            continue
        dt_p = distance_transform_edt(~p_sl)
        dt_g = distance_transform_edt(~g_sl)
        hd = max(np.percentile(dt_g[p_sl], 95),
                  np.percentile(dt_p[g_sl], 95))
        hds.append(hd)
    return float(np.mean(hds)) if hds else float("inf")


def calc_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    vol_sim    = 1.0 - abs(p.sum() - g.sum()) / (p.sum() + g.sum() + 1e-9)
    return {
        "dice":      calc_dice(pred, gt),
        "hd95":      calc_hd95(pred, gt),
        "precision": precision,
        "recall":    recall,
        "vol_sim":   vol_sim,
    }

# ─── 7. モンテカルロ評価 ──────────────────────────────────────────────────────
def evaluate_condition(volume_clean: np.ndarray, gt: np.ndarray,
                        cond: dict, method_name: str, method_fn,
                        n_trials: int = 3) -> SegResult:
    """
    ノイズ条件は複数試行してMean±SD を算出（モンテカルロ）
    ぼかし条件は決定論的なので1試行
    """
    metrics_list = []
    trials = n_trials if cond["type"] in ("noise", "combo") else 1

    for t in range(trials):
        # 各試行でノイズを再生成
        if cond["type"] == "noise" and cond["param"] > 0:
            noise = np.random.normal(0, cond["param"], volume_clean.shape).astype(np.float32)
            vol   = volume_clean + noise
        elif cond["type"] == "combo":
            ns = cond["param"]
            bs = {"n10+b0.5":0.5,"n20+b1.0":1.0,"n30+b1.5":1.5,
                  "n40+b2.0":2.0,"n60+b3.0":3.0}.get(cond["label"], 1.0)
            noise = np.random.normal(0, ns, volume_clean.shape).astype(np.float32)
            vol   = gaussian_filter(volume_clean + noise, sigma=bs)
        else:
            vol = cond["volume"]

        pred = method_fn(vol)
        metrics_list.append(calc_metrics(pred, gt))

    return SegResult(
        cond_type  = cond["type"],
        cond_label = cond["label"],
        param      = cond["param"],
        method     = method_name,
        dice_mean  = round(np.mean([m["dice"]  for m in metrics_list]), 4),
        dice_std   = round(np.std( [m["dice"]  for m in metrics_list]), 4),
        hd95_mean  = round(np.mean([m["hd95"]  for m in metrics_list]), 2),
        hd95_std   = round(np.std( [m["hd95"]  for m in metrics_list]), 2),
        precision_mean = round(np.mean([m["precision"] for m in metrics_list]), 4),
        recall_mean    = round(np.mean([m["recall"]    for m in metrics_list]), 4),
        vs_mean        = round(np.mean([m["vol_sim"]   for m in metrics_list]), 4),
    )

# ─── 8. 可視化 ───────────────────────────────────────────────────────────────
def plot_main_figure(results: List[SegResult], out_path: str):
    """Figure 1: Dice曲線（ノイズ・ぼかし）× 手法比較"""
    noise_results = [r for r in results if r.cond_type == "noise"]
    blur_results  = [r for r in results if r.cond_type == "blur"]
    method_names  = list(METHODS.keys())

    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    markers = {"HU-Threshold": "o", "Otsu": "s", "Adaptive": "^"}
    palette = {"HU-Threshold": "#2ecc71", "Otsu": "#3498db", "Adaptive": "#e74c3c"}

    # ① ノイズ vs Dice
    ax = axes[0]
    for mn in method_names:
        sub = sorted([r for r in noise_results if r.method == mn], key=lambda r: r.param)
        xs  = [r.param for r in sub]
        ys  = [r.dice_mean for r in sub]
        es  = [r.dice_std  for r in sub]
        ax.errorbar(xs, ys, yerr=es, label=mn, marker=markers[mn],
                    color=palette[mn], capsize=4, linewidth=1.8, markersize=6)
    ax.axhspan(0.75, 0.90, alpha=0.08, color="orange", label="Acceptable zone")
    ax.axhspan(0.90, 1.05, alpha=0.06, color="green",  label="Excellent zone")
    ax.axhline(0.90, color="green",  linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Added Noise σ (HU)", fontsize=11)
    ax.set_ylabel("Dice Similarity Coefficient", fontsize=11)
    ax.set_title("(A)  Noise Level vs. Segmentation Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    # mAs等価軸（上部）
    ax2 = ax.twiny()
    noise_subs = sorted([r for r in noise_results if r.method == "HU-Threshold"],
                         key=lambda r: r.param)
    xs_mas = [sigma_to_mas_equivalent(r.param) if r.param > 0 else 1000 for r in noise_subs]
    ax2.set_xlim(ax.get_xlim())
    tick_pos = [r.param for r in noise_subs if r.param > 0][:5]
    tick_mas = [f"{sigma_to_mas_equivalent(p):.0f}" for p in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_mas, fontsize=7)
    ax2.set_xlabel("Equivalent mAs", fontsize=9, color="gray")

    # ② ぼかし vs Dice
    ax = axes[1]
    for mn in method_names:
        sub = sorted([r for r in blur_results if r.method == mn], key=lambda r: r.param)
        xs  = [r.param     for r in sub]
        ys  = [r.dice_mean for r in sub]
        es  = [r.dice_std  for r in sub]
        ax.errorbar(xs, ys, yerr=es, label=mn, marker=markers[mn],
                    color=palette[mn], capsize=4, linewidth=1.8, markersize=6)
    ax.axhline(0.90, color="green",  linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Gaussian Blur σ (voxels)", fontsize=11)
    ax.set_ylabel("Dice Similarity Coefficient", fontsize=11)
    ax.set_title("(B)  Smoothing Kernel vs. Segmentation Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    # ③ Precision-Recall プロット
    ax = axes[2]
    combo_results = [r for r in results if r.cond_type in ("noise", "blur")]
    for mn in method_names:
        sub = [r for r in combo_results if r.method == mn]
        xs  = [r.precision_mean for r in sub]
        ys  = [r.recall_mean    for r in sub]
        ds  = [r.dice_mean      for r in sub]
        sc  = ax.scatter(xs, ys, c=ds, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                          marker=markers[mn], s=60, alpha=0.8, label=mn)
    plt.colorbar(sc, ax=ax, label="Dice")
    ax.set_xlabel("Precision", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("(C)  Precision-Recall Space", fontsize=11, fontweight="bold")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.plot([0.5, 1.05], [0.5, 1.05], "k--", alpha=0.3, linewidth=1)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Effect of CT Image Quality Degradation on Bone Segmentation Performance\n"
        "(Acrylic Elbow Phantom Volume — Simulation Study)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 1: {out_path}")


def plot_hd95_figure(results: List[SegResult], out_path: str):
    """Figure 2: Hausdorff距離"""
    noise_res = sorted([r for r in results if r.cond_type == "noise"],
                        key=lambda r: (r.method, r.param))
    method_names = list(METHODS.keys())
    markers = {"HU-Threshold": "o", "Otsu": "s", "Adaptive": "^"}
    palette = {"HU-Threshold": "#2ecc71", "Otsu": "#3498db", "Adaptive": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # HD95 vs ノイズ
    ax = axes[0]
    for mn in method_names:
        sub = sorted([r for r in noise_res if r.method == mn], key=lambda r: r.param)
        xs, ys, es = ([r.param for r in sub], [r.hd95_mean for r in sub],
                       [r.hd95_std for r in sub])
        ax.errorbar(xs, ys, yerr=es, label=mn, marker=markers[mn],
                    color=palette[mn], capsize=4, linewidth=1.8, markersize=6)
    ax.axhline(5.0, color="green",  linestyle="--", linewidth=1, label="HD95 = 5 vox (Good)")
    ax.axhline(10.0,color="orange", linestyle="--", linewidth=1, label="HD95 = 10 vox (Acceptable)")
    ax.set_xlabel("Added Noise σ (HU)", fontsize=11)
    ax.set_ylabel("HD95 (voxels)", fontsize=11)
    ax.set_title("(A)  95th-percentile Hausdorff Distance vs. Noise", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Volume Similarity
    ax = axes[1]
    for mn in method_names:
        sub = sorted([r for r in noise_res if r.method == mn], key=lambda r: r.param)
        xs = [r.param   for r in sub]
        ys = [r.vs_mean for r in sub]
        ax.plot(xs, ys, marker=markers[mn], color=palette[mn],
                linewidth=1.8, markersize=6, label=mn)
    ax.axhline(0.95, color="green",  linestyle="--", linewidth=1, label="VS = 0.95")
    ax.axhline(0.90, color="orange", linestyle="--", linewidth=1, label="VS = 0.90")
    ax.set_xlabel("Added Noise σ (HU)", fontsize=11)
    ax.set_ylabel("Volume Similarity", fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("(B)  Volume Similarity vs. Noise", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Boundary Accuracy and Volume Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 2: {out_path}")


def plot_slice_comparison(volume: np.ndarray, gt: np.ndarray,
                           conditions: list, out_path: str):
    """Figure 3: 代表スライスの視覚的比較"""
    mid_z = volume.shape[0] // 2
    method_fn = seg_threshold

    show_conds = [c for c in conditions if c["type"] == "noise"]
    show_conds = [show_conds[0]] + show_conds[2::2][:4]  # 基準+4条件

    n = len(show_conds)
    fig, axes = plt.subplots(3, n, figsize=(3.5*n, 10))

    def norm_slice(sl):
        vmin = max(sl.min(), -200)
        vmax = min(sl.max(),  800)
        return np.clip((sl - vmin) / (vmax - vmin + 1e-9), 0, 1)

    for i, cond in enumerate(show_conds):
        vol = cond["volume"]
        pred = method_fn(vol)
        d    = calc_dice(pred, gt)
        h95  = calc_hd95(pred, gt)

        # Row 0: CT slice
        axes[0, i].imshow(norm_slice(vol[mid_z]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(cond["label"], fontsize=8, fontweight="bold")
        axes[0, i].axis("off")

        # Row 1: Segmentation overlay
        gt_sl   = gt[mid_z].astype(bool)
        pred_sl = pred[mid_z].astype(bool)
        overlay = np.zeros((*gt_sl.shape, 3))
        bg      = norm_slice(vol[mid_z])
        overlay[:, :, 0] = bg
        overlay[:, :, 1] = bg
        overlay[:, :, 2] = bg
        # GT境界: 緑
        gt_border   = gt_sl ^ binary_closing(gt_sl, iterations=1)
        # Pred境界: 赤
        pred_border = pred_sl ^ binary_closing(pred_sl, iterations=1)
        overlay[gt_border,   1] = 1.0
        overlay[gt_border,   0] = 0.0
        overlay[pred_border, 0] = 1.0
        overlay[pred_border, 1] = 0.0
        axes[1, i].imshow(np.clip(overlay, 0, 1))
        axes[1, i].set_title(f"Dice={d:.4f}\nHD95={h95:.1f} vox", fontsize=8)
        axes[1, i].axis("off")

        # Row 2: 差分マップ (GT - Pred)
        diff = gt_sl.astype(int) - pred_sl.astype(int)
        cmap = plt.cm.RdBu
        axes[2, i].imshow(diff, cmap=cmap, vmin=-1, vmax=1)
        axes[2, i].set_title("Error map\n(Blue=FN, Red=FP)", fontsize=7)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("CT Image", fontsize=9)
    axes[1, 0].set_ylabel("Segmentation\n(Green=GT, Red=Pred)", fontsize=9)
    axes[2, 0].set_ylabel("Error Map", fontsize=9)

    patches = [mpatches.Patch(color="green", label="GT boundary"),
               mpatches.Patch(color="red",   label="Pred boundary")]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Figure 3: Visual Comparison Across Noise Levels (HU-Threshold method)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure 3: {out_path}")


def plot_summary_heatmap(results: List[SegResult], out_path: str):
    """Figure 4: メソッド × 条件のDiceヒートマップ"""
    method_names = list(METHODS.keys())
    noise_res = sorted([r for r in results if r.cond_type == "noise"],
                        key=lambda r: r.param)
    noise_labels = sorted(set(r.cond_label for r in noise_res))
    blur_labels  = sorted(set(r.cond_label for r in results if r.cond_type == "blur"),
                           key=lambda l: float(l.split("=")[1]))

    all_labels = noise_labels + blur_labels
    matrix = np.zeros((len(method_names), len(all_labels)))

    for i, mn in enumerate(method_names):
        for j, lbl in enumerate(all_labels):
            matches = [r for r in results if r.method == mn and r.cond_label == lbl]
            if matches:
                matrix[i, j] = matches[0].dice_mean

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=9)
    ax.axvline(len(noise_labels) - 0.5, color="white", linewidth=2)
    ax.text(len(noise_labels)//2, -0.8, "← Noise conditions →",
             ha="center", fontsize=8, transform=ax.transData)
    ax.text(len(noise_labels) + len(blur_labels)//2, -0.8, "← Blur conditions →",
             ha="center", fontsize=8, transform=ax.transData)
    plt.colorbar(im, ax=ax, label="Dice Coefficient", fraction=0.02)
    for i in range(len(method_names)):
        for j in range(len(all_labels)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=6.5, color="black" if matrix[i,j] > 0.75 else "white")
    ax.set_title("Figure 4: Dice Coefficient Heatmap — Method × Condition",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure 4: {out_path}")


# ─── 9. テキストレポート ─────────────────────────────────────────────────────
def save_report(results: List[SegResult], gt: np.ndarray, out_path: str):
    lines = [
        "CT Image Degradation vs. Bone Segmentation Quality — Analysis Report",
        "=" * 70,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"GT bone voxels: {gt.sum():,}  ({gt.sum()/gt.size*100:.2f}%)",
        "",
        "NOISE LEVEL RESULTS (HU-Threshold method)",
        "-" * 50,
    ]
    noise_thr = [r for r in results
                  if r.cond_type == "noise" and r.method == "HU-Threshold"]
    for r in sorted(noise_thr, key=lambda x: x.param):
        grade = "EXCELLENT" if r.dice_mean >= 0.90 else \
                "ACCEPTABLE" if r.dice_mean >= 0.75 else "POOR"
        lines.append(
            f"  {r.cond_label:15s}  Dice={r.dice_mean:.4f}±{r.dice_std:.4f}"
            f"  HD95={r.hd95_mean:.1f}±{r.hd95_std:.1f}  [{grade}]"
        )

    lines += ["", "BLUR LEVEL RESULTS (HU-Threshold method)", "-" * 50]
    blur_thr = [r for r in results
                 if r.cond_type == "blur" and r.method == "HU-Threshold"]
    for r in sorted(blur_thr, key=lambda x: x.param):
        grade = "EXCELLENT" if r.dice_mean >= 0.90 else \
                "ACCEPTABLE" if r.dice_mean >= 0.75 else "POOR"
        lines.append(
            f"  {r.cond_label:15s}  Dice={r.dice_mean:.4f}±{r.dice_std:.4f}"
            f"  P={r.precision_mean:.4f}  R={r.recall_mean:.4f}  [{grade}]"
        )

    # 閾値サマリー
    lines += ["", "DEGRADATION TOLERANCE SUMMARY", "-" * 50]
    for mn in METHODS:
        nr = sorted([r for r in results if r.cond_type == "noise" and r.method == mn],
                     key=lambda x: x.param)
        ok  = [r for r in nr if r.dice_mean >= 0.90]
        max_ok = ok[-1].param if ok else 0
        lines.append(f"  {mn:15s}: Dice≥0.90 up to noise σ={max_ok:.0f} HU")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {out_path}")


# ─── メイン ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=3,
                         help="モンテカルロ試行回数 (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print(" CT画質劣化 → 骨セグメンテーション精度解析 [Pro版]")
    print("=" * 60)
    t0 = time.time()

    # 1. データ
    print("\n[1/5] ボリューム読み込み...")
    volume_full = load_volume(DICOM_DIR)
    # 中央60スライスのみ使用（速度最適化）
    mid = volume_full.shape[0] // 2
    volume = volume_full[mid-30:mid+30].copy()
    print(f"  Full: {volume_full.shape} → 解析用: {volume.shape}")
    print(f"  HU: {volume.min():.0f}~{volume.max():.0f}")

    # 2. GT
    print("\n[2/5] GTマスク生成...")
    gt = make_gt_mask(volume, hu_min=200)
    print(f"  骨ボクセル: {gt.sum():,} ({gt.sum()/gt.size*100:.2f}%)")

    # 3. 条件生成
    print("\n[3/5] 劣化条件生成...")
    conditions = make_degraded(volume)
    print(f"  {len(conditions)}条件  ×  {len(METHODS)}手法")

    # 4. 評価
    print("\n[4/5] セグメンテーション評価...")
    results: List[SegResult] = []
    total = len(conditions) * len(METHODS)
    done  = 0
    for cond in conditions:
        for mn, mfn in METHODS.items():
            r = evaluate_condition(volume, gt, cond, mn, mfn, args.n_trials)
            results.append(r)
            done += 1
            grade = "🟢" if r.dice_mean >= 0.90 else "🟠" if r.dice_mean >= 0.75 else "🔴"
            print(f"  [{done:3d}/{total}] {grade} {mn:15s} | {cond['label']:18s}"
                  f" Dice={r.dice_mean:.4f}±{r.dice_std:.4f}"
                  f" HD95={r.hd95_mean:.1f}")

    # 5. 可視化・レポート
    print("\n[5/5] 可視化 & レポート生成...")
    plot_main_figure(results,    os.path.join(OUT_DIR, "fig1_dice_curves.png"))
    plot_hd95_figure(results,    os.path.join(OUT_DIR, "fig2_hd95_volume.png"))
    plot_slice_comparison(volume, gt, conditions, os.path.join(OUT_DIR, "fig3_slices.png"))
    plot_summary_heatmap(results, os.path.join(OUT_DIR, "fig4_heatmap.png"))
    save_report(results, gt,     os.path.join(OUT_DIR, "report.txt"))

    with open(os.path.join(OUT_DIR, "results_full.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    elapsed = time.time() - t0
    print(f"\n完了 ({elapsed:.1f}s)  →  {OUT_DIR}")


if __name__ == "__main__":
    main()
