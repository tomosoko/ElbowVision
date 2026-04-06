"""
ElbowVision 類似度マッチングによる屈曲角推定（本研究コアアルゴリズム）

患者の伸展CTから計算的に曲げたDRRと実LAT X線を比較し、
最も一致する曲げ角度 = 実際の屈曲角度として推定する。

直接回帰（プレ研究）との違い:
  - 直接回帰: DRR→モデル→角度値（単一CT訓練、他患者に汎化するか要検証）
  - 類似度マッチング: 各患者CT→DRR群→患者自身の実X線と比較（患者固有、汎化問題を回避）

コアパイプライン:
  伸展CT → bend_volume(各角度) → DRR群
                ↕ 類似度比較 (NCC / SSIM / Edge-NCC)
           実LAT X線（実際の屈曲位）

使い方:
  # 単一患者の評価
  python scripts/similarity_matching.py \
    --ct_dir data/raw_dicom/ct_volume/... \
    --xray   data/real_xray/images/008_LAT.png \
    --out_dir results/similarity_matching/

  # バッチ評価（患者リストCSV）
  python scripts/similarity_matching.py \
    --patient_list data/patients_with_xray.csv \
    --out_dir results/similarity_matching/

患者リストCSV形式（バッチ時）:
  patient_id, ct_dir, xray_path, gt_angle_deg, laterality, series_num, hu_min, hu_max
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    rotate_volume_and_landmarks,
    generate_drr,
)

# ── 設定 ──────────────────────────────────────────────────────────────────────

TARGET_SIZE = 256
SID_MM      = 1000.0
ANGLE_STEP  = 5.0           # 粗探索の角度ステップ（°）
ANGLE_FINE  = 1.0           # 精密探索のステップ（°）
ANGLE_MIN   = 60.0          # 探索範囲最小
ANGLE_MAX   = 180.0         # 探索範囲最大

def _generate_drr_at_angle(
    volume: np.ndarray,
    landmarks: dict,
    base_flex: float,
    voxel_mm: float,
    angle_deg: float,
) -> tuple[float, np.ndarray]:
    """指定角度でDRRを生成して返す（シーケンシャル実行）"""
    rot_vol, _ = rotate_volume_and_landmarks(
        volume, landmarks,
        forearm_rotation_deg=0.0,
        flexion_deg=angle_deg,
        base_flexion=base_flex,
        valgus_deg=0.0,
    )
    drr = generate_drr(rot_vol, axis="LAT", sid_mm=SID_MM, voxel_mm=voxel_mm)
    return angle_deg, drr


# ── 画像前処理 ────────────────────────────────────────────────────────────────

def crop_to_bone(img: np.ndarray, margin: float = 0.15, dark_thresh: int = 15) -> np.ndarray:
    """
    暗画素（背景）を除去して骨領域をクロップ。
    実X線の大きな黒背景に有効（実X線の背景はほぼ0、腕部は > 15）。
    dark_thresh: これ以下の画素値を背景とみなす（デフォルト=15）
    """
    mask = img > dark_thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img  # クロップ失敗時は元画像
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # 面積チェック: 元画像の70%以上ならクロップ不要（DRRなど背景が少ない場合）
    h, w = img.shape
    crop_ratio = ((rmax - rmin) * (cmax - cmin)) / (h * w)
    if crop_ratio > 0.7:
        return img
    # マージン追加
    pad_r = int((rmax - rmin) * margin)
    pad_c = int((cmax - cmin) * margin)
    rmin = max(0, rmin - pad_r)
    rmax = min(h - 1, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w - 1, cmax + pad_c)
    return img[rmin:rmax+1, cmin:cmax+1]


def preprocess_image(img: np.ndarray, size: int = 256,
                     apply_rot270: bool = False,
                     auto_crop: bool = False) -> np.ndarray:
    """
    類似度比較用の前処理:
    1. グレースケール変換
    2. 骨領域自動クロップ（auto_crop=True 時、実X線の黒背景除去）
    3. rot270（ConvNeXt推論用。類似度マッチングでは通常 False）
    4. リサイズ
    5. CLAHE（コントラスト均一化）
    6. 正規化 [0, 1]

    auto_crop:    実X線に大きな黒背景がある場合 True（DRRには不要）
    apply_rot270: ConvNeXt推論用。類似度マッチングでは False
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if auto_crop:
        img = crop_to_bone(img)
    if apply_rot270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270° = CCW90°
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.float32) / 255.0


def extract_edges(img_norm: np.ndarray) -> np.ndarray:
    """Canny エッジ抽出（骨輪郭に特化した類似度用）"""
    img_u8 = (img_norm * 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, threshold1=30, threshold2=90)
    return edges.astype(np.float32) / 255.0


# ── 類似度メトリクス ──────────────────────────────────────────────────────────

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Cross-Correlation（-1 〜 +1、高いほど類似）"""
    a_flat = a.ravel() - a.mean()
    b_flat = b.ravel() - b.mean()
    denom  = np.sqrt((a_flat**2).sum() * (b_flat**2).sum()) + 1e-8
    return float(np.dot(a_flat, b_flat) / denom)


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM（0 〜 1、高いほど類似）"""
    return float(ssim(a, b, data_range=1.0))


def nmi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """
    Normalized Mutual Information（1.0 〜 2.0、高いほど類似）

    NMI = (H(A) + H(B)) / H(A,B)

    NCC が線形強度変換に不変なのに対し、NMI は任意の単調変換に不変。
    DRR（散乱線なし）と実X線（散乱線・軟部組織あり）の非線形強度関係に有効。

    実X線評価での観察（patient008, GT=90°）:
      NMI ピーク: ~85° (-5° バイアス、edge_NCC と同方向)
      NCC ピーク: ~95° (+5° バイアス)
      → combined_nmi: (NCC + NMI) / 2 ≈ 90°（combined と同等精度）

    bins: 結合ヒストグラムのビン数（64で十分な精度と速度のバランス）
    """
    a_flat = (np.clip(a, 0, 1) * (bins - 1)).ravel().astype(np.int32)
    b_flat = (np.clip(b, 0, 1) * (bins - 1)).ravel().astype(np.int32)

    # 2D 結合ヒストグラム
    hist_2d, _, _ = np.histogram2d(a_flat, b_flat, bins=bins,
                                    range=[[0, bins], [0, bins]])
    pxy = hist_2d / (hist_2d.sum() + 1e-12)

    # 周辺分布
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # エントロピー計算（ゼロ除算回避）
    def _entropy(p: np.ndarray) -> float:
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p)))

    hx  = _entropy(px)
    hy  = _entropy(py)
    hxy = _entropy(pxy.ravel())

    if hxy < 1e-8:
        return 1.0
    return float((hx + hy) / hxy)


def compute_similarity(drr_norm: np.ndarray,
                        xray_norm: np.ndarray) -> dict[str, float]:
    """4種の類似度を計算して返す"""
    drr_edge  = extract_edges(drr_norm)
    xray_edge = extract_edges(xray_norm)
    return {
        "ncc":      ncc(drr_norm, xray_norm),
        "ssim":     ssim_score(drr_norm, xray_norm),
        "edge_ncc": ncc(drr_edge, xray_edge),
        "nmi":      nmi(drr_norm, xray_norm),
    }


# ── 類似度マッチング ──────────────────────────────────────────────────────────

def _parabolic_peak(scores_dict: dict[float, dict[str, float]], metric_key: str) -> float:
    """
    NCCスコアカーブを放物線フィットしてサブ1°精度のピーク角度を推定。

    ピーク前後の3点で numpy.polyfit(2次) を使い、等間隔でない場合も対応。
    ピークが端点または隣接点が存在しない場合はピーク角度をそのまま返す。
    """
    sorted_angles = sorted(scores_dict)
    peak_angle = max(sorted_angles, key=lambda a: scores_dict[a][metric_key])
    idx = sorted_angles.index(peak_angle)

    if idx == 0 or idx == len(sorted_angles) - 1:
        return peak_angle

    a_prev = sorted_angles[idx - 1]
    a_next = sorted_angles[idx + 1]

    x = np.array([a_prev, peak_angle, a_next])
    y = np.array([scores_dict[a_prev][metric_key],
                  scores_dict[peak_angle][metric_key],
                  scores_dict[a_next][metric_key]])

    # 2次多項式フィット
    coeffs = np.polyfit(x, y, 2)  # [a, b, c]
    a_coef = coeffs[0]
    if abs(a_coef) < 1e-12 or a_coef > 0:  # 下に凸 or 水平 → フィット失敗
        return peak_angle

    # 頂点 x* = -b / (2a)
    x_peak = -coeffs[1] / (2 * a_coef)

    # ピーク前後の範囲外に出たらクリップ
    x_peak = float(np.clip(x_peak, a_prev, a_next))
    return x_peak


class MatchResult(NamedTuple):
    best_angle:   float
    best_metric:  str
    scores:       dict[float, dict[str, float]]  # {angle: {metric: score}}
    drr_at_best:  np.ndarray
    peak_ncc:     float = 0.0   # ベスト角度のNCCスコア（1.0に近いほど高信頼）
    sharpness:    float = 0.0   # ピーク鋭敏度 = (peak - mean) / std（大きいほど高信頼）


def match_angle(
    volume: np.ndarray,
    landmarks: dict,
    lat: str,
    voxel_mm: float,
    base_flexion: float,
    xray_img: np.ndarray,
    metric: str = "combined",
    coarse_step: float = ANGLE_STEP,
    fine_step: float = ANGLE_FINE,
    fine_range: float = 10.0,
) -> MatchResult:
    """
    2段階探索:
    1. 粗探索（coarse_step 刻み、全範囲）
    2. 精密探索（fine_step 刻み、粗探索最良±fine_range 内）

    Args:
        metric: 使用する類似度指標 ("ncc" | "ssim" | "edge_ncc" | "combined" | "nmi")
                "combined": NCCとedge_nccのピーク角の平均（バイアス打ち消し）
                  NCC: +5°バイアス（95°推定傾向）
                  edge_ncc: -5°バイアス（85°推定傾向）
                  combined: 平均90°（検証済み）
                "nmi": Normalized Mutual Information（非線形強度変換に不変）
                  実X線 vs DRR のドメインギャップ対策に有効（実験的）
                "combined_nmi": NCC(+5°バイアス)とNMI(-5°バイアス)の平均
                  combinedと同等精度（実X線評価で確認済み）、NMIのモノトーン不変性を活用
    """
    xray_norm = preprocess_image(xray_img, apply_rot270=False, auto_crop=True)
    all_scores: dict[float, dict[str, float]] = {}

    _primary = "ncc" if metric == "combined" else metric

    def _run_angle(angle: float) -> tuple[float, np.ndarray]:
        return _generate_drr_at_angle(volume, landmarks, base_flexion, voxel_mm, angle)

    # ── 粗探索 ──────────────────────────────────────────────────────────────
    coarse_angles = np.arange(ANGLE_MIN, ANGLE_MAX + coarse_step, coarse_step).tolist()
    print(f"    粗探索: {len(coarse_angles)} 角度 (step={coarse_step}°) ...")

    for i, angle in enumerate(coarse_angles):
        _, drr = _run_angle(angle)
        drr_norm = preprocess_image(drr)
        scores = compute_similarity(drr_norm, xray_norm)
        all_scores[angle] = scores
        print(f"      {i+1}/{len(coarse_angles)}: {angle:.0f}° → {_primary}={scores[_primary]:.4f}", end="\r")

    print()
    coarse_best = max(coarse_angles, key=lambda a: all_scores[a][_primary])
    print(f"    粗探索最良: {coarse_best:.0f}° ({_primary}={all_scores[coarse_best][_primary]:.4f})")

    # ── 精密探索 ─────────────────────────────────────────────────────────────
    fine_min    = max(ANGLE_MIN, coarse_best - fine_range)
    fine_max    = min(ANGLE_MAX, coarse_best + fine_range)
    fine_angles = np.arange(fine_min, fine_max + fine_step, fine_step).tolist()
    # 粗探索済みは除外
    fine_angles = [a for a in fine_angles if a not in all_scores]

    if fine_angles:
        print(f"    精密探索: {len(fine_angles)} 角度 (step={fine_step}°) ...")
        for i, angle in enumerate(fine_angles):
            _, drr = _run_angle(angle)
            drr_norm = preprocess_image(drr)
            scores = compute_similarity(drr_norm, xray_norm)
            all_scores[angle] = scores
            print(f"      {i+1}/{len(fine_angles)}: {angle:.1f}° → {_primary}={scores[_primary]:.4f}", end="\r")
        print()

    if metric == "combined":
        best_ncc_int  = max(all_scores, key=lambda a: all_scores[a]["ncc"])
        best_encc_int = max(all_scores, key=lambda a: all_scores[a]["edge_ncc"])
        # edge_nccピークがfine探索済み範囲外であれば追加探索
        extra_min = max(ANGLE_MIN, best_encc_int - fine_range)
        extra_max = min(ANGLE_MAX, best_encc_int + fine_range)
        extra_angles = [a for a in np.arange(extra_min, extra_max + fine_step, fine_step).tolist()
                        if a not in all_scores]
        if extra_angles:
            print(f"    edge_ncc追加精密探索: {len(extra_angles)}角度 ({best_encc_int:.0f}°周辺) ...")
            for angle in extra_angles:
                _, drr = _run_angle(angle)
                drr_norm = preprocess_image(drr)
                all_scores[angle] = compute_similarity(drr_norm, xray_norm)
        # combined metricは整数ピーク（±5°バイアス相殺を維持）
        best_ncc  = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
        best_encc = float(max(all_scores, key=lambda a: all_scores[a]["edge_ncc"]))
        best_angle = (best_ncc + best_encc) / 2.0
        print(f"    Combined: ncc={best_ncc:.1f}° + edge_ncc={best_encc:.1f}° → mean={best_angle:.1f}°")
    else:
        best_angle = _parabolic_peak(all_scores, metric)

    # ベスト角度のDRRを再生成（可視化用）
    closest = min(all_scores.keys(), key=lambda a: abs(a - best_angle))
    _, best_drr = _run_angle(closest)

    # 信頼度指標計算
    ncc_vals   = np.array([all_scores[a]["ncc"] for a in all_scores])
    peak_ncc   = float(all_scores[closest]["ncc"])
    sharpness  = float((peak_ncc - ncc_vals.mean()) / (ncc_vals.std() + 1e-8))

    return MatchResult(
        best_angle=best_angle,
        best_metric=metric,
        scores=all_scores,
        drr_at_best=best_drr,
        peak_ncc=peak_ncc,
        sharpness=sharpness,
    )


# ── DRRライブラリ（キャッシュ）使用マッチング ────────────────────────────────────

def load_drr_library(library_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    build_drr_library.py で生成した .npz キャッシュを読み込む。
    Returns: (angles, drrs, meta)
      angles: float32 (N,)
      drrs:   uint8   (N, H, W) — CLAHE適用済み
      meta:   dict
    """
    import json as _json
    data = np.load(library_path, allow_pickle=False)
    angles = data["angles"]                  # float32 (N,)
    drrs   = data["drrs"]                    # uint8   (N, H, W)
    meta   = _json.loads(str(data["meta"]))
    return angles, drrs, meta


class DRRLibraryCache:
    """
    DRRライブラリをメモリに保持し、複数回のマッチングで再読み込みを防ぐ。

    バッチ処理（複数患者・複数X線）での使用推奨:
      cache = DRRLibraryCache("data/drr_library/patient008.npz")
      for xray in xray_list:
          result = cache.match(xray)

    同一ライブラリに対する繰り返しコールで I/O が0になる（初回ロード後は即時）。
    """

    def __init__(self, library_path: str) -> None:
        self.library_path = library_path
        self._angles_arr: np.ndarray | None = None
        self._drrs: np.ndarray | None = None
        self._meta: dict | None = None
        self._angle_to_drr: dict[float, np.ndarray] | None = None

    def _ensure_loaded(self) -> None:
        if self._angle_to_drr is not None:
            return
        t0 = time.time()
        self._angles_arr, self._drrs, self._meta = load_drr_library(self.library_path)
        self._angle_to_drr = {
            float(self._angles_arr[i]): self._drrs[i]
            for i in range(len(self._angles_arr))
        }
        print(f"DRRライブラリロード: {Path(self.library_path).name} "
              f"({len(self._angle_to_drr)}角度, {time.time()-t0:.2f}s)")

    @property
    def meta(self) -> dict:
        self._ensure_loaded()
        return self._meta

    def match(
        self,
        xray_img: np.ndarray,
        metric: str = "combined",
        coarse_step: float = ANGLE_STEP,
        fine_range: float = 10.0,
    ) -> "MatchResult":
        """プリロード済みライブラリで類似度マッチングを実行"""
        self._ensure_loaded()
        return match_angle_from_library(
            self.library_path,
            xray_img,
            metric=metric,
            coarse_step=coarse_step,
            fine_range=fine_range,
            _preloaded=self._angle_to_drr,
        )


def match_angle_from_library(
    library_path: str,
    xray_img: np.ndarray,
    metric: str = "combined",
    coarse_step: float = ANGLE_STEP,
    fine_range: float = 10.0,
    _preloaded: "dict[float, np.ndarray] | None" = None,
) -> MatchResult:
    """
    事前構築DRRライブラリを使った類似度マッチング（CT生成不要）。
    DRR生成をスキップするため 41s → ~1s に高速化。

    非キャッシュ版と同じ2段階探索:
      1. coarse_step 刻みでNCC粗探索
      2. NCC最良値 ±fine_range 内でedge_ncc精密評価（combined用）

    Args:
        library_path: build_drr_library.py で生成した .npz ファイルパス
        xray_img:     実X線グレースケール画像 (numpy)
        metric:       "ncc" | "edge_ncc" | "combined" | "nmi"
        coarse_step:  粗探索ステップ（°）
        fine_range:   NCC最良値からの精密探索範囲（±°）
    """
    t0 = time.time()
    if _preloaded is not None:
        # DRRLibraryCacheからプリロード済みデータを受け取った場合
        angle_to_drr = _preloaded
        angles_arr_list = sorted(angle_to_drr.keys())
        lib_step = round(angles_arr_list[1] - angles_arr_list[0], 4) if len(angles_arr_list) > 1 else 1.0
        angle_min = angles_arr_list[0]
        angle_max = angles_arr_list[-1]
    else:
        print(f"DRRライブラリ読み込み: {Path(library_path).name}")
        angles_arr, drrs, meta = load_drr_library(library_path)
        lib_step   = float(meta["angle_step"])
        angle_min  = float(meta["angle_min"])
        angle_max  = float(meta["angle_max"])
        angle_to_drr = {float(angles_arr[i]): drrs[i] for i in range(len(angles_arr))}
        print(f"  {len(angle_to_drr)}角度 ({angle_min:.0f}°〜{angle_max:.0f}°, step={lib_step:.1f}°) "
              f"— 読込{time.time()-t0:.2f}s")

    xray_norm = preprocess_image(xray_img, apply_rot270=False, auto_crop=True)
    xray_edge = extract_edges(xray_norm)
    all_scores: dict[float, dict[str, float]] = {}
    _primary = "ncc" if metric in ("combined", "combined_nmi") else metric

    def _score(angle: float) -> dict[str, float]:
        nearest  = min(angle_to_drr.keys(), key=lambda a: abs(a - angle))
        drr_norm = angle_to_drr[nearest].astype(np.float32) / 255.0
        drr_edge = extract_edges(drr_norm)
        return {
            "ncc":      ncc(drr_norm, xray_norm),
            "edge_ncc": ncc(drr_edge, xray_edge),
            "nmi":      nmi(drr_norm, xray_norm),
        }

    # ── 粗探索 ──────────────────────────────────────────────────────────────
    coarse_angles = np.arange(angle_min, angle_max + coarse_step, coarse_step).tolist()
    print(f"  粗探索: {len(coarse_angles)}角度 (step={coarse_step}°) ...")
    t1 = time.time()
    for i, a in enumerate(coarse_angles):
        all_scores[a] = _score(a)
        print(f"\r  {i+1}/{len(coarse_angles)}: {a:.0f}° → {_primary}={all_scores[a][_primary]:.4f}",
              end="", flush=True)
    coarse_best = max(coarse_angles, key=lambda a: all_scores[a][_primary])
    print(f"\n  粗探索最良: {coarse_best:.0f}° ({_primary}={all_scores[coarse_best][_primary]:.4f})")

    # ── 精密探索（ライブラリ解像度 step=lib_step で fine_range 内）────────────
    fine_min = max(angle_min, coarse_best - fine_range)
    fine_max = min(angle_max, coarse_best + fine_range)
    fine_angles = [a for a in np.arange(fine_min, fine_max + lib_step, lib_step).tolist()
                   if round(a, 4) not in {round(k, 4) for k in all_scores}]
    if fine_angles:
        print(f"  精密探索: {len(fine_angles)}角度 (step={lib_step}°) ...")
        for i, a in enumerate(fine_angles):
            all_scores[a] = _score(a)
            print(f"\r  {i+1}/{len(fine_angles)}: {a:.1f}° → {_primary}={all_scores[a][_primary]:.4f}",
                  end="", flush=True)
        print()
    print(f"  計算完了: {time.time()-t1:.2f}s")

    if metric == "combined":
        # combined metricは整数ピーク（±5°バイアス相殺を維持するため補間しない）
        # edge_nccピーク周辺が NCC fine探索範囲外にある場合は追加精密探索
        coarse_best_encc = max(coarse_angles, key=lambda a: all_scores[a]["edge_ncc"])
        if abs(coarse_best_encc - coarse_best) > coarse_step:
            e_min = max(angle_min, coarse_best_encc - fine_range)
            e_max = min(angle_max, coarse_best_encc + fine_range)
            extra = [a for a in np.arange(e_min, e_max + lib_step, lib_step).tolist()
                     if round(a, 4) not in {round(k, 4) for k in all_scores}]
            if extra:
                print(f"  edge_ncc追加精密探索: {len(extra)}角度 ({coarse_best_encc:.0f}°周辺) ...")
                for a in extra:
                    all_scores[a] = _score(a)
        best_ncc_raw  = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
        best_encc_raw = float(max(all_scores, key=lambda a: all_scores[a]["edge_ncc"]))
        best_angle = (best_ncc_raw + best_encc_raw) / 2.0
        print(f"  Combined: ncc={best_ncc_raw:.1f}° + edge_ncc={best_encc_raw:.1f}° → mean={best_angle:.1f}°")
    elif metric == "combined_nmi":
        # NCC(+5°バイアス) + NMI(-5°バイアス) の平均（combined と同等精度）
        coarse_best_nmi = max(coarse_angles, key=lambda a: all_scores[a]["nmi"])
        if abs(coarse_best_nmi - coarse_best) > coarse_step:
            n_min = max(angle_min, coarse_best_nmi - fine_range)
            n_max = min(angle_max, coarse_best_nmi + fine_range)
            extra = [a for a in np.arange(n_min, n_max + lib_step, lib_step).tolist()
                     if round(a, 4) not in {round(k, 4) for k in all_scores}]
            if extra:
                for a in extra:
                    all_scores[a] = _score(a)
        best_ncc_raw = float(max(all_scores, key=lambda a: all_scores[a]["ncc"]))
        best_nmi_raw = float(max(all_scores, key=lambda a: all_scores[a]["nmi"]))
        best_angle = (best_ncc_raw + best_nmi_raw) / 2.0
        print(f"  Combined_NMI: ncc={best_ncc_raw:.1f}° + nmi={best_nmi_raw:.1f}° → mean={best_angle:.1f}°")
    else:
        # 単一metricは放物線補間でサブ1°精度
        best_angle = _parabolic_peak(all_scores, metric)

    nearest  = min(angle_to_drr.keys(), key=lambda a: abs(a - best_angle))
    best_drr = angle_to_drr[nearest]

    # 信頼度指標計算
    # nearest は angle_to_drr のキー（all_scores にない場合に備えて all_scores のキーで近傍を探す）
    nearest_scored = min(all_scores.keys(), key=lambda a: abs(a - best_angle))
    ncc_vals  = np.array([all_scores[a]["ncc"] for a in all_scores])
    peak_ncc  = float(all_scores[nearest_scored]["ncc"])
    sharpness = float((peak_ncc - ncc_vals.mean()) / (ncc_vals.std() + 1e-8))

    return MatchResult(
        best_angle=best_angle,
        best_metric=metric,
        scores=all_scores,
        drr_at_best=best_drr,
        peak_ncc=peak_ncc,
        sharpness=sharpness,
    )


# ── 可視化 ────────────────────────────────────────────────────────────────────

def save_result_figure(
    result: MatchResult,
    xray_img: np.ndarray,
    gt_angle: float | None,
    out_path: str,
    patient_id: str = "",
) -> None:
    """類似度曲線 + 最良DRR vs 実X線の比較図を保存"""
    metric = result.best_metric
    angles = sorted(result.scores)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 類似度曲線
    ax0 = axes[0]
    if metric == "combined":
        # NCC と edge_ncc の両曲線を重ねて表示
        ncc_vals  = [result.scores[a]["ncc"]      for a in angles]
        encc_vals = [result.scores[a]["edge_ncc"] for a in angles]
        # 正規化して同一スケールに
        def _norm(v):
            mn, mx = min(v), max(v)
            return [(x - mn) / (mx - mn + 1e-8) for x in v]
        ax0.plot(angles, _norm(ncc_vals),  "b-o", markersize=3, linewidth=1.2, label="NCC (norm)")
        ax0.plot(angles, _norm(encc_vals), "c-s", markersize=3, linewidth=1.0, label="edge_ncc (norm)")
        ax0.set_ylabel("Normalized Score")
        ax0.set_title("Similarity Curve\n(combined = NCC+edge_ncc)")
    else:
        values = [result.scores[a][metric] for a in angles]
        ax0.plot(angles, values, "b-o", markersize=3, linewidth=1.2)
        ax0.set_ylabel(metric.upper())
        ax0.set_title(f"Similarity Curve\n({metric})")

    ax0.axvline(result.best_angle, color="red", linewidth=1.5,
                label=f"Pred: {result.best_angle:.1f}°")
    if gt_angle is not None:
        ax0.axvline(gt_angle, color="green", linewidth=1.5, linestyle="--",
                    label=f"GT: {gt_angle:.1f}°")
    ax0.set_xlabel("Flexion Angle [°]")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # 最良マッチDRR
    ax1 = axes[1]
    ax1.imshow(result.drr_at_best, cmap="gray")
    ax1.set_title(f"Best Match DRR\n({result.best_angle:.1f}°)")
    ax1.axis("off")

    # 実X線（自動クロップして表示）
    ax2 = axes[2]
    xray_gray = xray_img if xray_img.ndim == 2 else cv2.cvtColor(xray_img, cv2.COLOR_BGR2GRAY)
    xray_gray = crop_to_bone(xray_gray)
    xray_gray = cv2.resize(xray_gray, (256, 256))
    ax2.imshow(xray_gray, cmap="gray")
    gt_str = f" (GT={gt_angle:.1f}°)" if gt_angle is not None else ""
    ax2.set_title(f"Real X-ray{gt_str}")
    ax2.axis("off")

    title = f"Similarity Matching — {patient_id}" if patient_id else "Similarity Matching"
    conf_str = f"  peak_ncc={result.peak_ncc:.4f}  sharpness={result.sharpness:.2f}"
    if gt_angle is not None:
        err = abs(result.best_angle - gt_angle)
        title += f"\nPred={result.best_angle:.1f}°  GT={gt_angle:.1f}°  Error={err:.1f}°{conf_str}"
    else:
        title += f"\nPred={result.best_angle:.1f}°{conf_str}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    図保存: {out_path}")


# ── メイン（単一患者）────────────────────────────────────────────────────────

def run_single(
    ct_dir: str,
    xray_path: str,
    out_dir: str,
    laterality: str | None = None,
    series_num: int | None = None,
    hu_min: float = -400.0,
    hu_max: float = 1500.0,
    gt_angle: float | None = None,
    metric: str = "combined",
    patient_id: str = "patient",
    base_flexion: float = 180.0,
    library_path: str | None = None,
) -> MatchResult:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"実X線読み込み: {xray_path}")
    xray_img = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)
    if xray_img is None:
        raise FileNotFoundError(f"X線画像が読み込めません: {xray_path}")

    t0 = time.time()

    if library_path and Path(library_path).exists():
        # ── ライブラリキャッシュから高速マッチング ──────────────────────────
        print(f"類似度マッチング開始 (キャッシュ使用, metric={metric}) ...")
        result = match_angle_from_library(library_path, xray_img, metric=metric)
    else:
        # ── オンザフライDRR生成（従来モード）────────────────────────────────
        if library_path:
            print(f"  [警告] ライブラリが見つかりません: {library_path} → オンザフライ生成にフォールバック")
        print(f"CT読み込み: {ct_dir}")
        volume, _, lat, voxel_mm = load_ct_volume(
            ct_dir, laterality=laterality, series_num=series_num,
            hu_min=hu_min, hu_max=hu_max, target_size=TARGET_SIZE,
        )
        landmarks = auto_detect_landmarks(volume, laterality=lat)
        print(f"類似度マッチング開始 (metric={metric}) ...")
        result = match_angle(
            volume, landmarks, lat, voxel_mm, base_flexion,
            xray_img, metric=metric,
        )

    elapsed = time.time() - t0

    print(f"\n【結果】")
    print(f"  推定角度 : {result.best_angle:.1f}°")
    if gt_angle is not None:
        err = abs(result.best_angle - gt_angle)
        print(f"  GT角度   : {gt_angle:.1f}°")
        print(f"  誤差     : {err:.1f}°")
    print(f"  処理時間 : {elapsed:.1f}s")
    print(f"  信頼度   : peak_ncc={result.peak_ncc:.4f}  sharpness={result.sharpness:.2f}")

    # 低信頼度警告（DRR自己テスト基準: peak_ncc>0.9, sharpness>1.0）
    if result.peak_ncc < 0.3:
        print(f"  [警告] peak_ncc={result.peak_ncc:.4f} が低すぎます。"
              f"ポジショニング・画像品質を確認してください。")
    elif result.sharpness < 0.4:
        print(f"  [警告] sharpness={result.sharpness:.2f} が低く、推定結果が不確かな可能性があります。")

    # スコアサマリーCSV保存
    scores_csv = Path(out_dir) / f"{patient_id}_scores.csv"
    with open(scores_csv, "w", newline="") as f:
        angles = sorted(result.scores)
        fieldnames = ["angle_deg"] + list(result.scores[angles[0]].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in angles:
            row = {"angle_deg": a}
            row.update(result.scores[a])
            w.writerow(row)

    # 可視化
    fig_path = str(Path(out_dir) / f"{patient_id}_result.png")
    save_result_figure(result, xray_img, gt_angle, fig_path, patient_id)

    return result


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElbowVision 類似度マッチングによる屈曲角推定"
    )
    parser.add_argument("--ct_dir", type=str, help="CTディレクトリ（単一患者）")
    parser.add_argument("--xray",   type=str, help="実LAT X線画像パス（単一患者）")
    parser.add_argument("--gt_angle", type=float, default=None,
                        help="GT屈曲角度（°）—評価用（任意）")
    parser.add_argument("--patient_list", type=str, help="患者リストCSV（バッチ）")
    parser.add_argument("--out_dir", type=str,
                        default="results/similarity_matching",
                        help="出力ディレクトリ")
    parser.add_argument("--laterality", type=str, default=None)
    parser.add_argument("--series_num", type=int, default=None)
    parser.add_argument("--hu_min", type=float, default=50.0)
    parser.add_argument("--hu_max", type=float, default=800.0)
    parser.add_argument("--base_flexion", type=float, default=180.0,
                        help="CT撮影時の基底屈曲角（伸展CT=180°, 90°屈曲CT=90°）")
    parser.add_argument("--metric", type=str, default="combined",
                        choices=["ncc", "ssim", "edge_ncc", "combined", "nmi", "combined_nmi"],
                        help="類似度メトリクス (default: combined = ncc+edge_ncc平均、バイアス打消し)")
    parser.add_argument("--library", type=str, default=None,
                        help="事前構築DRRライブラリ(.npz)パス。指定するとCT生成をスキップし高速化")
    args = parser.parse_args()

    out_dir = str(_PROJECT_ROOT / args.out_dir)

    if args.patient_list:
        # バッチモード
        summary_rows = []
        with open(args.patient_list) as f:
            reader = csv.DictReader(f)
            patients = list(reader)

        print(f"バッチ処理: {len(patients)} 患者")
        for row in patients:
            pid = row["patient_id"]
            try:
                result = run_single(
                    ct_dir       = row["ct_dir"],
                    xray_path    = row["xray_path"],
                    out_dir      = os.path.join(out_dir, pid),
                    laterality   = row.get("laterality") or None,
                    series_num   = int(row["series_num"]) if row.get("series_num") else None,
                    hu_min       = float(row.get("hu_min", 50)),
                    hu_max       = float(row.get("hu_max", 800)),
                    gt_angle     = float(row["gt_angle_deg"]) if row.get("gt_angle_deg") else None,
                    metric       = args.metric,
                    patient_id   = pid,
                    library_path = row.get("library_path") or args.library,
                )
                gt = float(row["gt_angle_deg"]) if row.get("gt_angle_deg") else None
                summary_rows.append({
                    "patient_id":   pid,
                    "pred_angle":   result.best_angle,
                    "gt_angle":     gt,
                    "error":        abs(result.best_angle - gt) if gt else "",
                })
            except Exception as e:
                print(f"  [{pid}] ERROR: {e}")
                summary_rows.append({"patient_id": pid, "pred_angle": "", "gt_angle": "",
                                     "error": str(e)})

        # バッチサマリー
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        summary_csv = Path(out_dir) / "batch_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "pred_angle", "gt_angle", "error"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nサマリー: {summary_csv}")

        errors = [float(r["error"]) for r in summary_rows if r["error"] != ""]
        if errors:
            print(f"MAE = {np.mean(np.abs(errors)):.2f}° (n={len(errors)})")

    elif args.xray:
        # 単一患者モード（--library だけでも --ct_dir なしで動作）
        if not args.ct_dir and not args.library:
            parser.error("--xray 使用時は --ct_dir または --library が必要です")
        lib = str(_PROJECT_ROOT / args.library) if args.library else None
        run_single(
            ct_dir       = str(_PROJECT_ROOT / args.ct_dir) if args.ct_dir else "",
            xray_path    = str(_PROJECT_ROOT / args.xray),
            out_dir      = out_dir,
            laterality   = args.laterality,
            series_num   = args.series_num,
            hu_min       = args.hu_min,
            hu_max       = args.hu_max,
            gt_angle     = args.gt_angle,
            metric       = args.metric,
            patient_id   = Path(args.xray).stem,
            base_flexion = args.base_flexion,
            library_path = lib,
        )
    else:
        parser.error("--ct_dir と --xray、または --patient_list、または --library と --xray が必要です")


if __name__ == "__main__":
    main()
