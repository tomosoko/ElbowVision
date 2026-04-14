"""
DRR vs 実ファントムX線 ドメインギャップ定量比較スクリプト

比較項目:
  1. 画像並列表示（視覚的比較）
  2. ヒストグラム比較（輝度分布の差異）
  3. エッジ抽出比較（骨輪郭の鮮明度）
  4. SSIM / MSE（構造的類似度）
  5. 周波数解析（FFTパワースペクトル）
  6. 輝度プロファイル（骨−軟部組織のコントラスト比較）

使い方:
  cd ~/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/compare_drr_real.py
"""

import csv
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── パス設定 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_XRAY = {
    "AP":  os.path.join(PROJECT_ROOT, "data/real_xray/images/008_AP.png"),
    "LAT": os.path.join(PROJECT_ROOT, "data/real_xray/images/008_LAT.png"),
}

DATASET_CSV  = os.path.join(PROJECT_ROOT, "data/yolo_dataset_v2/convnext_labels.csv")
DATASET_IMGS = os.path.join(PROJECT_ROOT, "data/yolo_dataset_v2/images")

OUT_DIR = os.path.join(PROJECT_ROOT, "results/domain_gap_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def _select_representative_drr() -> dict:
    """
    convnext_labels.csv からビューごとに理想ポジショニングに近い
    DRRサンプルを1枚ずつ自動選択して返す。

    AP  : rotation_error_deg が最小のサンプル（|error| 最小）
    LAT : flexion_deg が 90° に最も近いサンプル
    """
    ap_best = {"path": None, "score": float("inf")}
    lat_best = {"path": None, "score": float("inf")}

    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split    = row["split"]
            filename = row["filename"]
            view     = row["view_type"]

            img_path = os.path.join(DATASET_IMGS, split, filename)
            if not os.path.exists(img_path):
                continue

            if view == "AP":
                score = abs(float(row["rotation_error_deg"]))
                if score < ap_best["score"]:
                    ap_best = {"path": img_path, "score": score}
            elif view == "LAT":
                score = abs(float(row["flexion_deg"]) - 90.0)
                if score < lat_best["score"]:
                    lat_best = {"path": img_path, "score": score}

    if ap_best["path"] is None or lat_best["path"] is None:
        raise FileNotFoundError(
            f"DRRサンプルが見つかりません。データセットを確認してください: {DATASET_IMGS}"
        )

    print(f"  DRR AP  サンプル: {os.path.basename(ap_best['path'])} "
          f"(rotation_error={ap_best['score']:.2f}°)")
    print(f"  DRR LAT サンプル: {os.path.basename(lat_best['path'])} "
          f"(flexion 90° との差={lat_best['score']:.2f}°)")

    return {"AP": ap_best["path"], "LAT": lat_best["path"]}


# ── ユーティリティ ────────────────────────────────────────────────────────────

def load_gray(path: str, target_size: int = 256) -> np.ndarray:
    """グレースケールで読み込み、正方形にリサイズ"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def extract_bone_region(img: np.ndarray) -> np.ndarray:
    """骨領域マスクを抽出（Otsu二値化）"""
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def compute_edge(img: np.ndarray) -> np.ndarray:
    """Cannyエッジ検出"""
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    return cv2.Canny(blurred, 50, 150)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """簡易SSIM（scikit-image不要版）"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_fft_profile(img: np.ndarray) -> np.ndarray:
    """放射方向平均FFTパワースペクトル"""
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    max_r = min(cy, cx)
    radial = np.zeros(max_r)
    counts = np.zeros(max_r)

    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    for ri in range(max_r):
        mask = r == ri
        radial[ri] = magnitude[mask].mean()
        counts[ri] = mask.sum()

    return radial


def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """ヒストグラム交差（0〜1, 1が完全一致）"""
    h1n = h1 / (h1.sum() + 1e-8)
    h2n = h2 / (h2.sum() + 1e-8)
    return float(np.minimum(h1n, h2n).sum())


def compute_contrast_ratio(img: np.ndarray) -> float:
    """骨/背景のコントラスト比"""
    mask = extract_bone_region(img)
    bone_pixels = img[mask > 0]
    bg_pixels = img[mask == 0]
    if len(bone_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0
    return float(bone_pixels.mean()) / max(float(bg_pixels.mean()), 1.0)


# ── メイン比較 ────────────────────────────────────────────────────────────────

def compare_view(view: str, real_path: str, drr_path: str):
    """1ビュー（AP or LAT）のDRR vs 実X線を比較"""
    print(f"\n{'='*60}")
    print(f"  {view} 像比較")
    print(f"{'='*60}")

    real = load_gray(real_path)
    drr = load_gray(drr_path)

    # ── 定量指標 ──
    ssim_val = compute_ssim(drr, real)
    mse_val = float(np.mean((drr.astype(float) - real.astype(float)) ** 2))

    hist_real = cv2.calcHist([real], [0], None, [256], [0, 256]).ravel()
    hist_drr = cv2.calcHist([drr], [0], None, [256], [0, 256]).ravel()
    hist_sim = histogram_intersection(hist_real, hist_drr)

    edge_real = compute_edge(real)
    edge_drr = compute_edge(drr)
    edge_density_real = edge_real.sum() / (255.0 * edge_real.size)
    edge_density_drr = edge_drr.sum() / (255.0 * edge_drr.size)

    contrast_real = compute_contrast_ratio(real)
    contrast_drr = compute_contrast_ratio(drr)

    fft_real = compute_fft_profile(real)
    fft_drr = compute_fft_profile(drr)

    # 高周波成分比率（全体の50%以降 / 全体）
    half = len(fft_real) // 2
    hf_ratio_real = fft_real[half:].sum() / (fft_real.sum() + 1e-8)
    hf_ratio_drr = fft_drr[half:].sum() / (fft_drr.sum() + 1e-8)

    print(f"  SSIM:                 {ssim_val:.4f}  (1.0=完全一致)")
    print(f"  MSE:                  {mse_val:.1f}")
    print(f"  ヒストグラム交差:     {hist_sim:.4f}  (1.0=完全一致)")
    print(f"  エッジ密度 実X線:     {edge_density_real:.4f}")
    print(f"  エッジ密度 DRR:       {edge_density_drr:.4f}")
    print(f"  コントラスト比 実X線: {contrast_real:.2f}")
    print(f"  コントラスト比 DRR:   {contrast_drr:.2f}")
    print(f"  高周波比率 実X線:     {hf_ratio_real:.4f}")
    print(f"  高周波比率 DRR:       {hf_ratio_drr:.4f}")

    # ── 図生成 ──
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Domain Gap Analysis — {view} View", fontsize=16, fontweight="bold")
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: 元画像 + エッジ
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(real, cmap="gray")
    ax1.set_title("Real X-ray", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(drr, cmap="gray")
    ax2.set_title("DRR (synthetic)", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(edge_real, cmap="gray")
    ax3.set_title(f"Edge: Real (density={edge_density_real:.4f})", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(edge_drr, cmap="gray")
    ax4.set_title(f"Edge: DRR (density={edge_density_drr:.4f})", fontsize=10)
    ax4.axis("off")

    # Row 2: ヒストグラム + 差分画像
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.plot(hist_real / hist_real.sum(), label="Real X-ray", alpha=0.8, linewidth=1.5)
    ax5.plot(hist_drr / hist_drr.sum(), label="DRR", alpha=0.8, linewidth=1.5)
    ax5.set_title(f"Intensity Histogram (intersection={hist_sim:.4f})", fontsize=11)
    ax5.set_xlabel("Pixel Value")
    ax5.set_ylabel("Normalized Frequency")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2:4])
    diff = np.abs(real.astype(float) - drr.astype(float))
    ax6.imshow(diff, cmap="hot")
    ax6.set_title(f"Absolute Difference (MSE={mse_val:.1f})", fontsize=11)
    ax6.axis("off")
    plt.colorbar(ax6.images[0], ax=ax6, fraction=0.046)

    # Row 3: FFTパワースペクトル + 骨領域マスク比較
    ax7 = fig.add_subplot(gs[2, 0:2])
    freqs = np.arange(len(fft_real)) / len(fft_real)
    ax7.plot(freqs, fft_real, label="Real X-ray", alpha=0.8, linewidth=1.5)
    ax7.plot(freqs[:len(fft_drr)], fft_drr, label="DRR", alpha=0.8, linewidth=1.5)
    ax7.set_title("Radial FFT Power Spectrum", fontsize=11)
    ax7.set_xlabel("Normalized Frequency")
    ax7.set_ylabel("Log Power")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    bone_real = extract_bone_region(real)
    bone_drr = extract_bone_region(drr)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.imshow(bone_real, cmap="gray")
    ax8.set_title("Bone Mask: Real", fontsize=10)
    ax8.axis("off")

    ax9 = fig.add_subplot(gs[2, 3])
    ax9.imshow(bone_drr, cmap="gray")
    ax9.set_title("Bone Mask: DRR", fontsize=10)
    ax9.axis("off")

    # Row 4: 中央ライン輝度プロファイル（横・縦）
    mid_y = real.shape[0] // 2
    mid_x = real.shape[1] // 2

    ax10 = fig.add_subplot(gs[3, 0:2])
    ax10.plot(real[mid_y, :], label="Real X-ray", alpha=0.8, linewidth=1.5)
    ax10.plot(drr[mid_y, :], label="DRR", alpha=0.8, linewidth=1.5)
    ax10.set_title("Horizontal Intensity Profile (mid-row)", fontsize=11)
    ax10.set_xlabel("Pixel Position")
    ax10.set_ylabel("Intensity")
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[3, 2:4])
    ax11.plot(real[:, mid_x], label="Real X-ray", alpha=0.8, linewidth=1.5)
    ax11.plot(drr[:, mid_x], label="DRR", alpha=0.8, linewidth=1.5)
    ax11.set_title("Vertical Intensity Profile (mid-col)", fontsize=11)
    ax11.set_xlabel("Pixel Position")
    ax11.set_ylabel("Intensity")
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, f"domain_gap_{view}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存: {out_path}")

    return {
        "view": view,
        "ssim": ssim_val,
        "mse": mse_val,
        "hist_intersection": hist_sim,
        "edge_density_real": edge_density_real,
        "edge_density_drr": edge_density_drr,
        "contrast_real": contrast_real,
        "contrast_drr": contrast_drr,
        "hf_ratio_real": hf_ratio_real,
        "hf_ratio_drr": hf_ratio_drr,
    }


def generate_summary(results: list):
    """サマリーレポートをテキスト出力"""
    print(f"\n{'='*60}")
    print("  ドメインギャップ サマリー")
    print(f"{'='*60}")

    header = f"{'指標':<28} {'AP':>10} {'LAT':>10} {'判定':>10}"
    print(header)
    print("-" * 60)

    r = {r["view"]: r for r in results}

    rows = [
        ("SSIM (↑ 良い)", "ssim", 0.3, True),
        ("MSE (↓ 良い)", "mse", 3000, False),
        ("ヒストグラム交差 (↑)", "hist_intersection", 0.5, True),
        ("エッジ密度比 DRR/Real", None, None, None),
        ("コントラスト比 DRR/Real", None, None, None),
        ("高周波比率 DRR/Real", None, None, None),
    ]

    for label, key, thresh, higher_better in rows:
        if key is not None:
            ap_val = r["AP"][key]
            lat_val = r["LAT"][key]
            if higher_better:
                judge = "OK" if (ap_val > thresh and lat_val > thresh) else "要改善"
            else:
                judge = "OK" if (ap_val < thresh and lat_val < thresh) else "要改善"
            print(f"  {label:<26} {ap_val:>10.4f} {lat_val:>10.4f} {judge:>10}")
        elif "エッジ" in label:
            ap_ratio = r["AP"]["edge_density_drr"] / max(r["AP"]["edge_density_real"], 1e-8)
            lat_ratio = r["LAT"]["edge_density_drr"] / max(r["LAT"]["edge_density_real"], 1e-8)
            judge = "OK" if (0.5 < ap_ratio < 2.0 and 0.5 < lat_ratio < 2.0) else "要改善"
            print(f"  {label:<26} {ap_ratio:>10.4f} {lat_ratio:>10.4f} {judge:>10}")
        elif "コントラスト" in label:
            ap_ratio = r["AP"]["contrast_drr"] / max(r["AP"]["contrast_real"], 1e-8)
            lat_ratio = r["LAT"]["contrast_drr"] / max(r["LAT"]["contrast_real"], 1e-8)
            judge = "OK" if (0.7 < ap_ratio < 1.3 and 0.7 < lat_ratio < 1.3) else "要改善"
            print(f"  {label:<26} {ap_ratio:>10.4f} {lat_ratio:>10.4f} {judge:>10}")
        elif "高周波" in label:
            ap_ratio = r["AP"]["hf_ratio_drr"] / max(r["AP"]["hf_ratio_real"], 1e-8)
            lat_ratio = r["LAT"]["hf_ratio_drr"] / max(r["LAT"]["hf_ratio_real"], 1e-8)
            judge = "OK" if (0.5 < ap_ratio < 2.0 and 0.5 < lat_ratio < 2.0) else "要改善"
            print(f"  {label:<26} {ap_ratio:>10.4f} {lat_ratio:>10.4f} {judge:>10}")

    # 改善提案
    print(f"\n{'='*60}")
    print("  改善提案")
    print(f"{'='*60}")

    suggestions = []
    ap, lat = r["AP"], r["LAT"]

    if ap["ssim"] < 0.3 or lat["ssim"] < 0.3:
        suggestions.append(
            "SSIM低: target_sizeを128→256以上に上げてDRR解像度を向上させる"
        )

    if ap["hist_intersection"] < 0.5 or lat["hist_intersection"] < 0.5:
        suggestions.append(
            "ヒストグラム乖離大: HUウィンドウ(hu_min/hu_max)を調整し、"
            "ガンマ補正・CLAHE パラメータを実X線に合わせる"
        )

    edge_ratio_ap = ap["edge_density_drr"] / max(ap["edge_density_real"], 1e-8)
    edge_ratio_lat = lat["edge_density_drr"] / max(lat["edge_density_real"], 1e-8)
    if edge_ratio_ap < 0.5 or edge_ratio_lat < 0.5:
        suggestions.append(
            "エッジ密度低(DRR): 解像度不足 → target_size増加 + ガウスブラー低減"
        )
    elif edge_ratio_ap > 2.0 or edge_ratio_lat > 2.0:
        suggestions.append(
            "エッジ密度高(DRR): CLAHE が強すぎる → clipLimit を下げる"
        )

    hf_ratio_ap = ap["hf_ratio_drr"] / max(ap["hf_ratio_real"], 1e-8)
    hf_ratio_lat = lat["hf_ratio_drr"] / max(lat["hf_ratio_real"], 1e-8)
    if hf_ratio_ap < 0.5 or hf_ratio_lat < 0.5:
        suggestions.append(
            "高周波成分不足(DRR): 骨テクスチャが欠落 → "
            "解像度向上 or テクスチャ合成augmentation追加"
        )

    contrast_ratio_ap = ap["contrast_drr"] / max(ap["contrast_real"], 1e-8)
    contrast_ratio_lat = lat["contrast_drr"] / max(lat["contrast_real"], 1e-8)
    if contrast_ratio_ap < 0.7 or contrast_ratio_lat < 0.7:
        suggestions.append(
            "コントラスト不足(DRR): ファントム外殻がHUウィンドウに含まれている可能性 "
            "→ hu_min を引き上げて外殻除去"
        )
    elif contrast_ratio_ap > 1.3 or contrast_ratio_lat > 1.3:
        suggestions.append(
            "コントラスト過剰(DRR): ウィンドウが狭すぎ or ガンマ補正強すぎ"
        )

    if not suggestions:
        suggestions.append("大きな問題は検出されませんでした")

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    # テキストレポート保存
    report_path = os.path.join(OUT_DIR, "domain_gap_report.txt")
    with open(report_path, "w") as f:
        f.write("Domain Gap Analysis Report\n")
        f.write(f"{'='*60}\n\n")
        for view in ["AP", "LAT"]:
            rv = r[view]
            f.write(f"[{view} View]\n")
            for k, v in rv.items():
                if k != "view":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("Suggestions:\n")
        for i, s in enumerate(suggestions, 1):
            f.write(f"  {i}. {s}\n")
    print(f"\n  レポート保存: {report_path}")


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    for view in ["AP", "LAT"]:
        if not os.path.exists(REAL_XRAY[view]):
            print(f"ERROR: 実X線画像が見つかりません: {REAL_XRAY[view]}")
            sys.exit(1)

    print("代表DRRサンプルを自動選択中...")
    try:
        drr_samples = _select_representative_drr()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = []
    for view in ["AP", "LAT"]:
        r = compare_view(view, REAL_XRAY[view], drr_samples[view])
        results.append(r)

    generate_summary(results)
    print(f"\n完了。結果は {OUT_DIR}/ に保存されました。")


if __name__ == "__main__":
    main()
