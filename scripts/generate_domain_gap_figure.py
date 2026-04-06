"""
ドメインギャップ可視化図生成スクリプト（論文 Figure 10 / Supplementary）

DRRと実ファントムLAT X線の構造的差異を4パネルで可視化:
  (A) DRR 90° image
  (B) Real X-ray LAT
  (C) Intensity histograms
  (D) Quantitative metrics bar chart

出力: results/figures/fig10_domain_gap.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_drr_at_90(library_path: Path) -> np.ndarray | None:
    """DRRライブラリから90°に最も近い画像を取得"""
    try:
        data = np.load(str(library_path), allow_pickle=True)
        angles = data["angles"]
        images = data["drrs"]   # key name in library NPZ
        idx = int(np.argmin(np.abs(angles - 90.0)))
        img = images[idx]
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        return img
    except Exception as e:
        print(f"DRR読み込みエラー: {e}")
        return None


def _load_dataset_drr_at_90(dataset_csv: Path, dataset_imgs: Path) -> np.ndarray | None:
    """yolo_dataset_v2 CSV から flexion_deg が90°に最も近いLAT DRRを選択"""
    import csv as _csv
    import cv2 as _cv2

    best = {"path": None, "score": float("inf")}
    try:
        with open(dataset_csv, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("view_type") != "LAT":
                    continue
                score = abs(float(row["flexion_deg"]) - 90.0)
                split = row["split"]
                img_path = dataset_imgs / split / row["filename"]
                if img_path.exists() and score < best["score"]:
                    best = {"path": img_path, "score": score}
    except (FileNotFoundError, KeyError):
        return None

    if best["path"] is None:
        return None

    img = _cv2.imread(str(best["path"]), _cv2.IMREAD_GRAYSCALE)
    print(f"  Dataset DRR: {best['path'].name} (flexion≈{90 - best['score']:.0f}°)")
    return img


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--library",
                        default="data/drr_library/patient008_series4_R_60to180.npz")
    parser.add_argument("--dataset_csv",
                        default="data/yolo_dataset_v2/convnext_labels.csv")
    parser.add_argument("--dataset_imgs",
                        default="data/yolo_dataset_v2/images")
    parser.add_argument("--xray",
                        default="data/real_xray/images/008_LAT.png")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    })

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required (pip install opencv-python)")
        return

    # ── データ読み込み ─────────────────────────────────────────────────────────
    library_path  = _PROJECT_ROOT / args.library
    dataset_csv   = _PROJECT_ROOT / args.dataset_csv
    dataset_imgs  = _PROJECT_ROOT / args.dataset_imgs
    xray_path     = _PROJECT_ROOT / args.xray

    # データセットDRR（yolo_dataset_v2）を優先。なければライブラリDRRを使用
    drr_img = _load_dataset_drr_at_90(dataset_csv, dataset_imgs)
    if drr_img is None:
        print("  Dataset DRRが見つかりません。ライブラリDRRを使用します。")
        drr_img = _load_drr_at_90(library_path)
    if drr_img is None:
        print(f"ERROR: DRR画像が読み込めません")
        return

    real_img = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
    if real_img is None:
        print(f"ERROR: 実X線画像が読み込めません: {xray_path}")
        return

    # ── 画像を同サイズにリサイズ ───────────────────────────────────────────────
    target_size = 256
    drr_resized  = cv2.resize(drr_img,  (target_size, target_size))
    real_resized = cv2.resize(real_img, (target_size, target_size))

    # ── ドメインギャップ指標の計算 ─────────────────────────────────────────────
    def edge_density(img: np.ndarray) -> float:
        edges = cv2.Canny(img, 50, 150)
        return float(edges.sum()) / (edges.size * 255)

    def contrast_rms(img: np.ndarray) -> float:
        return float(img.astype(float).std())

    def hist_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
        h1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
        h2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
        h1 = h1 / (h1.sum() + 1e-8)
        h2 = h2 / (h2.sum() + 1e-8)
        return float(np.minimum(h1, h2).sum())

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(drr_resized, real_resized, data_range=255)
    except ImportError:
        # fallback: simple correlation
        d = drr_resized.astype(float)
        r = real_resized.astype(float)
        ssim_val = float(np.corrcoef(d.ravel(), r.ravel())[0, 1])

    ed_drr  = edge_density(drr_resized)
    ed_real = edge_density(real_resized)
    ct_drr  = contrast_rms(drr_resized)
    ct_real = contrast_rms(real_resized)
    hi      = hist_intersection(drr_resized, real_resized)

    # ── Figure 生成 ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Panel A: DRR image
    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(drr_resized, cmap="gray", vmin=0, vmax=255)
    ax_a.set_title("(A) DRR — 90° Flexion\n(from library)", fontsize=10)
    ax_a.axis("off")
    ax_a.text(0.5, -0.04, f"contrast={ct_drr:.1f}  edge_density={ed_drr:.4f}",
              ha="center", transform=ax_a.transAxes, fontsize=8.5, color="#555")

    # Panel B: Real X-ray
    ax_b = fig.add_subplot(gs[1])
    ax_b.imshow(real_resized, cmap="gray", vmin=0, vmax=255)
    ax_b.set_title("(B) Real Phantom X-ray\n(GT = 90°)", fontsize=10)
    ax_b.axis("off")
    ax_b.text(0.5, -0.04, f"contrast={ct_real:.1f}  edge_density={ed_real:.4f}",
              ha="center", transform=ax_b.transAxes, fontsize=8.5, color="#555")

    # Panel C: Intensity histograms
    ax_c = fig.add_subplot(gs[2])
    bins = np.arange(0, 257, 4)
    h_drr  = np.histogram(drr_resized,  bins=bins)[0].astype(float)
    h_real = np.histogram(real_resized, bins=bins)[0].astype(float)
    h_drr  /= h_drr.sum()
    h_real /= h_real.sum()
    bc = (bins[:-1] + bins[1:]) / 2
    ax_c.plot(bc, h_drr,  color="#1565C0", linewidth=1.5, label="DRR")
    ax_c.plot(bc, h_real, color="#C62828", linewidth=1.5, label="Real X-ray")
    ax_c.fill_between(bc, np.minimum(h_drr, h_real),
                      alpha=0.3, color="#4CAF50",
                      label=f"Intersection={hi:.3f}")
    ax_c.set_xlabel("Pixel Intensity")
    ax_c.set_ylabel("Frequency (normalized)")
    ax_c.set_title("(C) Intensity Histograms", fontsize=10)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # Panel D: Quantitative metrics comparison
    ax_d = fig.add_subplot(gs[3])

    # Show edge density and normalized contrast as grouped bars
    metrics = ["Edge Density\n(×10⁻²)", "Hist. Intersect.\n(0–1)"]
    drr_vals  = [ed_drr  * 100,  hi]
    real_vals = [ed_real * 100,  hi]

    xpos = np.array([0.0, 1.2])
    w = 0.4
    bars1 = ax_d.bar(xpos,       drr_vals,  width=w, color="#1565C0", alpha=0.85, label="DRR")
    bars2 = ax_d.bar(xpos + w,   real_vals, width=w, color="#C62828", alpha=0.85, label="Real X-ray")

    # Override hist intersection to only show one bar (it's a joint metric)
    # Reset the hist intersection bars and draw once centered
    bars1[1].set_height(0)
    bars2[1].set_height(0)
    ax_d.bar(xpos[1] + w / 2, [hi], width=w * 1.5, color="#4CAF50", alpha=0.85, label="Hist Intersect")

    ax_d.set_xticks(xpos + w / 2)
    ax_d.set_xticklabels(metrics)
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.set_title("(D) Domain Gap Metrics", fontsize=10)
    ax_d.grid(True, alpha=0.3, axis="y")

    # SSIM and value annotations
    ax_d.annotate(f"{ed_drr * 100:.2f}", xy=(xpos[0],       drr_vals[0]),
                  xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax_d.annotate(f"{ed_real * 100:.2f}", xy=(xpos[0] + w, real_vals[0]),
                  xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax_d.text(0.5, 0.96, f"SSIM = {ssim_val:.3f}",
              transform=ax_d.transAxes, ha="center", va="top",
              fontsize=9.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                        edgecolor="#F9A825", alpha=0.9))

    fig.suptitle(
        "Domain Gap: DRR vs Real Phantom Lateral X-ray\n"
        f"SSIM={ssim_val:.3f} | Hist Intersection={hi:.3f} | "
        f"Edge density: DRR={ed_drr:.4f} vs Real={ed_real:.4f}",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(top=0.88, bottom=0.14)

    out_dir  = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig10_domain_gap.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"  SSIM             = {ssim_val:.3f}")
    print(f"  Hist Intersection = {hi:.3f}")
    print(f"  Edge density DRR  = {ed_drr:.4f}")
    print(f"  Edge density Real = {ed_real:.4f}")
    print(f"  Contrast DRR      = {ct_drr:.2f}")
    print(f"  Contrast Real     = {ct_real:.2f}")


if __name__ == "__main__":
    main()
