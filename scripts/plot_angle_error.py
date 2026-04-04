"""
角度別誤差可視化スクリプト（Phase 2 分析用）

predictions.csv から角度ごとのMAEをプロットし、
系統誤差（特定の屈曲角で誤差が大きいか）を確認する。

使い方:
  python scripts/plot_angle_error.py \
    --csv results/phase2_eval/predictions.csv \
    --out results/phase2_eval/angle_error_plot.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="角度別誤差可視化")
    parser.add_argument("--csv", required=True, help="predictions.csv パス")
    parser.add_argument("--out", default=None, help="出力PNG（省略時はCSVと同ディレクトリ）")
    parser.add_argument("--bin", type=float, default=10.0, help="角度ビン幅(°)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "angle_error_plot.png"

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("gt_flexion_deg") and r.get("pred_flexion_deg"):
                rows.append({
                    "gt":   float(r["gt_flexion_deg"]),
                    "pred": float(r["pred_flexion_deg"]),
                    "err":  abs(float(r["pred_flexion_deg"]) - float(r["gt_flexion_deg"])),
                    "pid":  r.get("patient_id", ""),
                })

    if not rows:
        print("有効データなし（gt_flexion_deg / pred_flexion_deg 列を確認）")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError as e:
        print(f"依存ライブラリ不足: {e}")
        sys.exit(1)

    gt   = np.array([r["gt"]   for r in rows])
    pred = np.array([r["pred"] for r in rows])
    err  = np.array([r["err"]  for r in rows])
    diff = pred - gt  # 符号付き誤差

    # ── 角度ビン別統計 ─────────────────────────────────────────────────────────
    bins = np.arange(
        np.floor(gt.min() / args.bin) * args.bin,
        np.ceil(gt.max() / args.bin) * args.bin + args.bin,
        args.bin,
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_mae  = []
    bin_bias = []
    bin_n    = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gt >= lo) & (gt < hi)
        if mask.any():
            bin_mae.append(np.abs(diff[mask]).mean())
            bin_bias.append(diff[mask].mean())
            bin_n.append(mask.sum())
        else:
            bin_mae.append(np.nan)
            bin_bias.append(np.nan)
            bin_n.append(0)
    bin_mae  = np.array(bin_mae)
    bin_bias = np.array(bin_bias)

    # ── プロット ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. 角度ビン別MAE
    ax = axes[0, 0]
    valid = ~np.isnan(bin_mae)
    ax.bar(bin_centers[valid], bin_mae[valid], width=args.bin * 0.8,
           color="#2196F3", alpha=0.8)
    ax.axhline(3.0, color="orange", linestyle="--", linewidth=1.2, label="3° 臨床閾値")
    ax.axhline(np.nanmean(bin_mae), color="red", linestyle=":", linewidth=1.2,
               label=f"Overall MAE={np.nanmean(bin_mae):.2f}°")
    for xc, m, n_ in zip(bin_centers[valid], bin_mae[valid], np.array(bin_n)[valid]):
        if n_ > 0:
            ax.text(xc, m + 0.1, f"n={n_}", ha="center", fontsize=8)
    ax.set_xlabel("GT Flexion Angle [°]")
    ax.set_ylabel("MAE [°]")
    ax.set_title("MAE by Angle Bin")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Bias by angle bin
    ax = axes[0, 1]
    colors = ["#f44336" if b < 0 else "#4CAF50" for b in bin_bias[valid]]
    ax.bar(bin_centers[valid], bin_bias[valid], width=args.bin * 0.8,
           color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(3.0,  color="orange", linestyle="--", linewidth=1.0, label="±3° threshold")
    ax.axhline(-3.0, color="orange", linestyle="--", linewidth=1.0)
    ax.set_xlabel("GT Flexion Angle [°]")
    ax.set_ylabel("Mean Bias (Pred − GT) [°]")
    ax.set_title("Systematic Bias by Angle Bin")
    ax.grid(True, alpha=0.3)

    # 3. 散布図 (GT vs Pred)
    ax = axes[1, 0]
    sc = ax.scatter(gt, pred, c=err, cmap="RdYlGn_r", vmin=0, vmax=10,
                    s=60, alpha=0.8, zorder=3)
    lim = [min(gt.min(), pred.min()) - 5, max(gt.max(), pred.max()) + 5]
    ax.plot(lim, lim, "k--", linewidth=1.0, label="Identity")
    ax.set_xlabel("GT Angle [°]")
    ax.set_ylabel("Pred Angle [°]")
    ax.set_title(f"GT vs Pred  (n={len(rows)})")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Error [°]")

    # 4. 誤差分布 ヒストグラム
    ax = axes[1, 1]
    ax.hist(diff, bins=15, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.axvline(diff.mean(), color="red", linewidth=1.5,
               label=f"Mean={diff.mean():.2f}°")
    ax.axvline(diff.mean() + 1.96 * diff.std(ddof=1), color="orange",
               linestyle="--", linewidth=1.2,
               label=f"95% LoA ±{1.96*diff.std(ddof=1):.2f}°")
    ax.axvline(diff.mean() - 1.96 * diff.std(ddof=1), color="orange",
               linestyle="--", linewidth=1.2)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Pred − GT [°]")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"ElbowVision Phase 2 — Similarity Matching Error Analysis  (n={len(rows)})\n"
        f"Overall MAE={np.abs(diff).mean():.2f}°  Bias={diff.mean():.2f}°  "
        f"LoA=[{diff.mean()-1.96*diff.std(ddof=1):.1f}, {diff.mean()+1.96*diff.std(ddof=1):.1f}]°",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")
    print(f"n={len(rows)}, MAE={np.abs(diff).mean():.3f}°, Bias={diff.mean():.3f}°")


if __name__ == "__main__":
    main()
