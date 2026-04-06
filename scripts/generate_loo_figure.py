"""
LOO精度マップ図生成スクリプト（論文 Figure 9）

DRR Library Leave-One-Out (LOO) 検証結果を
論文品質（300 DPI）で可視化する。

出力: results/figures/fig9_loo_accuracy.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo_csv",
                        default="results/self_test_loo/self_test_results.csv")
    parser.add_argument("--loo_nb_csv",
                        default="results/self_test_loo_no_boundary/self_test_results.csv")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    })

    loo_csv = _PROJECT_ROOT / args.loo_csv
    if not loo_csv.exists():
        print(f"ERROR: {loo_csv} が存在しません")
        return

    rows = []
    with open(loo_csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gt":    float(r["gt_angle"]),
                "pred":  float(r["pred_angle"]),
                "error": float(r["error"]),
                "ncc":   float(r.get("peak_ncc", 0)),
            })

    gt_arr   = np.array([r["gt"]    for r in rows])
    pred_arr = np.array([r["pred"]  for r in rows])
    err_arr  = np.array([r["error"] for r in rows])
    ncc_arr  = np.array([r["ncc"]   for r in rows])
    diff_arr = pred_arr - gt_arr

    angle_min = gt_arr.min()
    angle_max = gt_arr.max()
    mae  = err_arr.mean()
    rmse = np.sqrt((err_arr**2).mean())

    # Boundary flag
    is_boundary = (np.abs(gt_arr - angle_min) < 0.01) | (np.abs(gt_arr - angle_max) < 0.01)
    interior_mae = err_arr[~is_boundary].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # ── Panel A: GT vs Pred scatter ─────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(gt_arr[~is_boundary], pred_arr[~is_boundary],
                    c=err_arr[~is_boundary], cmap="RdYlGn_r",
                    vmin=0, vmax=1.0, s=20, alpha=0.8, zorder=3,
                    label="Interior")
    ax.scatter(gt_arr[is_boundary], pred_arr[is_boundary],
               c=err_arr[is_boundary], cmap="RdYlGn_r",
               vmin=0, vmax=1.0, s=60, alpha=1.0, marker="D", zorder=4,
               label="Boundary (60°/180°)")
    lim = [angle_min - 3, angle_max + 3]
    ax.plot(lim, lim, "k--", linewidth=0.8, label="Identity")
    ax.set_xlabel("GT Angle [°]")
    ax.set_ylabel("Predicted Angle [°]")
    ax.set_title(f"(A) GT vs Pred — LOO Self-Test\nMAE={mae:.3f}° (interior: {interior_mae:.3f}°)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Error [°]")

    # ── Panel B: Error by angle ─────────────────────────────────────────────
    ax = axes[1]
    bar_colors = ["#FF5722" if b else "#2196F3" for b in is_boundary]
    ax.bar(gt_arr, err_arr, width=1.8, color=bar_colors, alpha=0.85)
    ax.axhline(mae, color="red", linewidth=1.5, linestyle=":",
               label=f"MAE = {mae:.3f}° (all)")
    ax.axhline(interior_mae, color="green", linewidth=1.2, linestyle=":",
               label=f"MAE = {interior_mae:.3f}° (interior)")
    ax.axhline(1.0, color="orange", linewidth=1.0, linestyle="--", alpha=0.7,
               label="1° threshold")
    ax.set_xlabel("GT Angle [°]")
    ax.set_ylabel("Error [°]")
    ax.set_title(f"(B) Error by Angle\nn={len(rows)}, RMSE={rmse:.3f}°")
    ax.legend(fontsize=8, handles=ax.get_legend_handles_labels()[0] + [
        Patch(color="#FF5722", alpha=0.85, label="Boundary (1° LOO error)"),
        Patch(color="#2196F3", alpha=0.85, label="Interior"),
    ])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(angle_min - 2, angle_max + 2)
    ax.set_ylim(0, max(err_arr.max() * 1.3, 1.5))

    # ── Panel C: peak_ncc vs angle ──────────────────────────────────────────
    ax = axes[2]
    ax.scatter(gt_arr[~is_boundary], ncc_arr[~is_boundary],
               s=20, alpha=0.7, color="#2196F3", label=f"Interior (mean={ncc_arr[~is_boundary].mean():.3f})")
    ax.scatter(gt_arr[is_boundary], ncc_arr[is_boundary],
               s=60, alpha=1.0, marker="D", color="#FF5722",
               label=f"Boundary (mean={ncc_arr[is_boundary].mean():.3f})")
    ax.axhline(ncc_arr.mean(), color="gray", linewidth=1.0, linestyle="--",
               label=f"Mean = {ncc_arr.mean():.3f}")
    ax.set_xlabel("GT Angle [°]")
    ax.set_ylabel("peak_ncc (DRR self-test)")
    ax.set_title("(C) DRR-to-DRR Peak NCC by Angle\n(Reference: real X-ray ≈0.48)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(angle_min - 2, angle_max + 2)
    ax.set_ylim(0.95, 1.01)
    # Annotate real X-ray level
    ax.axhline(0.48, color="orange", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.text(angle_max - 5, 0.48 + 0.002, "Real X-ray level (~0.48)",
            ha="right", fontsize=7.5, color="orange")

    fig.suptitle(
        f"DRR Library LOO Validation — n={len(rows)} angles ({int(angle_min)}°–{int(angle_max)}°, 1° step)\n"
        f"Combined NCC metric | MAE={mae:.3f}° (all) | {interior_mae:.3f}° (interior, excl. boundaries)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.35)

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig9_loo_accuracy.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
