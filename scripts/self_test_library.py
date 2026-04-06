"""
DRRライブラリ自己テスト（Phase 2事前検証）

ライブラリ内のDRRを擬似X線として使い、全角度で類似度マッチングを実行。
「実X線が入手できる前に、アルゴリズムの理論的精度上限を測定する」ためのベンチマーク。

目的:
  - 角度別の推定精度マップを生成（どの角度で失敗しやすいか）
  - Phase 2で実X線を撮影するにあたっての優先角度を特定
  - combined metric vs NCC-only の比較

使い方:
  python scripts/self_test_library.py \
    --library data/drr_library/patient008_series4_R_60to180.npz \
    --test_angles "90,100,110,120,130,140,150,160,170,180" \
    --out_dir results/self_test/

  # 全角度テスト（遅い）
  python scripts/self_test_library.py \
    --library data/drr_library/patient008_series4_R_60to180.npz \
    --test_angles all \
    --out_dir results/self_test/

出力:
  results/self_test/
  ├── self_test_results.csv    — GT, Pred, Error 全行
  ├── self_test_summary.txt    — MAE, 角度別統計
  └── self_test_plot.png       — 精度マップ
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="DRRライブラリ自己テスト")
    parser.add_argument("--library", required=True, help=".npz DRRライブラリパス")
    parser.add_argument(
        "--test_angles",
        default="90,100,110,120,130,140,150,160,170,180",
        help="テストするGT角度（カンマ区切り or 'all'）",
    )
    parser.add_argument("--metric", default="combined",
                        choices=["ncc", "edge_ncc", "combined"])
    parser.add_argument("--coarse_step", type=float, default=5.0)
    parser.add_argument("--fine_range", type=float, default=10.0)
    parser.add_argument("--out_dir", default="results/self_test")
    args = parser.parse_args()

    from scripts.similarity_matching import load_drr_library, match_angle_from_library

    lib_path = _PROJECT_ROOT / args.library
    out_dir  = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ライブラリ読み込み
    angles_arr, drrs, meta = load_drr_library(str(lib_path))
    lib_step  = float(meta["angle_step"])
    angle_min = float(meta["angle_min"])
    angle_max = float(meta["angle_max"])
    print(f"ライブラリ: {lib_path.name}")
    print(f"  {len(angles_arr)}角度 ({angle_min:.0f}°〜{angle_max:.0f}°, step={lib_step:.1f}°)")

    # テスト角度リスト
    if args.test_angles.strip().lower() == "all":
        test_angles = angles_arr.tolist()
    else:
        test_angles = [float(a.strip()) for a in args.test_angles.split(",")]
        # ライブラリにある最近傍角度にスナップ
        snapped = []
        for ta in test_angles:
            nearest = min(angles_arr.tolist(), key=lambda a: abs(a - ta))
            snapped.append(nearest)
        test_angles = snapped

    print(f"\nテスト角度: {len(test_angles)}角度")

    # ── 各テスト角度でマッチング ─────────────────────────────────────────────
    results = []
    angle_to_drr: dict[float, np.ndarray] = {
        float(angles_arr[i]): drrs[i] for i in range(len(angles_arr))
    }

    for gt_angle in test_angles:
        # テスト用DRRを擬似X線として渡す（ライブラリ自身からは外す必要はない — LOO不要）
        # ライブラリ内DRRに完全一致するので理論的には0°誤差が期待されるが、
        # LOO（leave-one-out）モードでは差が出る可能性がある
        test_drr = angle_to_drr[gt_angle]  # uint8

        t0 = time.time()
        result = match_angle_from_library(
            str(lib_path),
            test_drr,
            metric=args.metric,
            coarse_step=args.coarse_step,
            fine_range=args.fine_range,
        )
        elapsed = time.time() - t0

        error = abs(result.best_angle - gt_angle)
        results.append({
            "gt_angle":   gt_angle,
            "pred_angle": result.best_angle,
            "error":      error,
            "peak_ncc":   round(result.peak_ncc, 4),
            "sharpness":  round(result.sharpness, 3),
            "elapsed_s":  round(elapsed, 2),
        })
        print(f"  GT={gt_angle:5.1f}° → Pred={result.best_angle:5.1f}° "
              f"Err={error:4.1f}°  ncc={result.peak_ncc:.4f}  sharp={result.sharpness:.2f} "
              f"({elapsed:.2f}s)")

    # ── 統計 ─────────────────────────────────────────────────────────────────
    errors = np.array([r["error"] for r in results])
    diffs  = np.array([r["pred_angle"] - r["gt_angle"] for r in results])
    mae    = errors.mean()
    rmse   = np.sqrt((errors**2).mean())
    bias   = diffs.mean()
    sd     = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    perfect = (errors == 0.0).sum()

    summary_text = (
        f"DRR Library Self-Test — {lib_path.name}\n"
        f"{'='*55}\n"
        f"n              = {len(results)}\n"
        f"Metric         = {args.metric}\n"
        f"MAE            = {mae:.3f}°\n"
        f"RMSE           = {rmse:.3f}°\n"
        f"Mean Bias      = {bias:.3f}°\n"
        f"SD             = {sd:.3f}°\n"
        f"Perfect (0°)   = {perfect}/{len(results)} ({100*perfect/len(results):.0f}%)\n"
        f"{'='*55}\n"
        f"Per-angle results:\n"
    )
    for r in results:
        summary_text += (
            f"  {r['gt_angle']:5.1f}° → {r['pred_angle']:5.1f}° "
            f"(err={r['error']:.1f}°)\n"
        )
    print("\n" + summary_text)

    # CSV保存
    csv_path = out_dir / "self_test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gt_angle","pred_angle","error",
                                               "peak_ncc","sharpness","elapsed_s"])
        writer.writeheader()
        writer.writerows(results)

    # サマリー保存
    summary_path = out_dir / "self_test_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    # ── プロット ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 11, "figure.dpi": 300})
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        gt_arr   = np.array([r["gt_angle"]   for r in results])
        pred_arr = np.array([r["pred_angle"] for r in results])
        err_arr  = np.array([r["error"]      for r in results])
        diff_arr = pred_arr - gt_arr

        # 1. GT vs Pred scatter
        ax = axes[0]
        sc = ax.scatter(gt_arr, pred_arr, c=err_arr, cmap="RdYlGn_r",
                        vmin=0, vmax=10, s=80, alpha=0.9, zorder=3)
        lim = [gt_arr.min() - 5, gt_arr.max() + 5]
        ax.plot(lim, lim, "k--", linewidth=1.0, label="Identity")
        ax.set_xlabel("GT Angle [°]")
        ax.set_ylabel("Pred Angle [°]")
        ax.set_title(f"GT vs Pred (DRR self-test)\nMAE={mae:.2f}°")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Error [°]")

        # 2. Error by angle
        ax = axes[1]
        ax.bar(gt_arr, err_arr, width=3.0, color="#2196F3", alpha=0.8)
        ax.axhline(3.0, color="orange", linestyle="--", linewidth=1.2,
                   label="3° threshold")
        ax.axhline(mae, color="red", linestyle=":", linewidth=1.2,
                   label=f"MAE={mae:.2f}°")
        ax.set_xlabel("GT Angle [°]")
        ax.set_ylabel("Error [°]")
        ax.set_title("Error by Angle (DRR self-test)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Bias by angle
        ax = axes[2]
        colors = ["#f44336" if d < 0 else "#4CAF50" for d in diff_arr]
        ax.bar(gt_arr, diff_arr, width=3.0, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axhline(3.0,  color="orange", linestyle="--", linewidth=1.0)
        ax.axhline(-3.0, color="orange", linestyle="--", linewidth=1.0,
                   label="±3° threshold")
        ax.set_xlabel("GT Angle [°]")
        ax.set_ylabel("Bias (Pred - GT) [°]")
        ax.set_title("Systematic Bias by Angle")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"DRR Library Self-Test — {lib_path.name}\n"
            f"n={len(results)}, metric={args.metric}, MAE={mae:.2f}°",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / "self_test_plot.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"図保存: {out_dir}/self_test_plot.png")
    except Exception as e:
        print(f"図生成失敗: {e}")

    print(f"\n完了。結果: {out_dir}/")


if __name__ == "__main__":
    main()
