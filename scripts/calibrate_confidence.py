"""
信頼度スコアのキャリブレーション分析

peak_ncc / sharpness スコアと角度推定誤差の相関を定量化し、
「信頼度スコアが低い場合に予測を棄却すべき閾値」を決定する。

目的:
  - Phase 2 で信頼度の低い予測を自動フラグ立て
  - 撮影品質の定量的な合否判定ラインを設定
  - 論文 Methods 2.5 の信頼度指標説明に使用

使い方:
  python scripts/calibrate_confidence.py \
    --loo_csv results/self_test_loo/self_test_results.csv \
    --out_dir results/confidence_calibration/

  # 実X線データを追加評価
  python scripts/calibrate_confidence.py \
    --loo_csv results/self_test_loo/self_test_results.csv \
    --realxray_csv results/phase2_eval/predictions.csv \
    --out_dir results/confidence_calibration/

出力:
  confidence_calibration/
  ├── calibration_plot.png    — 散布図 + 閾値
  ├── threshold_table.csv     — 閾値ごとの検出率・見逃し率
  └── calibration_summary.txt — 推奨閾値まとめ
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def load_results(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            peak_ncc  = row.get("peak_ncc")
            sharpness = row.get("sharpness")
            error_key = "error" if "error" in row else "error_deg"
            error_val = row.get(error_key)
            if peak_ncc is not None and sharpness is not None and error_val is not None \
                    and peak_ncc != "" and sharpness != "" and error_val != "":
                try:
                    rows.append({
                        "peak_ncc":  float(peak_ncc),
                        "sharpness": float(sharpness),
                        "error":     float(error_val),
                        "source":    Path(csv_path).stem,
                    })
                except ValueError:
                    pass
    return rows


def compute_threshold_metrics(
    rows: list[dict],
    score_col: str,
    thresholds: list[float],
    error_threshold: float = 3.0,
) -> list[dict]:
    """
    各スコア閾値に対して:
      - 棄却率 (score < threshold → 棄却)
      - 棄却した中で誤差 > error_threshold の割合（True Reject Rate）
      - 通過した中で誤差 > error_threshold の割合（False Pass Rate）
    """
    results = []
    for thr in thresholds:
        rejected = [r for r in rows if r[score_col] < thr]
        accepted = [r for r in rows if r[score_col] >= thr]

        n_total   = len(rows)
        n_rejected = len(rejected)
        n_accepted = len(accepted)

        # 実際に誤差が大きいサンプル
        high_error = [r for r in rows if r["error"] > error_threshold]
        n_high = len(high_error)

        # 棄却された中で正しく棄却（高誤差）
        true_reject  = sum(1 for r in rejected if r["error"] > error_threshold)
        # 通過した中で見逃し（高誤差）
        false_pass   = sum(1 for r in accepted if r["error"] > error_threshold)

        results.append({
            "threshold":      round(thr, 3),
            "rejection_rate": round(n_rejected / n_total, 3) if n_total > 0 else 0,
            "true_reject_rate": round(true_reject / n_high, 3) if n_high > 0 else 0,
            "false_pass_rate":  round(false_pass / n_accepted, 3) if n_accepted > 0 else 0,
            "n_rejected":     n_rejected,
            "n_accepted":     n_accepted,
            "mae_accepted":   round(np.mean([r["error"] for r in accepted]), 3) if accepted else 999,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="信頼度スコアキャリブレーション")
    parser.add_argument("--loo_csv",      required=True,
                        help="LOOテスト結果CSV (peak_ncc, sharpness, error列)")
    parser.add_argument("--realxray_csv", default=None,
                        help="実X線評価CSV（オプション）")
    parser.add_argument("--error_thresh", type=float, default=3.0,
                        help="誤差閾値（これ以上で「高誤差」）")
    parser.add_argument("--out_dir",      default="results/confidence_calibration")
    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    loo_rows = load_results(str(_PROJECT_ROOT / args.loo_csv))
    print(f"LOO DRRデータ: {len(loo_rows)}行")

    realxray_rows = []
    if args.realxray_csv:
        realxray_rows = load_results(str(_PROJECT_ROOT / args.realxray_csv))
        print(f"実X線データ: {len(realxray_rows)}行")
        for r in realxray_rows:
            r["source"] = "real_xray"

    all_rows = loo_rows + realxray_rows

    if not all_rows:
        print("ERROR: データが読み込めません")
        return

    # 統計
    peak_nccs  = np.array([r["peak_ncc"]  for r in all_rows])
    sharpnesses = np.array([r["sharpness"] for r in all_rows])
    errors      = np.array([r["error"]     for r in all_rows])

    print(f"\n全データ統計 (n={len(all_rows)}):")
    print(f"  peak_ncc: mean={peak_nccs.mean():.3f}, std={peak_nccs.std():.3f}, "
          f"min={peak_nccs.min():.3f}, max={peak_nccs.max():.3f}")
    print(f"  sharpness: mean={sharpnesses.mean():.3f}, std={sharpnesses.std():.3f}, "
          f"min={sharpnesses.min():.3f}, max={sharpnesses.max():.3f}")
    print(f"  error: MAE={errors.mean():.3f}°, max={errors.max():.3f}°, "
          f"p95={np.percentile(errors, 95):.3f}°")

    # 相関
    from scipy import stats as scipy_stats
    r_ncc, p_ncc   = scipy_stats.pearsonr(peak_nccs, errors)
    r_shp, p_shp   = scipy_stats.pearsonr(sharpnesses, errors)
    r_ncc_sp, _    = scipy_stats.spearmanr(peak_nccs, errors)
    r_shp_sp, _    = scipy_stats.spearmanr(sharpnesses, errors)

    print(f"\n相関分析:")
    print(f"  peak_ncc vs error:   Pearson r={r_ncc:.3f} (p={p_ncc:.3e}), "
          f"Spearman ρ={r_ncc_sp:.3f}")
    print(f"  sharpness vs error:  Pearson r={r_shp:.3f} (p={p_shp:.3e}), "
          f"Spearman ρ={r_shp_sp:.3f}")

    # ── 閾値分析 ─────────────────────────────────────────────────────────────
    ncc_thresholds = np.arange(0.1, 1.01, 0.05).tolist()
    shp_thresholds = np.arange(0.0, 5.1, 0.25).tolist()

    ncc_metrics = compute_threshold_metrics(all_rows, "peak_ncc",  ncc_thresholds, args.error_thresh)
    shp_metrics = compute_threshold_metrics(all_rows, "sharpness", shp_thresholds, args.error_thresh)

    # 推奨閾値: False Pass Rate < 5% になる最小閾値
    def _find_recommended(metrics: list[dict]) -> dict | None:
        for m in sorted(metrics, key=lambda x: x["threshold"]):
            if m["false_pass_rate"] < 0.05 and m["n_accepted"] > 3:
                return m
        return metrics[-1] if metrics else None

    rec_ncc = _find_recommended(ncc_metrics)
    rec_shp = _find_recommended(shp_metrics)

    summary_text = (
        f"Confidence Score Calibration — ElbowVision LOO + Real X-ray\n"
        f"{'='*60}\n"
        f"n (DRR LOO)    = {len(loo_rows)}\n"
        f"n (Real X-ray) = {len(realxray_rows)}\n"
        f"Error threshold = {args.error_thresh}°\n"
        f"\n--- Statistics ---\n"
        f"peak_ncc  : mean={peak_nccs.mean():.3f}  std={peak_nccs.std():.3f}  "
        f"[{peak_nccs.min():.3f}, {peak_nccs.max():.3f}]\n"
        f"sharpness : mean={sharpnesses.mean():.3f}  std={sharpnesses.std():.3f}  "
        f"[{sharpnesses.min():.3f}, {sharpnesses.max():.3f}]\n"
        f"error     : MAE={errors.mean():.3f}°  p95={np.percentile(errors, 95):.3f}°\n"
        f"\n--- Correlation with Error ---\n"
        f"peak_ncc:  Pearson r={r_ncc:.3f}  Spearman ρ={r_ncc_sp:.3f}\n"
        f"sharpness: Pearson r={r_shp:.3f}  Spearman ρ={r_shp_sp:.3f}\n"
        f"\n--- Recommended Thresholds (False Pass Rate < 5%) ---\n"
    )
    if rec_ncc:
        summary_text += (
            f"peak_ncc  >= {rec_ncc['threshold']:.2f} : "
            f"rejection_rate={rec_ncc['rejection_rate']:.1%}, "
            f"MAE_accepted={rec_ncc['mae_accepted']:.3f}°\n"
        )
    if rec_shp:
        summary_text += (
            f"sharpness >= {rec_shp['threshold']:.2f} : "
            f"rejection_rate={rec_shp['rejection_rate']:.1%}, "
            f"MAE_accepted={rec_shp['mae_accepted']:.3f}°\n"
        )

    # ── Domain gap observation ──────────────────────────────────────────────
    if realxray_rows:
        drr_ncc = np.array([r["peak_ncc"] for r in loo_rows])
        rx_ncc  = np.array([r["peak_ncc"] for r in realxray_rows])
        drr_shp = np.array([r["sharpness"] for r in loo_rows])
        rx_shp  = np.array([r["sharpness"] for r in realxray_rows])

        summary_text += (
            f"\n--- Domain Gap (DRR-to-DRR vs DRR-to-Real-X-ray) ---\n"
            f"peak_ncc  DRR:  mean={drr_ncc.mean():.3f} [{drr_ncc.min():.3f}, {drr_ncc.max():.3f}]\n"
            f"peak_ncc  Real: mean={rx_ncc.mean():.3f} [{rx_ncc.min():.3f}, {rx_ncc.max():.3f}]\n"
            f"sharpness DRR:  mean={drr_shp.mean():.3f} [{drr_shp.min():.3f}, {drr_shp.max():.3f}]\n"
            f"sharpness Real: mean={rx_shp.mean():.3f} [{rx_shp.min():.3f}, {rx_shp.max():.3f}]\n"
            f"\nIMPORTANT NOTE: Non-standard positioning may have HIGHER peak_ncc\n"
            f"than standard positioning (matching is confident but at WRONG angle).\n"
            f"Example: cr_008_2_50kVp (non-standard, err=87.5°) peak_ncc=0.594 > standard ~0.48\n"
            f"→ peak_ncc indicates DRR MATCH QUALITY, not positioning CORRECTNESS.\n"
            f"→ Use peak_ncc as LOW threshold (reject if < 0.3) only, not for non-standard detection.\n"
        )
    print("\n" + summary_text)

    # サマリー保存
    (out_dir / "calibration_summary.txt").write_text(summary_text, encoding="utf-8")

    # 閾値テーブルCSV
    csv_path = out_dir / "threshold_table.csv"
    fieldnames = ["score", "threshold", "rejection_rate",
                  "true_reject_rate", "false_pass_rate", "n_rejected", "n_accepted", "mae_accepted"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in ncc_metrics:
            w.writerow({"score": "peak_ncc", **m})
        for m in shp_metrics:
            w.writerow({"score": "sharpness", **m})
    print(f"閾値テーブル保存: {csv_path}")

    # ── プロット ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

        COLOR_DRR  = "#2196F3"
        COLOR_REAL = "#FF5722"
        COLOR_HIGH = "#f44336"
        COLOR_LOW  = "#4CAF50"

        # ── (A) peak_ncc vs error 散布図 ──────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        loo_ncc  = [r["peak_ncc"] for r in loo_rows]
        loo_err  = [r["error"]    for r in loo_rows]
        ax1.scatter(loo_ncc, loo_err, s=8, alpha=0.5, color=COLOR_DRR, label=f"DRR LOO (n={len(loo_rows)})")
        if realxray_rows:
            rx_ncc = [r["peak_ncc"] for r in realxray_rows]
            rx_err = [r["error"]    for r in realxray_rows]
            ax1.scatter(rx_ncc, rx_err, s=60, alpha=0.8, color=COLOR_REAL, marker="*",
                        label=f"Real X-ray (n={len(realxray_rows)})", zorder=5)
        ax1.axhline(args.error_thresh, color="orange", linewidth=1.2, linestyle="--",
                    label=f"{args.error_thresh}° threshold")
        if rec_ncc:
            ax1.axvline(rec_ncc["threshold"], color="gray", linewidth=1.0, linestyle=":",
                        label=f"Rec. threshold={rec_ncc['threshold']:.2f}")
        ax1.set_xlabel("peak_ncc")
        ax1.set_ylabel("Absolute Error [°]")
        ax1.set_title(f"(A) peak_ncc vs Error\nPearson r={r_ncc:.3f}")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.05)

        # ── (B) sharpness vs error 散布図 ──────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        loo_shp = [r["sharpness"] for r in loo_rows]
        ax2.scatter(loo_shp, loo_err, s=8, alpha=0.5, color=COLOR_DRR, label=f"DRR LOO (n={len(loo_rows)})")
        if realxray_rows:
            rx_shp = [r["sharpness"] for r in realxray_rows]
            ax2.scatter(rx_shp, rx_err, s=60, alpha=0.8, color=COLOR_REAL, marker="*",
                        label=f"Real X-ray (n={len(realxray_rows)})", zorder=5)
        ax2.axhline(args.error_thresh, color="orange", linewidth=1.2, linestyle="--",
                    label=f"{args.error_thresh}° threshold")
        if rec_shp:
            ax2.axvline(rec_shp["threshold"], color="gray", linewidth=1.0, linestyle=":",
                        label=f"Rec. threshold={rec_shp['threshold']:.2f}")
        ax2.set_xlabel("sharpness")
        ax2.set_ylabel("Absolute Error [°]")
        ax2.set_title(f"(B) sharpness vs Error\nPearson r={r_shp:.3f}")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── (C) peak_ncc 分布（高/低誤差別） ──────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        high_err_mask = errors > args.error_thresh
        low_err_mask  = ~high_err_mask
        ax3.hist(peak_nccs[low_err_mask],  bins=30, alpha=0.6, color=COLOR_LOW,
                 label=f"Error ≤ {args.error_thresh}° (n={low_err_mask.sum()})", density=True)
        ax3.hist(peak_nccs[high_err_mask], bins=30, alpha=0.6, color=COLOR_HIGH,
                 label=f"Error > {args.error_thresh}° (n={high_err_mask.sum()})", density=True)
        ax3.set_xlabel("peak_ncc")
        ax3.set_ylabel("Density")
        ax3.set_title(f"(C) peak_ncc Distribution\nby Error Level")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── (D) 閾値 vs 棄却率 / MAE(合格) ──────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ncc_thr_vals    = [m["threshold"]      for m in ncc_metrics]
        ncc_rej_rates   = [m["rejection_rate"] for m in ncc_metrics]
        ncc_mae_acc     = [m["mae_accepted"]   for m in ncc_metrics]
        ncc_fpr         = [m["false_pass_rate"] for m in ncc_metrics]

        ax4_twin = ax4.twinx()
        ax4.plot(ncc_thr_vals, ncc_rej_rates, "b-o", markersize=4, linewidth=1.5,
                 label="Rejection rate")
        ax4_twin.plot(ncc_thr_vals, ncc_mae_acc, "r-s", markersize=4, linewidth=1.5,
                      label="MAE (accepted)")
        ax4.set_xlabel("peak_ncc threshold")
        ax4.set_ylabel("Rejection Rate", color="blue")
        ax4_twin.set_ylabel("MAE of Accepted [°]", color="red")
        ax4.set_title("(D) peak_ncc Threshold vs\nRejection Rate / MAE")
        ax4.set_xlim(0, 1.05)
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1+lines2, labels1+labels2, fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ── (E) 閾値 vs False Pass Rate ──────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        shp_thr_vals  = [m["threshold"]       for m in shp_metrics]
        shp_rej_rates = [m["rejection_rate"]   for m in shp_metrics]
        shp_mae_acc   = [m["mae_accepted"]     for m in shp_metrics]

        ax5_twin = ax5.twinx()
        ax5.plot(shp_thr_vals, shp_rej_rates, "b-o", markersize=4, linewidth=1.5,
                 label="Rejection rate")
        ax5_twin.plot(shp_thr_vals, shp_mae_acc, "r-s", markersize=4, linewidth=1.5,
                      label="MAE (accepted)")
        ax5.set_xlabel("sharpness threshold")
        ax5.set_ylabel("Rejection Rate", color="blue")
        ax5_twin.set_ylabel("MAE of Accepted [°]", color="red")
        ax5.set_title("(E) sharpness Threshold vs\nRejection Rate / MAE")
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1+lines2, labels1+labels2, fontsize=8)
        ax5.grid(True, alpha=0.3)

        # ── (F) 2D マップ peak_ncc × sharpness → error ───────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        sc = ax6.scatter(peak_nccs, sharpnesses, c=errors,
                         cmap="RdYlGn_r", s=12, alpha=0.7,
                         vmin=0, vmax=args.error_thresh * 2)
        plt.colorbar(sc, ax=ax6, label="Error [°]")
        if rec_ncc:
            ax6.axvline(rec_ncc["threshold"], color="blue", linewidth=1.0, linestyle="--",
                        label=f"peak_ncc≥{rec_ncc['threshold']:.2f}")
        if rec_shp:
            ax6.axhline(rec_shp["threshold"], color="orange", linewidth=1.0, linestyle="--",
                        label=f"sharpness≥{rec_shp['threshold']:.2f}")
        ax6.set_xlabel("peak_ncc")
        ax6.set_ylabel("sharpness")
        ax6.set_title(f"(F) 2D Confidence Space\n(color=error)")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        fig.suptitle(
            f"Confidence Score Calibration — DRR LOO (n={len(loo_rows)})"
            + (f" + Real X-ray (n={len(realxray_rows)})" if realxray_rows else ""),
            fontsize=11, fontweight="bold",
        )
        fig.savefig(str(out_dir / "calibration_plot.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"図保存: {out_dir / 'calibration_plot.png'}")

    except Exception as e:
        print(f"図生成失敗: {e}")
        import traceback; traceback.print_exc()

    print(f"\n完了。結果: {out_dir}/")


if __name__ == "__main__":
    main()
