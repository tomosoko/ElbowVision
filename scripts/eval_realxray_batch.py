"""
実X線バッチ評価スクリプト（Phase 2向け）

複数患者・複数角度の実X線を類似度マッチングで評価し、
Bland-Altman解析用CSVを生成する。

使い方:
  python scripts/eval_realxray_batch.py \
    --patient_list data/real_xray/patients_phase2_template.csv \
    --out_dir results/phase2_eval/ \
    --method similarity

出力:
  results/phase2_eval/
  ├── predictions.csv          — GT vs Pred 全行
  ├── bland_altman_realxray.png — Bland-Altmanプロット
  └── summary.txt              — MAE, Bias, LoA, ICC, r²

患者リストCSV形式:
  patient_id, ct_dir, xray_path, gt_angle_deg, laterality, series_num,
  hu_min, hu_max, library_path (省略可), note (省略可)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))
sys.path.insert(0, str(_PROJECT_ROOT))


def run_similarity(row: dict, out_dir: str, lib_cache=None,
                   metric: str = "combined") -> dict | None:
    """類似度マッチングで1画像を評価"""
    from scripts.similarity_matching import run_single

    pid       = row["patient_id"]
    ct_dir    = str(_PROJECT_ROOT / row["ct_dir"])
    xray_path = str(_PROJECT_ROOT / row["xray_path"])
    gt_angle  = float(row["gt_angle_deg"]) if row.get("gt_angle_deg") else None
    laterality= row.get("laterality") or None
    series_num= int(row["series_num"]) if row.get("series_num") else None
    hu_min    = float(row.get("hu_min", 50))
    hu_max    = float(row.get("hu_max", 800))
    lib_path  = str(_PROJECT_ROOT / row["library_path"]) if row.get("library_path") else None

    t0 = time.time()
    if lib_cache is not None:
        # プリロード済みキャッシュを使って直接マッチング
        import cv2 as _cv2
        xray_img = _cv2.imread(xray_path, _cv2.IMREAD_GRAYSCALE)
        if xray_img is None:
            raise FileNotFoundError(f"X線画像が読み込めません: {xray_path}")
        result = lib_cache.match(xray_img, metric=metric)
        _out = os.path.join(out_dir, pid)
        os.makedirs(_out, exist_ok=True)
    else:
        result = run_single(
            ct_dir       = ct_dir,
            xray_path    = xray_path,
            out_dir      = os.path.join(out_dir, pid),
            laterality   = laterality,
            series_num   = series_num,
            hu_min       = hu_min,
            hu_max       = hu_max,
            gt_angle     = gt_angle,
            metric       = metric,
            patient_id   = pid,
            library_path = lib_path,
        )
    elapsed = time.time() - t0

    return {
        "patient_id":     pid,
        "xray_path":      row["xray_path"],
        "gt_flexion_deg": gt_angle,
        "pred_flexion_deg": result.best_angle,
        "error_deg":      abs(result.best_angle - gt_angle) if gt_angle is not None else None,
        "peak_ncc":       round(result.peak_ncc, 4),
        "sharpness":      round(result.sharpness, 3),
        "elapsed_s":      round(elapsed, 1),
        "note":           row.get("note", ""),
    }


def bland_altman_analysis(predictions_csv: str, out_dir: str) -> None:
    """GT vs Pred CSV からBland-Altman解析を実行"""
    from scipy import stats

    rows = []
    with open(predictions_csv) as f:
        for r in csv.DictReader(f):
            if r["gt_flexion_deg"] not in (None, "") and r["pred_flexion_deg"] not in (None, ""):
                rows.append({
                    "gt":   float(r["gt_flexion_deg"]),
                    "pred": float(r["pred_flexion_deg"]),
                })

    if len(rows) < 2:
        print("  BA解析: データ不足（n<2）")
        return

    gt   = np.array([r["gt"]   for r in rows])
    pred = np.array([r["pred"] for r in rows])
    diff = pred - gt
    mean_vals = (pred + gt) / 2.0

    n     = len(rows)
    bias  = diff.mean()
    sd    = diff.std(ddof=1)
    loa_l = bias - 1.96 * sd
    loa_u = bias + 1.96 * sd
    mae   = np.abs(diff).mean()
    rmse  = np.sqrt((diff**2).mean())
    r, p  = stats.pearsonr(gt, pred)

    # ICC(3,1) 2-way mixed, absolute agreement (Shrout & Fleiss 1979)
    # k=2 raters (GT, Pred), n subjects
    # Data matrix: rows=subjects, cols=[gt, pred]
    data  = np.column_stack([gt, pred])          # (n, 2)
    grand_mean = data.mean()
    ss_r  = 2 * np.sum((data.mean(axis=1) - grand_mean) ** 2)   # between-subjects
    ss_c  = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)   # between-raters
    ss_e  = np.sum((data - data.mean(axis=1, keepdims=True)
                       - data.mean(axis=0, keepdims=True)
                       + grand_mean) ** 2)                       # residual
    ms_r  = ss_r / (n - 1)
    ms_e  = ss_e / ((n - 1) * (2 - 1))
    ms_c  = ss_c / (2 - 1)
    # ICC(3,1): absolute agreement
    icc   = (ms_r - ms_e) / (ms_r + (2 - 1) * ms_e)
    icc   = float(np.clip(icc, -1.0, 1.0))

    icc_str = f"{icc:.4f}" if n >= 5 else f"{icc:.4f} (n<5, 参考値)"

    summary = (
        f"Bland-Altman Analysis — Real X-ray Phase 2\n"
        f"{'='*55}\n"
        f"n              = {n}\n"
        f"MAE            = {mae:.3f}°\n"
        f"RMSE           = {rmse:.3f}°\n"
        f"Mean Bias      = {bias:.3f}°\n"
        f"95% LoA        = [{loa_l:.3f}, {loa_u:.3f}]°\n"
        f"Pearson r      = {r:.4f} (p={p:.4f})\n"
        f"ICC(3,1)       = {icc_str}\n"
        f"{'='*55}\n"
        f"臨床許容基準 (flexion):\n"
        f"  Bias ≤±3°: {'PASS' if abs(bias)<=3 else 'FAIL'} ({bias:.2f}°)\n"
        f"  LoA ≤±8°:  {'PASS' if abs(loa_l)<=8 and abs(loa_u)<=8 else 'FAIL'}"
        f" ({loa_l:.2f} to {loa_u:.2f}°)\n"
        f"  ICC ≥0.90: {'PASS' if icc>=0.90 else 'FAIL'} ({icc:.4f})\n"
    )
    print(summary)

    summary_path = Path(out_dir) / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    # Bland-Altmanプロット
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 11, "figure.dpi": 300})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bland-Altman
        ax = axes[0]
        ax.scatter(mean_vals, diff, s=60, alpha=0.7, color="#2196F3")
        ax.axhline(bias,  color="red",    linewidth=1.5, label=f"Bias: {bias:.2f}°")
        ax.axhline(loa_u, color="orange", linewidth=1.2, linestyle="--",
                   label=f"LoA: [{loa_l:.1f}, {loa_u:.1f}]°")
        ax.axhline(loa_l, color="orange", linewidth=1.2, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Mean of GT and Pred [°]")
        ax.set_ylabel("Pred − GT [°]")
        ax.set_title(f"Bland-Altman Plot (n={n})\nMAE={mae:.2f}°")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Identity plot
        ax2 = axes[1]
        lim = [min(gt.min(), pred.min()) - 5, max(gt.max(), pred.max()) + 5]
        ax2.scatter(gt, pred, s=60, alpha=0.7, color="#4CAF50")
        ax2.plot(lim, lim, "k--", linewidth=1, label="Identity")
        ax2.set_xlabel("GT Angle [°]")
        ax2.set_ylabel("Pred Angle [°]")
        ax2.set_title(f"GT vs Pred\nr={r:.3f}")
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / "bland_altman_realxray.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  BA図保存: {Path(out_dir)/'bland_altman_realxray.png'}")
    except Exception as e:
        print(f"  BA図生成失敗: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="実X線バッチ評価（Phase 2向け）")
    parser.add_argument("--patient_list", required=True,
                        help="患者リストCSV（patients_phase2_template.csv 参照）")
    parser.add_argument("--out_dir", default="results/phase2_eval")
    parser.add_argument("--method", default="similarity",
                        choices=["similarity"],
                        help="評価手法（現在: similarity のみ）")
    parser.add_argument("--metric", default="combined",
                        choices=["ncc", "edge_ncc", "combined", "nmi", "combined_nmi"],
                        help="類似度メトリクス（推奨: combined_nmi）")
    parser.add_argument("--skip_ba", action="store_true",
                        help="Bland-Altman解析をスキップ")
    args = parser.parse_args()

    out_dir = str(_PROJECT_ROOT / args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 患者リスト読み込み
    with open(_PROJECT_ROOT / args.patient_list) as f:
        patients = [row for row in csv.DictReader(f)
                    if not row.get("patient_id", "").startswith("#")]

    print(f"評価開始: {len(patients)} X線画像")

    # DRRLibraryCacheをライブラリパスごとに1つ作成（同一ライブラリを複数X線で共有）
    from scripts.similarity_matching import DRRLibraryCache
    _lib_caches: dict[str, DRRLibraryCache] = {}

    def _get_cache(lib_path: str) -> DRRLibraryCache:
        if lib_path not in _lib_caches:
            _lib_caches[lib_path] = DRRLibraryCache(lib_path)
        return _lib_caches[lib_path]

    results = []
    for i, row in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] {row['patient_id']} — {Path(row['xray_path']).name}")
        try:
            r = run_similarity(row, out_dir,
                               lib_cache=_get_cache(
                                   str(_PROJECT_ROOT / row["library_path"])
                               ) if row.get("library_path") else None,
                               metric=args.metric)
            if r:
                results.append(r)
                if r["error_deg"] is not None:
                    print(f"  → Pred={r['pred_flexion_deg']:.1f}° GT={r['gt_flexion_deg']:.1f}° "
                          f"Err={r['error_deg']:.1f}° ({r['elapsed_s']}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "patient_id": row["patient_id"], "xray_path": row["xray_path"],
                "gt_flexion_deg": row.get("gt_angle_deg"), "pred_flexion_deg": None,
                "error_deg": None, "peak_ncc": None, "sharpness": None,
                "elapsed_s": 0, "note": f"ERROR: {e}",
            })

    # CSV出力
    pred_csv = Path(out_dir) / "predictions.csv"
    fieldnames = ["patient_id", "xray_path", "gt_flexion_deg", "pred_flexion_deg",
                  "error_deg", "peak_ncc", "sharpness", "elapsed_s", "note"]
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\n予測CSV保存: {pred_csv}")

    # サマリー
    valid = [r for r in results if r["error_deg"] is not None]
    if valid:
        mae = np.mean([r["error_deg"] for r in valid])
        print(f"\nMAE = {mae:.2f}° (n={len(valid)})")

    # Bland-Altman解析
    if not args.skip_ba and len(valid) >= 2:
        print("\nBland-Altman解析中...")
        bland_altman_analysis(str(pred_csv), out_dir)

    print(f"\n完了。結果: {out_dir}/")


if __name__ == "__main__":
    main()
