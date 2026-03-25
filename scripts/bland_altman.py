"""
ElbowVision Bland-Altman 検証スクリプト

推定値とGround Truth（手動計測値/CT真値）の一致度を統計的に評価する。

使い方:
  python scripts/bland_altman.py --csv results.csv --out_dir results/

CSV形式（ヘッダー必須）:
  filename, gt_carrying_angle, pred_carrying_angle, gt_flexion_deg, pred_flexion_deg

出力:
  - results/bland_altman_carrying.png
  - results/bland_altman_flexion.png
  - results/summary.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# -- 臨床許容範囲（docs/06_精度検証手順.md 準拠） --
CLINICAL_THRESHOLDS = {
    "carrying_angle": {"bias": 1.0, "loa": 5.0},
    "flexion_deg": {"bias": 3.0, "loa": 8.0},
}


# ---------------------------------------------------------------------------
# 統計関数
# ---------------------------------------------------------------------------

@dataclass
class BlandAltmanResult:
    """Bland-Altman 解析の結果を格納するデータクラス"""
    n: int
    mean_diff: float
    std_diff: float
    loa_upper: float
    loa_lower: float
    mae: float
    rmse: float
    r_squared: float
    icc: float
    pearson_r: float
    p_value: float


def compute_bland_altman(gt: np.ndarray, pred: np.ndarray) -> BlandAltmanResult:
    """GT と予測値から Bland-Altman 統計量を算出する。"""
    if len(gt) != len(pred):
        raise ValueError("gt と pred の長さが一致しません")
    if len(gt) < 2:
        raise ValueError("データが2点未満です")

    diff = pred - gt
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    r, p = stats.pearsonr(gt, pred)
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    icc_val = compute_icc(gt, pred)

    return BlandAltmanResult(
        n=len(gt),
        mean_diff=mean_diff,
        std_diff=std_diff,
        loa_upper=loa_upper,
        loa_lower=loa_lower,
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
        icc=icc_val,
        pearson_r=float(r),
        p_value=float(p),
    )


def compute_icc(gt: np.ndarray, pred: np.ndarray) -> float:
    """ICC(3,1) — Two-way mixed, single measures, consistency.

    各被験者を1行、2つの評価者（GT / Pred）を列として計算する。
    """
    n = len(gt)
    if n < 2:
        return 0.0

    # Two-way ANOVA components
    data = np.column_stack([gt, pred])  # (n, 2)
    k = 2  # raters

    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(3,1): (MS_rows - MS_error) / (MS_rows + (k-1)*MS_error)
    denom = ms_rows + (k - 1) * ms_error
    if denom == 0:
        return 0.0
    icc_val = (ms_rows - ms_error) / denom
    return float(np.clip(icc_val, -1.0, 1.0))


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

def plot_bland_altman(
    gt: np.ndarray,
    pred: np.ndarray,
    result: BlandAltmanResult,
    angle_name: str,
    out_path: str,
    unit: str = "deg",
) -> None:
    """Bland-Altman プロットを生成して PNG 保存する。"""
    mean_vals = (gt + pred) / 2
    diff_vals = pred - gt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter
    ax.scatter(mean_vals, diff_vals, color="#3b82f6", alpha=0.7, s=40, zorder=5,
               edgecolors="white", linewidths=0.5)

    # Mean bias line
    ax.axhline(result.mean_diff, color="#22c55e", linewidth=2,
               label=f"Mean bias: {result.mean_diff:+.2f}{unit}")

    # LoA lines
    ax.axhline(result.loa_upper, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"+1.96SD: {result.loa_upper:+.2f}{unit}")
    ax.axhline(result.loa_lower, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"-1.96SD: {result.loa_lower:+.2f}{unit}")

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f"Mean of GT and Prediction [{unit}]", fontsize=12)
    ax.set_ylabel(f"Difference (Pred - GT) [{unit}]", fontsize=12)
    ax.set_title(f"Bland-Altman Plot - {angle_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = (
        f"n={result.n}  |  "
        f"MAE={result.mae:.2f}{unit}  |  "
        f"RMSE={result.rmse:.2f}{unit}  |  "
        f"ICC={result.icc:.3f}  |  "
        f"r\u00b2={result.r_squared:.3f}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", color="#6b7280")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# サマリー出力
# ---------------------------------------------------------------------------

def format_summary(
    results: dict[str, BlandAltmanResult],
) -> str:
    """全角度の統計結果をテキストにまとめる。"""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ElbowVision Bland-Altman Analysis Summary")
    lines.append("=" * 60)

    for name, r in results.items():
        lines.append("")
        lines.append(f"--- {name} (n={r.n}) ---")
        lines.append(f"  Mean Bias      : {r.mean_diff:+.3f} deg")
        lines.append(f"  SD of Diff     : {r.std_diff:.3f} deg")
        lines.append(f"  95% LoA        : [{r.loa_lower:+.3f}, {r.loa_upper:+.3f}] deg")
        lines.append(f"  MAE            : {r.mae:.3f} deg")
        lines.append(f"  RMSE           : {r.rmse:.3f} deg")
        lines.append(f"  Pearson r      : {r.pearson_r:.4f} (p={r.p_value:.2e})")
        lines.append(f"  r^2            : {r.r_squared:.4f}")
        lines.append(f"  ICC(3,1)       : {r.icc:.4f}")

        # 臨床許容範囲の判定
        thresh = CLINICAL_THRESHOLDS.get(name)
        if thresh:
            bias_ok = abs(r.mean_diff) <= thresh["bias"]
            loa_ok = max(abs(r.loa_upper), abs(r.loa_lower)) <= thresh["loa"]
            lines.append(f"  --- Clinical Threshold ---")
            lines.append(
                f"  Bias <=+/-{thresh['bias']} deg : "
                f"{'PASS' if bias_ok else 'FAIL'} ({abs(r.mean_diff):.3f})"
            )
            lines.append(
                f"  LoA  <=+/-{thresh['loa']} deg : "
                f"{'PASS' if loa_ok else 'FAIL'} "
                f"(max={max(abs(r.loa_upper), abs(r.loa_lower)):.3f})"
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    """CSVを読み込み、列名の空白を除去して返す。"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def run_analysis(csv_path: str, out_dir: str) -> dict[str, BlandAltmanResult]:
    """CSV読み込み -> 解析 -> プロット保存 -> サマリー出力。"""
    os.makedirs(out_dir, exist_ok=True)

    df = load_csv(csv_path)

    angle_pairs = [
        ("carrying_angle", "gt_carrying_angle", "pred_carrying_angle"),
        ("flexion_deg", "gt_flexion_deg", "pred_flexion_deg"),
    ]

    results: dict[str, BlandAltmanResult] = {}

    for angle_name, gt_col, pred_col in angle_pairs:
        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"SKIP: {gt_col} or {pred_col} not found in CSV")
            continue

        gt = df[gt_col].to_numpy(dtype=float)
        pred = df[pred_col].to_numpy(dtype=float)

        result = compute_bland_altman(gt, pred)
        results[angle_name] = result

        # carrying_angle -> carrying, flexion_deg -> flexion
        short_name = angle_name.replace("_angle", "").replace("_deg", "")
        out_path = os.path.join(out_dir, f"bland_altman_{short_name}.png")
        plot_bland_altman(gt, pred, result, angle_name, out_path)
        print(f"Saved: {out_path}")

    # サマリー出力
    summary = format_summary(results)
    print(summary)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSummary saved: {summary_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElbowVision Bland-Altman Analysis"
    )
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--out_dir", default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    run_analysis(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
