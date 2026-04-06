"""
Phase 2 実X線撮影プロトコルシート生成

撮影計画表（PNG）を生成。研究者が実X線撮影時のチェックリストとして使用。

使い方:
  python scripts/generate_shooting_protocol.py \
    --patient_id patient009 \
    --angles "90,100,110,120,130,140,150,160,170,180" \
    --repeats 3 \
    --out results/shooting_protocol_patient009.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 撮影プロトコルシート生成")
    parser.add_argument("--patient_id", default="patient_XXX")
    parser.add_argument("--angles", default="90,100,110,120,130,140,150,160,170,180")
    parser.add_argument("--repeats", type=int, default=3, help="各角度の撮影回数")
    parser.add_argument("--laterality", default="R", help="撮影側（R/L）")
    parser.add_argument("--kv", default="50-60", help="管電圧 (kVp)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("ERROR: matplotlib required")
        sys.exit(1)

    angles = [float(a.strip()) for a in args.angles.split(",")]
    n_angles = len(angles)
    total_shots = n_angles * args.repeats

    out_path = Path(args.out) if args.out else (
        _PROJECT_ROOT / "results" / f"shooting_protocol_{args.patient_id}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    plt.rcParams.update({"font.size": 10})

    # タイトル
    fig.text(0.5, 0.96, "ElbowVision Phase 2 — X-ray Shooting Protocol",
             ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93, f"Patient: {args.patient_id}   Laterality: {args.laterality}   "
             f"kVp: {args.kv}   Repeats: {args.repeats}x   Total: {total_shots} shots",
             ha="center", va="top", fontsize=11, color="#555555")

    # チェックリスト表
    ax = fig.add_axes([0.05, 0.08, 0.60, 0.80])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, n_angles + 1.5)
    ax.axis("off")

    # ヘッダー
    headers = ["Angle", "Shot 1", "Shot 2", "Shot 3" if args.repeats >= 3 else "",
               "GT (°)", "Notes"]
    col_x = [0.1, 1.2, 2.3, 3.4, 4.5, 5.0]
    y_header = n_angles + 0.8
    for h, x in zip(headers, col_x):
        ax.text(x, y_header, h, fontsize=11, fontweight="bold", va="center",
                color="#1565C0")

    ax.axhline(n_angles + 0.4, color="#1565C0", linewidth=1.5, xmax=0.97)

    # 各角度行
    for i, angle in enumerate(angles):
        y = n_angles - i - 0.1
        bg_color = "#EBF5FB" if i % 2 == 0 else "white"
        rect = FancyBboxPatch((0, y - 0.45), 5.9, 0.85,
                              boxstyle="round,pad=0.02",
                              facecolor=bg_color, edgecolor="none")
        ax.add_patch(rect)

        # 角度表示
        ax.text(col_x[0], y, f"{angle:.0f}°", fontsize=12, fontweight="bold",
                va="center", color="#1A237E")

        # チェックボックス (撮影回数分)
        for r in range(min(args.repeats, 3)):
            ax.text(col_x[r + 1], y, "☐", fontsize=14, va="center", color="#555")

        # GT記入欄
        ax.plot([col_x[4], col_x[4] + 0.4], [y - 0.25, y - 0.25],
                color="#999", linewidth=0.8)

        # 備考欄
        ax.plot([col_x[5], 5.85], [y - 0.25, y - 0.25],
                color="#999", linewidth=0.8)

    ax.axhline(0.4, color="#999999", linewidth=0.8, xmax=0.97)

    # 合計
    ax.text(col_x[0], 0.1, f"Total: {total_shots} shots", fontsize=10,
            va="center", color="#555", style="italic")

    # 右側: 注意事項
    ax2 = fig.add_axes([0.68, 0.08, 0.30, 0.80])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    notes = [
        ("Positioning Checklist", True),
        ("□ LAT view (true lateral)", False),
        ("□ Elbow at 90° to X-ray beam", False),
        ("□ Humerus parallel to table", False),
        ("□ Forearm in neutral rotation", False),
        ("□ SID: 100 cm", False),
        ("", False),
        ("Goniometer Measurement", True),
        ("□ Zero point calibrated", False),
        ("□ Measure at lateral epicondyle", False),
        ("□ Record before + after X-ray", False),
        ("□ 2 independent measurements", False),
        ("", False),
        ("DICOM Save", True),
        ("□ Patient ID anonymized", False),
        ("□ Date correct", False),
        ("□ Series labeled (e.g. LAT_90)", False),
        ("□ Backup to SSD", False),
        ("", False),
        ("Critical Notes", True),
        ("• laterality=R for CT (even left arm)", False),
        ("• Angle range: 60-180° in library", False),
        ("• GT = goniometer, not estimated", False),
    ]

    y_pos = 0.97
    for text, is_header in notes:
        if not text:
            y_pos -= 0.025
            continue
        if is_header:
            ax2.text(0.02, y_pos, text, fontsize=10, fontweight="bold",
                     va="top", color="#1565C0")
            ax2.axhline(y_pos - 0.028, color="#1565C0", linewidth=0.8,
                        xmin=0.01, xmax=0.99)
            y_pos -= 0.06
        else:
            ax2.text(0.04, y_pos, text, fontsize=9, va="top", color="#333")
            y_pos -= 0.04

    # フッター
    fig.text(0.5, 0.02,
             "ElbowVision — Phantom-based Elbow Flexion Angle Estimation Study",
             ha="center", va="bottom", fontsize=9, color="#999")

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"プロトコルシート保存: {out_path}")
    print(f"  角度: {angles}")
    print(f"  総撮影枚数: {total_shots}枚")


if __name__ == "__main__":
    main()
