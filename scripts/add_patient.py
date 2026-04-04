"""
新患者登録 + DRRライブラリ自動生成スクリプト

患者ID・CT・X線・GT角度を指定すると:
  1. DRRライブラリを生成 (data/drr_library/ に .npz 保存)
  2. patients_phase2.csv に行を追記
  3. オプションで即座に類似度マッチング評価を実行

使い方:
  python scripts/add_patient.py \
    --patient_id patient009 \
    --ct_dir "data/raw_dicom/ct_volume/PATIENT009_DIR" \
    --xray data/real_xray/images/009_LAT.png \
    --gt_angle 120.0 \
    --series_num 4 \
    [--run_eval]

  # 複数X線ある場合 (xray/gt_angleをカンマ区切り)
  python scripts/add_patient.py \
    --patient_id patient009 \
    --ct_dir "data/raw_dicom/ct_volume/PATIENT009_DIR" \
    --xray "data/real_xray/images/009_90deg.png,data/real_xray/images/009_120deg.png" \
    --gt_angle "90,120" \
    --series_num 4 \
    --run_eval
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LIBRARY_DIR  = _PROJECT_ROOT / "data" / "drr_library"
_CSV_PATH     = _PROJECT_ROOT / "data" / "real_xray" / "patients_phase2.csv"
_VENV_PYTHON  = _PROJECT_ROOT / "elbow-api" / "venv" / "bin" / "python3"

CSV_FIELDS = [
    "patient_id", "ct_dir", "xray_path", "gt_angle_deg",
    "laterality", "series_num", "hu_min", "hu_max",
    "library_path", "note",
]


def _python() -> str:
    """venv or system python3"""
    return str(_VENV_PYTHON) if _VENV_PYTHON.exists() else "python3"


def build_library(patient_id: str, ct_dir: str, laterality: str,
                  series_num: int, hu_min: float, hu_max: float,
                  angle_min: float, angle_max: float) -> Path:
    """DRRライブラリを生成してパスを返す。既存なら再生成をスキップ"""
    lib_name = (
        f"{patient_id}_series{series_num}_{laterality}_"
        f"{int(angle_min)}to{int(angle_max)}.npz"
    )
    lib_path = _LIBRARY_DIR / lib_name

    if lib_path.exists():
        print(f"  ライブラリ既存: {lib_name} → スキップ")
        return lib_path

    print(f"  DRRライブラリ生成: {lib_name}")
    cmd = [
        _python(),
        str(_PROJECT_ROOT / "scripts" / "build_drr_library.py"),
        "--ct_dir",     ct_dir,
        "--out_path",   str(lib_path),
        "--laterality", laterality,
        "--series_num", str(series_num),
        "--hu_min",     str(hu_min),
        "--hu_max",     str(hu_max),
        "--angle_min",  str(angle_min),
        "--angle_max",  str(angle_max),
        "--angle_step", "1",
    ]
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        print("  ERROR: ライブラリ生成失敗")
        sys.exit(1)
    return lib_path


def append_to_csv(rows: list[dict]) -> None:
    """patients_phase2.csv に患者行を追記（ヘッダ自動生成）"""
    exists = _CSV_PATH.exists()
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            # 既存行のチェック（patient_id + xray_path が一致すればスキップ）
            pass  # 追記のみ
        writer.writerows(rows)
    print(f"  CSV追記: {_CSV_PATH} (+{len(rows)}行)")


def deduplicate_csv() -> None:
    """CSV内の重複行（patient_id + xray_path が同じ）を除去"""
    if not _CSV_PATH.exists():
        return
    seen: set[tuple] = set()
    rows = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("patient_id", ""), row.get("xray_path", ""))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_eval(out_dir: str) -> None:
    """eval_realxray_batch.py を実行"""
    print("\n評価実行中...")
    cmd = [
        _python(),
        str(_PROJECT_ROOT / "scripts" / "eval_realxray_batch.py"),
        "--patient_list", str(_CSV_PATH),
        "--out_dir",      out_dir,
    ]
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="新患者登録 + DRRライブラリ生成")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--ct_dir",     required=True,
                        help="CT DICOMディレクトリ（_PROJECT_ROOT基準の相対パスまたは絶対パス）")
    parser.add_argument("--xray",       required=True,
                        help="X線画像パス（複数はカンマ区切り）")
    parser.add_argument("--gt_angle",   required=True,
                        help="GT角度（度）、複数はカンマ区切り (xrayと同数)")
    parser.add_argument("--laterality", default="R")
    parser.add_argument("--series_num", type=int, default=4)
    parser.add_argument("--hu_min",     type=float, default=50.0)
    parser.add_argument("--hu_max",     type=float, default=800.0)
    parser.add_argument("--angle_min",  type=float, default=60.0)
    parser.add_argument("--angle_max",  type=float, default=180.0)
    parser.add_argument("--note",       default="")
    parser.add_argument("--run_eval",   action="store_true",
                        help="登録後に即座に評価を実行")
    parser.add_argument("--eval_out",   default="results/phase2_eval",
                        help="評価結果出力ディレクトリ")
    args = parser.parse_args()

    xray_paths = [p.strip() for p in args.xray.split(",")]
    gt_angles  = [float(a.strip()) for a in args.gt_angle.split(",")]

    if len(xray_paths) != len(gt_angles):
        print("ERROR: --xray と --gt_angle の数が一致しません")
        sys.exit(1)

    print(f"\n患者登録: {args.patient_id}")
    print(f"  CT: {args.ct_dir}")
    print(f"  X線: {len(xray_paths)}枚")

    # 1. DRRライブラリ生成
    lib_path = build_library(
        patient_id  = args.patient_id,
        ct_dir      = str(_PROJECT_ROOT / args.ct_dir),
        laterality  = args.laterality,
        series_num  = args.series_num,
        hu_min      = args.hu_min,
        hu_max      = args.hu_max,
        angle_min   = args.angle_min,
        angle_max   = args.angle_max,
    )

    # 2. CSV追記
    lib_rel = str(lib_path.relative_to(_PROJECT_ROOT))
    ct_rel  = args.ct_dir  # 渡された値をそのまま使用

    rows = []
    for xp, gt in zip(xray_paths, gt_angles):
        rows.append({
            "patient_id":   args.patient_id,
            "ct_dir":       ct_rel,
            "xray_path":    xp.strip(),
            "gt_angle_deg": gt,
            "laterality":   args.laterality,
            "series_num":   args.series_num,
            "hu_min":       args.hu_min,
            "hu_max":       args.hu_max,
            "library_path": lib_rel,
            "note":         args.note,
        })
    append_to_csv(rows)
    deduplicate_csv()

    print(f"\n登録完了: {args.patient_id} — {len(rows)}行追加")
    print(f"  ライブラリ: {lib_path.name}")
    print(f"  CSV: {_CSV_PATH}")

    # 3. 評価
    if args.run_eval:
        run_eval(str(_PROJECT_ROOT / args.eval_out))


if __name__ == "__main__":
    main()
