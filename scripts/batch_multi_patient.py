"""
ElbowVision マルチ患者バッチDRR生成スクリプト（本研究向け）

患者リストCSVを読み込み、各患者の伸展CTからangle_dataset形式のDRRを
並列生成してプールデータセットを構築する。

使い方:
  cd ElbowVision
  source elbow-api/venv/bin/activate

  # サンプル患者リストCSVを生成
  python scripts/batch_multi_patient.py --generate_sample_csv

  # 実行
  python scripts/batch_multi_patient.py \
    --patient_list data/patients.csv \
    --out_dir data/multi_patient_dataset/ \
    --n_angles 91 \
    --n_aug 15

患者リストCSV形式:
  patient_id, ct_dir, laterality, series_num, hu_min, hu_max
  P001, /path/to/ct/P001/, R, 4, 50, 800
  P002, /path/to/ct/P002/, L, None, -400, 1500

出力:
  data/multi_patient_dataset/
  ├── patients/
  │   ├── P001/
  │   │   ├── images/         ← 患者固有DRR画像
  │   │   ├── labels.csv      ← angle_deg GT
  │   │   └── status.json     ← 処理状態（チェックポイント）
  │   └── P002/ ...
  ├── pooled/
  │   ├── images/             ← 全患者統合画像
  │   ├── train.csv           ← train split
  │   └── val.csv             ← val split
  └── progress.json           ← 全体進捗
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# elbow-train をパスに追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    rotate_volume_and_landmarks,
    generate_drr,
)

# ── 設定 ──────────────────────────────────────────────────────────────────────

N_WORKERS   = 10        # M4 Pro 14コア → 10ワーカー
TARGET_SIZE = 256       # CTボリューム処理サイズ
SID_MM      = 1000.0    # X線管焦点-検出器間距離
VAL_RATIO   = 0.15      # 検証セット割合
ANGLE_MIN   = 90.0      # 屈曲角最小（°）
ANGLE_MAX   = 180.0     # 屈曲角最大（°）

# ── グローバル（forkで子プロセスへ共有）──────────────────────────────────────

_g_volume    = None
_g_landmarks = None
_g_lat       = None
_g_voxel_mm  = None
_g_base_flex = None


def _init_worker(volume, landmarks, lat, voxel_mm, base_flex):
    """fork後の子プロセス初期化（グローバルにボリュームをセット）"""
    global _g_volume, _g_landmarks, _g_lat, _g_voxel_mm, _g_base_flex
    _g_volume    = volume
    _g_landmarks = landmarks
    _g_lat       = lat
    _g_voxel_mm  = voxel_mm
    _g_base_flex = base_flex


def _generate_one(args_tuple):
    """1枚のDRR生成（ワーカー関数）"""
    angle_deg, aug_idx, out_path, domain_aug = args_tuple

    # 小さなランダム回転（augmentation）
    rot_delta = random.uniform(-10.0, 10.0) if aug_idx > 0 else 0.0

    try:
        rot_vol, _ = rotate_volume_and_landmarks(
            _g_volume, _g_landmarks,
            forearm_rotation_deg=rot_delta,
            flexion_deg=angle_deg,
            base_flexion=_g_base_flex,
            valgus_deg=0.0,
        )

        drr = generate_drr(rot_vol, axis="LAT", sid_mm=SID_MM, voxel_mm=_g_voxel_mm)

        if domain_aug and aug_idx > 0:
            drr = _apply_domain_aug(drr)

        cv2.imwrite(out_path, drr)
        return True, angle_deg, aug_idx, rot_delta

    except Exception as e:
        return False, angle_deg, aug_idx, str(e)


def _apply_domain_aug(img: np.ndarray) -> np.ndarray:
    """軽量ドメイン適応augmentation（グレースケール画像用）"""
    img = img.astype(np.float32)
    # ガウスノイズ
    img += np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
    # コントラスト・輝度
    img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-15, 15), 0, 255)
    # ガンマ
    gamma = random.uniform(0.75, 1.3)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0
    img = lut[img.astype(np.uint8)]
    # ブラー（40%）
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0).astype(np.float32)
    return img.astype(np.uint8)


# ── 患者1人分の処理 ──────────────────────────────────────────────────────────

@dataclass
class PatientConfig:
    patient_id: str
    ct_dir: str
    laterality: str | None = None
    series_num: int | None = None
    hu_min: float = -400.0
    hu_max: float = 1500.0


def process_patient(
    cfg: PatientConfig,
    out_dir: Path,
    n_angles: int = 91,
    n_aug: int = 15,
    domain_aug: bool = True,
) -> dict:
    """
    患者1人分のDRR生成。

    Returns: {"patient_id", "n_images", "status", "error"}
    """
    patient_dir = out_dir / "patients" / cfg.patient_id
    status_file = patient_dir / "status.json"

    # チェックポイント: 完了済みならスキップ
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get("done"):
            print(f"  [SKIP] {cfg.patient_id} — 完了済み ({status['n_images']} 枚)")
            return status

    img_dir = patient_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n  [{cfg.patient_id}] CT読み込み中 ...")

    try:
        volume, _, lat, voxel_mm = load_ct_volume(
            cfg.ct_dir,
            laterality=cfg.laterality,
            series_num=cfg.series_num,
            hu_min=cfg.hu_min,
            hu_max=cfg.hu_max,
            target_size=TARGET_SIZE,
        )
        landmarks = auto_detect_landmarks(volume, laterality=lat)
    except Exception as e:
        status = {"patient_id": cfg.patient_id, "done": False,
                  "n_images": 0, "status": "error", "error": str(e)}
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False))
        print(f"  [ERROR] {cfg.patient_id}: {e}")
        return status

    # 基底屈曲角（伸展CT想定: 180°）
    base_flexion = 180.0

    # タスクリスト生成: n_angles × n_aug 枚
    angles = np.linspace(ANGLE_MIN, ANGLE_MAX, n_angles).tolist()
    tasks = []
    for angle in angles:
        for aug_i in range(n_aug):
            rot_sign = f"+{0.0:.1f}" if aug_i == 0 else ""
            fname = f"angle{int(round(angle)):03d}_aug{aug_i:02d}_rot{rot_sign}.png"
            out_path = str(img_dir / fname)
            tasks.append((angle, aug_i, out_path, domain_aug))

    print(f"  [{cfg.patient_id}] {len(tasks)} DRR生成中 (workers={N_WORKERS}) ...")

    # fork前にグローバルセット（ゼロコピー共有・pickle不要）
    _init_worker(volume, landmarks, lat, voxel_mm, base_flexion)
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        results_raw = []
        for i, r in enumerate(pool.imap_unordered(_generate_one, tasks, chunksize=10)):
            results_raw.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                print(f"    {i+1}/{len(tasks)}", end="\r")

    print()

    # labels.csv 保存
    rows = []
    n_ok = 0
    for ok, angle, aug_i, extra in results_raw:
        if ok:
            rot_sign = f"+{0.0:.1f}" if aug_i == 0 else ""
            fname = f"angle{int(round(angle)):03d}_aug{aug_i:02d}_rot{rot_sign}.png"
            rows.append({"filename": fname, "angle_deg": angle,
                         "patient_id": cfg.patient_id})
            n_ok += 1

    label_csv = patient_dir / "labels.csv"
    if rows:
        with open(label_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "angle_deg", "patient_id"])
            w.writeheader()
            w.writerows(rows)

    elapsed = time.time() - t0
    status = {
        "patient_id": cfg.patient_id,
        "done": True,
        "n_images": n_ok,
        "n_failed": len(tasks) - n_ok,
        "status": "ok",
        "elapsed_sec": round(elapsed, 1),
    }
    status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(f"  [{cfg.patient_id}] 完了: {n_ok}/{len(tasks)} 枚 ({elapsed:.0f}s)")
    return status


# ── プールデータセット構築 ────────────────────────────────────────────────────

def build_pooled_dataset(out_dir: Path, val_ratio: float = VAL_RATIO) -> None:
    """全患者のlabels.csvを統合してtrain.csv / val.csvを生成する。"""
    all_rows = []
    patients_dir = out_dir / "patients"

    for patient_dir in sorted(patients_dir.iterdir()):
        label_csv = patient_dir / "labels.csv"
        if not label_csv.exists():
            continue
        with open(label_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["img_path"] = str(patient_dir / "images" / row["filename"])
                all_rows.append(row)

    if not all_rows:
        print("  WARNING: データが見つかりません")
        return

    random.shuffle(all_rows)
    n_val = int(len(all_rows) * val_ratio)
    val_rows   = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    pooled_dir = out_dir / "pooled"
    pooled_dir.mkdir(exist_ok=True)

    for split, rows in [("train", train_rows), ("val", val_rows)]:
        csv_path = pooled_dir / f"{split}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["img_path", "angle_deg", "patient_id", "filename"])
            w.writeheader()
            w.writerows(rows)
        print(f"  {split}.csv: {len(rows)} 枚")

    print(f"  合計: {len(all_rows)} 枚 (train={len(train_rows)}, val={len(val_rows)})")


# ── メイン ────────────────────────────────────────────────────────────────────

SAMPLE_CSV_CONTENT = """\
patient_id,ct_dir,laterality,series_num,hu_min,hu_max
P001,data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000,R,4,50,800
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElbowVision マルチ患者バッチDRR生成（本研究向け）"
    )
    parser.add_argument("--patient_list", type=str,
                        help="患者リストCSV (patient_id, ct_dir, laterality, ...)")
    parser.add_argument("--out_dir", type=str,
                        default="data/multi_patient_dataset",
                        help="出力ディレクトリ (default: data/multi_patient_dataset)")
    parser.add_argument("--n_angles", type=int, default=91,
                        help="生成角度数 (default: 91 = 90〜180°, 1°刻み)")
    parser.add_argument("--n_aug", type=int, default=15,
                        help="角度あたりaugmentation数 (default: 15)")
    parser.add_argument("--no_domain_aug", action="store_true",
                        help="ドメイン適応augmentationを無効化")
    parser.add_argument("--generate_sample_csv", action="store_true",
                        help="サンプル患者リストCSVを生成して終了")
    args = parser.parse_args()

    if args.generate_sample_csv:
        sample_path = _PROJECT_ROOT / "data" / "patients.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(SAMPLE_CSV_CONTENT)
        print(f"サンプルCSV生成: {sample_path}")
        print("patient_id, ct_dir, laterality, series_num, hu_min, hu_max を記入してください")
        return

    if not args.patient_list:
        parser.error("--patient_list または --generate_sample_csv が必要です")

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 患者リスト読み込み
    patients: list[PatientConfig] = []
    with open(args.patient_list) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sn = row.get("series_num", "").strip()
            patients.append(PatientConfig(
                patient_id  = row["patient_id"].strip(),
                ct_dir      = row["ct_dir"].strip(),
                laterality  = row.get("laterality", "").strip() or None,
                series_num  = int(sn) if sn and sn.lower() != "none" else None,
                hu_min      = float(row.get("hu_min", -400)),
                hu_max      = float(row.get("hu_max", 1500)),
            ))

    print(f"患者数: {len(patients)}")
    print(f"設定: angles={args.n_angles}, aug={args.n_aug}, "
          f"domain_aug={not args.no_domain_aug}")
    print(f"出力: {out_dir}")
    print(f"{'='*60}")

    # 全体進捗ファイル
    progress_file = out_dir / "progress.json"
    progress: dict = {}
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())

    # 患者ごとに処理
    for i, cfg in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] {cfg.patient_id}")
        result = process_patient(
            cfg, out_dir,
            n_angles=args.n_angles,
            n_aug=args.n_aug,
            domain_aug=not args.no_domain_aug,
        )
        progress[cfg.patient_id] = result
        progress_file.write_text(json.dumps(progress, indent=2, ensure_ascii=False))

    # プールデータセット構築
    print(f"\n{'='*60}")
    print("プールデータセット構築中...")
    build_pooled_dataset(out_dir)

    # サマリー
    done    = sum(1 for r in progress.values() if r.get("done"))
    errors  = sum(1 for r in progress.values() if not r.get("done"))
    n_total = sum(r.get("n_images", 0) for r in progress.values())
    print(f"\n完了: {done}/{len(patients)} 患者, 総DRR数: {n_total}, エラー: {errors}")


if __name__ == "__main__":
    main()
