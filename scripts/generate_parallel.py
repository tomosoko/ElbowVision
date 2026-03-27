"""
12コア並列DRR生成スクリプト（M4 Pro 64GB最適化）

ボリュームをメインプロセスで1回読み込み、fork で子プロセスに共有。
各ワーカーが独立にDRRを生成・保存する。
"""
import os
import sys
import csv
import random
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "elbow-train"))
from elbow_synth import (
    load_ct_volume, auto_detect_landmarks,
    rotate_volume_and_landmarks, generate_drr, make_yolo_label,
)

# ── 設定 ──────────────────────────────────────────────────────────────────────

CT_DIR    = "data/raw_dicom/ct_all/"
OUT_DIR   = "data/yolo_dataset_v3"
LATERALITY = "L"
SERIES     = [(3, 180.0), (7, 135.0), (11, 90.0)]
HU_MIN, HU_MAX = 50, 800
TARGET_SIZE = 512
SID_MM     = 1000.0
N_AP       = 600
N_LAT      = 600
VAL_RATIO  = 0.15
DOMAIN_AUG = True
N_WORKERS  = 10  # M4 Pro 12コア → 10ワーカー

# ── グローバル変数（fork で子プロセスに共有） ────────────────────────────────

_volumes = {}  # {view: [(volume, landmarks, laterality, voxel_mm, base_flexion), ...]}
_real_cdfs = []


def init_real_cdfs():
    """実X線のCDFをロード（ヒストグラムマッチング用）"""
    global _real_cdfs
    real_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "real_xray", "images")
    for f in ["008_AP.png", "008_LAT.png"]:
        p = os.path.join(real_dir, f)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            h = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
            cdf = h.cumsum()
            cdf = cdf / (cdf[-1] + 1e-8)
            _real_cdfs.append(cdf)


def apply_domain_aug(drr_bgr):
    """DRR画像に実X線らしさを付与するaugmentation"""
    img = drr_bgr.copy().astype(np.float32)
    noise = np.random.normal(0, random.uniform(3, 12), img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    alpha = random.uniform(0.75, 1.25)
    beta = random.uniform(-20, 20)
    img = np.clip(alpha * img + beta, 0, 255)
    gamma = random.uniform(0.7, 1.4)
    inv_gamma = 1.0 / gamma
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
    img = lut[img.astype(np.uint8)]
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)
    # ヒストグラムマッチング（50%）
    if random.random() < 0.5 and _real_cdfs:
        ref_cdf = random.choice(_real_cdfs)
        img_u8 = img.astype(np.uint8)
        src_hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).ravel()
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / (src_cdf[-1] + 1e-8)
        lut_hm = np.zeros(256, dtype=np.uint8)
        for s in range(256):
            lut_hm[s] = np.argmin(np.abs(ref_cdf - src_cdf[s]))
        img = lut_hm[img_u8].astype(np.float32)
    return img.astype(np.uint8)


def generate_one_sample(args_tuple):
    """1枚のDRRを生成（ワーカー関数）"""
    idx, view, vol_idx, rotation_err, flexion, valgus_deg, base_flexion, split = args_tuple

    vols = _volumes[view]
    vd = vols[vol_idx % len(vols)]
    volume, landmarks, lat, voxel_mm, bf = vd

    rot_vol, rot_lm = rotate_volume_and_landmarks(
        volume, landmarks, rotation_err, flexion,
        base_flexion=bf, valgus_deg=valgus_deg,
    )

    drr = generate_drr(rot_vol, axis=view, sid_mm=SID_MM, voxel_mm=voxel_mm)
    drr_bgr = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)

    if DOMAIN_AUG:
        drr_bgr = apply_domain_aug(drr_bgr)

    label = make_yolo_label(rot_lm, view, drr.shape[0], drr.shape[1],
                             vol_shape=rot_vol.shape, sid_mm=SID_MM, voxel_mm=voxel_mm)

    fname = f"elbow_{idx:05d}"
    cv2.imwrite(os.path.join(OUT_DIR, "images", split, f"{fname}.png"), drr_bgr)
    with open(os.path.join(OUT_DIR, "labels", split, f"{fname}.txt"), "w") as f:
        f.write(label)

    return {
        "filename": f"{fname}.png",
        "split": split,
        "view_type": view,
        "rotation_error_deg": round(rotation_err, 2),
        "flexion_deg": round(flexion, 2),
        "base_flexion": base_flexion,
        "carrying_angle": 0.0,
        "valgus_deg": round(valgus_deg, 2),
    }


def main():
    global _volumes

    print(f"DRR並列生成 (workers={N_WORKERS}, target_size={TARGET_SIZE})")
    print(f"{'='*60}")

    # ディレクトリ作成
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    # ── ボリューム読み込み（メインプロセスで1回だけ）──
    print("\nLoading CT volumes...")
    all_vols = []
    for sn, bf in SERIES:
        print(f"  Series {sn} (base_flexion={bf}°)...")
        vol, _, lat, vox_mm = load_ct_volume(
            CT_DIR, laterality=LATERALITY, series_num=sn,
            hu_min=HU_MIN, hu_max=HU_MAX, target_size=TARGET_SIZE,
        )
        lm = auto_detect_landmarks(vol, laterality=lat)
        all_vols.append((vol, lm, lat, vox_mm, bf))

    # AP/LAT 用ボリューム分類
    ap_vols = [v for v in all_vols if v[4] >= 150.0]
    if not ap_vols:
        ap_vols = [max(all_vols, key=lambda x: x[4])]
    lat_vols = [v for v in all_vols if v[4] <= 120.0]
    if not lat_vols:
        lat_vols = [min(all_vols, key=lambda x: x[4])]

    _volumes["AP"] = ap_vols
    _volumes["LAT"] = lat_vols

    # 実X線CDF読み込み
    init_real_cdfs()

    # ── タスク生成 ──
    tasks = []
    idx = 0

    # AP tasks
    print(f"\nPreparing {N_AP} AP tasks...")
    for i in range(N_AP):
        rotation_err = random.uniform(-25.0, 25.0)
        valgus_deg = random.uniform(-10.0, 10.0)
        bf = ap_vols[i % len(ap_vols)][4]
        flex_center = max(150.0, min(180.0, bf))
        flexion = max(150.0, min(180.0, random.uniform(flex_center - 10.0, flex_center + 10.0)))
        split = "val" if i < int(N_AP * VAL_RATIO) else "train"
        tasks.append((idx, "AP", i % len(ap_vols), rotation_err, flexion, valgus_deg, bf, split))
        idx += 1

    # LAT tasks
    print(f"Preparing {N_LAT} LAT tasks...")
    for i in range(N_LAT):
        rotation_err = random.uniform(-30.0, 30.0)
        valgus_deg = random.uniform(-8.0, 8.0)
        bf = lat_vols[i % len(lat_vols)][4]
        flex_center = max(60.0, min(120.0, bf))
        flexion = max(60.0, min(120.0, random.uniform(flex_center - 20.0, flex_center + 20.0)))
        split = "val" if i < int(N_LAT * VAL_RATIO) else "train"
        tasks.append((idx, "LAT", i % len(lat_vols), rotation_err, flexion, valgus_deg, bf, split))
        idx += 1

    # ── 並列実行 ──
    print(f"\nGenerating {len(tasks)} DRRs with {N_WORKERS} workers...")
    random.shuffle(tasks)  # 負荷分散

    # fork で子プロセスにボリュームを共有
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_one_sample, tasks)):
            results.append(result)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] generated")

    # ── 出力ファイル ──
    results.sort(key=lambda r: r["filename"])

    # dataset.yaml
    yaml_path = os.path.join(OUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("nc: 1\nnames: [elbow_joint]\n")
        f.write("kpt_shape: [6, 3]\nflip_idx: [0, 2, 1, 3, 4, 5]\n")

    # CSV
    csv_path = os.path.join(OUT_DIR, "dataset_summary.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    # ConvNeXt CSV
    convnext_fields = ["filename", "split", "view_type", "rotation_error_deg",
                       "flexion_deg", "carrying_angle", "valgus_deg"]
    convnext_path = os.path.join(OUT_DIR, "convnext_labels.csv")
    with open(convnext_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=convnext_fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in convnext_fields})

    n_train = sum(1 for r in results if r["split"] == "train")
    n_val = sum(1 for r in results if r["split"] == "val")
    print(f"\n完了: {len(results)} DRRs (train={n_train}, val={n_val})")
    print(f"  → {OUT_DIR}/")


if __name__ == "__main__":
    main()
