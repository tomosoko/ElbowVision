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
    compute_carrying_angle, compute_flexion_angle,
)

# ── 設定 ──────────────────────────────────────────────────────────────────────

CT_DIR    = "data/raw_dicom/ct_volume/ﾃｽﾄ 008_0009900008_20260310_108Y_F_000"
OUT_DIR   = "data/yolo_dataset_v6"
LATERALITY = "L"  # ファントムは左腕
SERIES     = [(4, 180.0), (8, 135.0), (12, 90.0)]  # FC85骨カーネル
HU_MIN, HU_MAX = 50, 1000  # 骨等価樹脂P95=972HUをカバー
TARGET_SIZE = 256
SID_MM     = 1000.0  # CR実X線のSIDと一致
N_AP       = 600
N_LAT      = 3400  # v4同等
VAL_RATIO  = 0.15
DOMAIN_AUG = True
N_WORKERS  = 10  # M4 Pro 12コア → 10ワーカー（OS+他に余裕）

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
    """DRR画像に実X線らしさを付与するaugmentation（elbow_synth.py改善版と同期）"""
    img = drr_bgr.copy().astype(np.float32)
    # 0a. 回転アーティファクト均一化ブラー（60%）
    if random.random() < 0.6:
        sigma = random.uniform(0.5, 1.0)
        img = cv2.GaussianBlur(img.astype(np.uint8), (0, 0), sigma).astype(np.float32)
    # 0b. 骨テクスチャ合成（50%）
    if random.random() < 0.5:
        h, w = img.shape[:2] if img.ndim == 3 else (img.shape[0], img.shape[1])
        texture = np.zeros((h, w), dtype=np.float32)
        for scale in [16, 32, 64]:
            noise_small = np.random.randn(max(1, h // scale), max(1, w // scale)).astype(np.float32)
            noise_up = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
            texture += noise_up / (scale ** 0.5)
        gray = img[:, :, 0] if img.ndim == 3 else img
        bone_mask = (gray > np.percentile(gray, 60)).astype(np.float32)
        bone_mask = cv2.GaussianBlur(bone_mask, (5, 5), 2.0)
        strength = random.uniform(4.0, 12.0)
        texture_masked = texture * bone_mask * strength
        if img.ndim == 3:
            img += texture_masked[:, :, None]
        else:
            img += texture_masked
        img = np.clip(img, 0, 255)
    # 1. ガウスノイズ
    noise = np.random.normal(0, random.uniform(3, 12), img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    # 2. コントラスト・輝度（実X線に近づけるため控えめ）
    alpha = random.uniform(0.70, 1.15)
    beta = random.uniform(-20, 20)
    img = np.clip(alpha * img + beta, 0, 255)
    # 3. ガンマ補正
    gamma = random.uniform(0.7, 1.4)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0
    img = lut[img.astype(np.uint8)]
    # 4. ブラー（40%）
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)
    # 5. ヒストグラムマッチング（50%）
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

    # LAT像: base_flexion<=120の場合、AP軸投影がLAT像に一致
    if view == "LAT" and base_flexion <= 120:
        proj_axis = "AP"
    else:
        proj_axis = view

    # AP投影(LAT像)でML方向が不足する場合、左側にパディングして正方形DRRを生成
    # 屈曲腕の前腕がML負方向に伸びて画像外に出ないようにする
    if proj_axis == "AP":
        NP, NA, NM = rot_vol.shape
        if NM < NP:
            pad_left = NP - NM  # 正方形にするためのパディング量
            rot_vol_drr = np.pad(rot_vol, [(0, 0), (0, 0), (pad_left, 0)], mode='constant')
            # ランドマークのML正規化座標を更新（パディング後のNMで再スケール）
            NM_new = NP
            rot_lm_drr = {
                k: (v[0], v[1], (v[2] * NM + pad_left) / NM_new)
                for k, v in rot_lm.items()
            }
            vol_shape_drr = rot_vol_drr.shape
        else:
            rot_vol_drr, rot_lm_drr, vol_shape_drr = rot_vol, rot_lm, rot_vol.shape
    else:
        rot_vol_drr, rot_lm_drr, vol_shape_drr = rot_vol, rot_lm, rot_vol.shape

    drr = generate_drr(rot_vol_drr, axis=proj_axis, sid_mm=SID_MM, voxel_mm=voxel_mm)
    drr_bgr = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)

    if DOMAIN_AUG:
        drr_bgr = apply_domain_aug(drr_bgr)

    label = make_yolo_label(rot_lm_drr, proj_axis, drr.shape[0], drr.shape[1],
                             vol_shape=vol_shape_drr, sid_mm=SID_MM, voxel_mm=voxel_mm,
                             view_type=view)

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

    # AP用: 最も伸展したボリューム（180°）
    ap_vols = [max(all_vols, key=lambda x: x[4])]
    # LAT用: 最も屈曲したボリューム（90°）のみ
    # → 投影軸をAP固定で統一し、YOLOの学習を安定化
    # （異なるボリュームでは投影軸がAP/LAT切り替わり、
    #   画像の見た目が全く変わるため混在は訓練不安定の原因）
    lat_vols = [min(all_vols, key=lambda x: x[4])]

    _volumes["AP"] = ap_vols
    _volumes["LAT"] = lat_vols

    print(f"  AP volumes: base_flexion={[v[4] for v in ap_vols]}")
    print(f"  LAT volumes: base_flexion={[v[4] for v in lat_vols]} (投影軸統一)")

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

    # LAT tasks — 90°ボリュームから60-120°を生成（投影軸=AP固定）
    print(f"Preparing {N_LAT} LAT tasks (90° vol, range 60-120°)...")
    for i in range(N_LAT):
        rotation_err = random.uniform(-30.0, 30.0)
        valgus_deg = random.uniform(-8.0, 8.0)
        bf = lat_vols[0][4]  # 90°
        flex_center = max(60.0, min(120.0, bf))
        flexion = max(60.0, min(120.0, random.uniform(flex_center - 20.0, flex_center + 20.0)))
        split = "val" if i < int(N_LAT * VAL_RATIO) else "train"
        tasks.append((idx, "LAT", 0, rotation_err, flexion, valgus_deg, bf, split))
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
