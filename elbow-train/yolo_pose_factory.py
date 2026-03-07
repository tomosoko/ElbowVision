"""
ElbowVision — YOLOv8-Pose 訓練データ自動生成スクリプト
OsteoVision/OsteoSynth/yolo_pose_factory.py を肘用に移植

【何をするか】
  CT DICOM → DRR（模擬X線）生成 + YOLOキーポイントラベルを自動生成
  LabelStudioでのアノテーション作業が不要になる

【キーポイント定義（4点）】
  0: humerus_shaft       — 上腕骨幹部（近位端中央）
  1: lateral_epicondyle  — 外側上顆
  2: medial_epicondyle   — 内側上顆
  3: forearm_shaft       — 前腕骨幹部（遠位端中央）

【生成するバリエーション】
  - 屈曲角（flexion）: 0, 30, 60, 90, 120°
  - 回内外（pronation/supination）: -30, 0, 30°
  - 撮影角度ゆらぎ（tilt, rot）: ±5°, ±15°
  - ビュー: AP（正面） + LAT（側面）

使い方:
  # CT DICOMが data/raw_dicom/ct/CT_001/ にある場合
  python elbow-train/yolo_pose_factory.py

  # 出力先を指定
  python elbow-train/yolo_pose_factory.py --ct_dir data/raw_dicom/ct/CT_001 --out_dir data/yolo_dataset

  # 合成ファントムでテスト（実CTなしで動作確認）
  python elbow-train/yolo_pose_factory.py --synthetic
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys

import cv2
import numpy as np

try:
    import pydicom
    import pydicom.config
    pydicom.config.enforce_valid_values = False
except ImportError:
    print("pip install pydicom を実行してください")
    sys.exit(1)

try:
    from scipy.ndimage import affine_transform, zoom
    from tqdm import tqdm
except ImportError:
    print("pip install scipy tqdm を実行してください")
    sys.exit(1)

# ct_reorient.py から解剖学的補正関数をインポート
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
try:
    from ct_reorient import (
        load_dicom_series,
        correct_scan_direction,
        detect_humeral_axis,
        detect_transepicondylar_axis,
        build_anatomical_rotation,
        apply_rotation,
    )
    CT_REORIENT_AVAILABLE = True
except ImportError:
    CT_REORIENT_AVAILABLE = False
    print("警告: ct_reorient.py が見つかりません。傾き補正なしで動作します。")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ─── 回転行列 ──────────────────────────────────────────────────────────────────

def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx),  math.cos(rx)]])
    Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz),  math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


# ─── CT 読み込み ───────────────────────────────────────────────────────────────

def load_real_ct_with_landmarks(dicom_dir, size=128):
    """
    実CT DICOMを読み込み、解剖学的座標系に補正してから骨ボリュームを返す。

    ct_reorient.py のパイプラインを使用:
      1. DICOM読み込み + HFS/FFS方向補正
      2. 骨セグメンテーション（HU > 200）
      3. 上腕骨長軸をPCAで検出
      4. 経顆軸を検出（AP/LAT基準面の定義）
      5. 解剖学的座標系に回転補正 → AP/LAT方向が正確に揃った状態でDRR生成可能
    """
    BONE_HU = 200

    if CT_REORIENT_AVAILABLE:
        # ── ct_reorient.py パイプライン ──────────────────────────────────────
        print("  ct_reorient.py で解剖学的補正を実行...")
        volume_raw, info = load_dicom_series(dicom_dir)
        volume_raw = correct_scan_direction(volume_raw, info)

        spacing = (info["z_spacing"],
                   info["pixel_spacing"][0],
                   info["pixel_spacing"][1])
        print(f"  spacing: z={spacing[0]:.2f} y={spacing[1]:.2f} x={spacing[2]:.2f} mm")

        # 等方リサンプリング
        target_sp = min(spacing)
        volume_iso = zoom(volume_raw,
                          (spacing[0] / target_sp, spacing[1] / target_sp, spacing[2] / target_sp),
                          order=1)
        print(f"  等方リサンプリング後: {volume_iso.shape}")

        # 骨抽出
        volume_bone = volume_iso.copy()
        volume_bone[volume_bone < BONE_HU] = 0

        # 上腕骨長軸検出（前腕骨汚染排除済みPCA）
        iso_spacing = (target_sp, target_sp, target_sp)
        humeral_axis, centroid = detect_humeral_axis(volume_bone, BONE_HU, iso_spacing)

        # 経顆軸検出（AP方向の基準）
        trans_axis = detect_transepicondylar_axis(
            volume_bone, humeral_axis, centroid, BONE_HU, iso_spacing)

        # 解剖学的座標系に回転補正
        rot_mat = build_anatomical_rotation(humeral_axis, trans_axis)
        volume_corrected = apply_rotation(volume_bone, rot_mat, centroid)
        print("  解剖学的座標系への補正完了")

    else:
        # ── フォールバック: 手動DICOM読み込み（補正なし） ────────────────────
        print("  警告: ct_reorient.py 未使用（傾き補正なし）")
        dcm_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith(('.dcm', '.dicom'))]
        if not dcm_files:
            raise ValueError(f"DICOMファイルが見つかりません: {dicom_dir}")
        slices = sorted(
            [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in dcm_files],
            key=lambda s: float(getattr(s, 'ImagePositionPatient', [0, 0, 0])[2])
        )
        ref = slices[0]
        slope = float(getattr(ref, 'RescaleSlope', 1.0))
        intercept = float(getattr(ref, 'RescaleIntercept', 0.0))
        volume_hu = np.stack([s.pixel_array.astype(np.float32) * slope + intercept for s in slices])

        row_sp = float(ref.PixelSpacing[0])
        z_sp = float(getattr(ref, 'SliceThickness', 1.0))
        target_sp = min(row_sp, z_sp)
        volume_iso = zoom(volume_hu, (z_sp / target_sp, 1.0, 1.0), order=1)

        volume_corrected = volume_iso.copy()
        volume_corrected[volume_corrected < BONE_HU] = 0

    # 0-255 正規化
    bone_max = np.max(volume_corrected)
    if bone_max > 0:
        volume_corrected = (volume_corrected / bone_max * 255.0).astype(np.float32)

    # 正方形キューブにリサイズ
    volume = zoom(volume_corrected,
                  (size / volume_corrected.shape[0],
                   size / volume_corrected.shape[1],
                   size / volume_corrected.shape[2]),
                  order=1)

    landmarks_3d = _define_elbow_landmarks(size)
    return volume, landmarks_3d


def _define_elbow_landmarks(size):
    """
    正規化キューブ内の肘ランドマーク近似位置。
    Z軸 = 上腕骨長軸（近位端 = Z大, 遠位端 = Z小）
    Y軸 = 前後方向（前方 = Y小, 後方 = Y大）
    X軸 = 内外側方向（外側 = X小, 内側 = X大）
    """
    return {
        "humerus_shaft":      (int(size * 0.78), int(size * 0.50), int(size * 0.50)),
        "lateral_epicondyle": (int(size * 0.35), int(size * 0.50), int(size * 0.30)),
        "medial_epicondyle":  (int(size * 0.35), int(size * 0.50), int(size * 0.70)),
        "forearm_shaft":      (int(size * 0.12), int(size * 0.50), int(size * 0.50)),
    }


# ─── 合成ファントム（実CTなしで動作確認） ──────────────────────────────────────

def create_synthetic_elbow_with_landmarks(size=128):
    """
    合成肘ファントムを生成する。実CTが手に入るまでのテスト用。
    上腕骨（シャフト + 顆部）と前腕骨（シャフト）を模した単純な形状。
    """
    volume = np.zeros((size, size, size), dtype=np.float32)
    bone_val = 1000.0

    cx, cy = int(size * 0.5), int(size * 0.5)

    # 上腕骨シャフト（Z上半分、細い円柱）
    for z in range(int(size * 0.45), size):
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 <= 10**2:
                    volume[z, y, x] = bone_val

    # 顆部（Z 35-48%、横に広がった楕円）
    epicondyle_z_start = int(size * 0.30)
    epicondyle_z_end   = int(size * 0.45)
    lat_cx = int(size * 0.30)
    med_cx = int(size * 0.70)
    for z in range(epicondyle_z_start, epicondyle_z_end):
        for y in range(size):
            for x in range(size):
                if (x - lat_cx)**2 + (y - cy)**2 <= 11**2:
                    volume[z, y, x] = bone_val
                if (x - med_cx)**2 + (y - cy)**2 <= 11**2:
                    volume[z, y, x] = bone_val
                if (x - cx)**2 + (y - cy)**2 <= 10**2:
                    volume[z, y, x] = bone_val

    # 前腕骨シャフト（Z下半分、やや細い円柱）
    for z in range(0, int(size * 0.30)):
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 <= 8**2:
                    volume[z, y, x] = bone_val

    landmarks_3d = _define_elbow_landmarks(size)
    return volume, landmarks_3d


# ─── 投影 ─────────────────────────────────────────────────────────────────────

def project_3d_point_to_2d(point_3d, rot_matrix, center, out_shape, volume_shape):
    """
    3Dランドマーク座標を DRR 上の2D座標に変換（正射影）。
    volume shape: [Z, Y, X]、axis=2 方向に sum して DRR を生成するため
    Z → row（縦）、Y → col（横）に対応。
    """
    p = np.array(point_3d, dtype=np.float64)
    p_centered = p - center
    p_rot = rot_matrix.dot(p_centered) + center
    z, y, _ = p_rot  # X方向（depth）は投影で消える

    pixel_row = z * (out_shape[0] / volume_shape[0])
    pixel_col = y * (out_shape[1] / volume_shape[1])

    return (int(pixel_col), int(pixel_row))  # OpenCV: (x=col, y=row)


def project_volume(volume, rot_matrix, center):
    """CTボリュームを回転して axis=2 方向に投影（Beer-Lambert）。"""
    offset = center - rot_matrix.T.dot(center)
    rotated = affine_transform(volume, rot_matrix.T, offset=offset, order=1, mode='constant', cval=0.0)
    proj = np.sum(rotated, axis=2)
    proj = np.clip(proj, 0, None)
    if np.max(proj) > 0:
        proj = (proj / np.max(proj) * 255.0)
    return proj.astype(np.uint8), rotated


def make_drr_image(proj_raw, out_shape=(512, 512)):
    """投影データを DRR 画像（CLAHE強調済み）に変換。"""
    img = cv2.resize(proj_raw, out_shape, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ─── YOLO ラベル生成 ───────────────────────────────────────────────────────────

def convert_to_yolov8_pose(points_2d, img_w, img_h):
    """
    2D キーポイント座標 → YOLOv8-Pose ラベル文字列。
    キーポイント順: humerus_shaft, lateral_epicondyle, medial_epicondyle, forearm_shaft
    """
    key_order = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle", "forearm_shaft"]

    pts = np.array([points_2d[k] for k in key_order])
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

    pad = 20
    min_x = max(0, min_x - pad)
    max_x = min(img_w, max_x + pad)
    min_y = max(0, min_y - pad)
    max_y = min(img_h, max_y + pad)

    bbox_w  = max_x - min_x
    bbox_h  = max_y - min_y
    bbox_cx = (min_x + bbox_w / 2.0) / img_w
    bbox_cy = (min_y + bbox_h / 2.0) / img_h
    bbox_w  /= img_w
    bbox_h  /= img_h

    line = f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"
    for key in key_order:
        px, py = points_2d[key]
        vis = 2 if 0 <= px < img_w and 0 <= py < img_h else 0
        line += f" {min(max(px/img_w, 0.0), 1.0):.6f} {min(max(py/img_h, 0.0), 1.0):.6f} {vis}"

    return line


# ─── メイン生成ループ ──────────────────────────────────────────────────────────

def run_factory(ct_dir, out_dir, size=128, out_img_size=(512, 512), synthetic=False):
    """
    DRR + YOLOラベルを一括生成する。
    """
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    VAL_RATIO = 0.15

    # CT 読み込み
    if synthetic:
        print("合成ファントムを使用（実CTなし）")
        volume, landmarks_3d = create_synthetic_elbow_with_landmarks(size)
    else:
        print(f"CT読み込み: {ct_dir}")
        try:
            volume, landmarks_3d = load_real_ct_with_landmarks(ct_dir, size)
            print("実CT読み込み成功")
        except Exception as e:
            print(f"CT読み込み失敗: {e}")
            print("合成ファントムにフォールバック")
            volume, landmarks_3d = create_synthetic_elbow_with_landmarks(size)

    # 関節分割 Z 位置（上腕骨遠位端と前腕骨近位端の境界）
    joint_z = int(size * 0.38)

    # ── バリエーション定義 ─────────────────────────────────────────
    # 撮影角度ゆらぎ（体位のズレを模擬）
    tilts = [-5, 0, 5]          # X軸回転（前後傾き）
    rots  = [-15, 0, 15]        # Y軸回転（左右回転）

    # 肘の運動学
    flexions   = [0, 30, 60, 90, 120]  # 屈曲角（°）
    pronations = [-30, 0, 30]          # 回内外（°）: 負=回外, 正=回内

    total = len(tilts) * len(rots) * len(flexions) * len(pronations) * 2  # x2: AP + LAT
    print(f"生成予定: {total} 枚 (AP + LAT x {total//2} 設定)")

    csv_data = []
    count = 0

    center = np.array(volume.shape) / 2.0
    joint_center = np.array([joint_z, size / 2.0, size / 2.0])

    # 上腕骨（近位）と前腕（遠位）に分割
    humerus_vol = np.zeros_like(volume)
    humerus_vol[joint_z:, :, :] = volume[joint_z:, :, :]

    with tqdm(total=total) as pbar:
        for t in tilts:
            for r in rots:
                for flex in flexions:
                    for pro in pronations:

                        # ── 前腕の運動学回転（屈曲 + 回内外） ─────────────
                        forearm_vol = np.zeros_like(volume)
                        forearm_vol[:joint_z, :, :] = volume[:joint_z, :, :]

                        # 屈曲: X軸回転。解剖学的めり込み防止の平行移動を加える
                        flex_matrix = get_rotation_matrix(rx_deg=flex, ry_deg=0, rz_deg=0)
                        anatomical_offset = np.array([-abs(flex) * 0.12, 0, 0])
                        offset_flex = joint_center - flex_matrix.T.dot(joint_center + anatomical_offset)
                        forearm_flexed = affine_transform(
                            forearm_vol, flex_matrix.T, offset=offset_flex, order=1, mode='constant')

                        # 回内外: Y軸回転（前腕長軸まわり）
                        pro_matrix = get_rotation_matrix(rx_deg=0, ry_deg=pro, rz_deg=0)
                        offset_pro = joint_center - pro_matrix.T.dot(joint_center)
                        forearm_moved = affine_transform(
                            forearm_flexed, pro_matrix.T, offset=offset_pro, order=1, mode='constant')

                        # 前腕ランドマークの3D変換（屈曲→回内外の順）
                        def transform_forearm_pt(pt3d):
                            p = np.array(pt3d, dtype=np.float64)
                            p = flex_matrix.dot(p - joint_center) + joint_center + anatomical_offset
                            p = pro_matrix.dot(p - joint_center) + joint_center
                            return p

                        # ── AP / LAT 各ビューを生成 ────────────────────────
                        for view_type, view_ry_offset in [("AP", 0), ("LAT", 90)]:
                            count += 1
                            name_base = f"drr_{view_type}_t{t}_r{r}_f{flex}_p{pro}"
                            split = "val" if hash(name_base) % 100 < (VAL_RATIO * 100) else "train"

                            img_path = os.path.join(out_dir, "images", split, name_base + ".png")
                            lbl_path = os.path.join(out_dir, "labels", split, name_base + ".txt")

                            # グローバル回転行列（カメラ位置）
                            cam_matrix = get_rotation_matrix(rx_deg=t, ry_deg=r + view_ry_offset, rz_deg=0)
                            cam_offset  = center - cam_matrix.T.dot(center)

                            if not os.path.exists(img_path):
                                # 上腕骨と前腕を個別に投影して合成
                                humerus_rot = affine_transform(
                                    humerus_vol, cam_matrix.T, offset=cam_offset, order=1, mode='constant')
                                forearm_rot = affine_transform(
                                    forearm_moved, cam_matrix.T, offset=cam_offset, order=1, mode='constant')

                                proj_raw = np.sum(humerus_rot, axis=2) + np.sum(forearm_rot, axis=2)
                                proj_raw = np.clip(proj_raw, 0, None)
                                if np.max(proj_raw) > 0:
                                    proj_raw = (proj_raw / np.max(proj_raw) * 255.0)

                                # ポアソンノイズ（X線量子ノイズ模擬）
                                noise = np.random.poisson(
                                    np.clip(proj_raw / 255.0 * 100, 0, 100)
                                ) / 100.0 * 255.0
                                proj_sim = cv2.addWeighted(
                                    proj_raw.astype(np.float32), 0.88,
                                    noise.astype(np.float32), 0.12, 0)
                                proj_sim = np.clip(proj_sim, 0, 255).astype(np.uint8)

                                drr_img = make_drr_image(proj_sim, out_img_size)
                                cv2.imwrite(img_path, drr_img)

                            # ── キーポイント投影（ファイル有無に関わらず計算） ──
                            points_2d = {}
                            for name, pt3d in landmarks_3d.items():
                                if name in ("forearm_shaft",):
                                    # 前腕ランドマーク: 運動学変換を適用
                                    pt3d_transformed = transform_forearm_pt(pt3d)
                                else:
                                    pt3d_transformed = np.array(pt3d, dtype=np.float64)

                                # グローバルカメラ回転を適用して2D投影
                                points_2d[name] = project_3d_point_to_2d(
                                    pt3d_transformed, cam_matrix, center,
                                    out_img_size, volume.shape)

                            if not os.path.exists(lbl_path):
                                yolo_txt = convert_to_yolov8_pose(
                                    points_2d, out_img_size[0], out_img_size[1])
                                with open(lbl_path, "w") as f:
                                    f.write(yolo_txt + "\n")

                            csv_data.append({
                                "filename": name_base + ".png",
                                "split": split,
                                "view_type": view_type,
                                "tilt_deg": t,
                                "rot_deg": r,
                                "flexion_deg": flex,
                                "pronation_deg": pro,
                                "humerus_x": points_2d["humerus_shaft"][0],
                                "humerus_y": points_2d["humerus_shaft"][1],
                                "lat_epic_x": points_2d["lateral_epicondyle"][0],
                                "lat_epic_y": points_2d["lateral_epicondyle"][1],
                                "med_epic_x": points_2d["medial_epicondyle"][0],
                                "med_epic_y": points_2d["medial_epicondyle"][1],
                                "forearm_x": points_2d["forearm_shaft"][0],
                                "forearm_y": points_2d["forearm_shaft"][1],
                            })
                            pbar.update(1)

    # CSV サマリ
    csv_path = os.path.join(out_dir, "dataset_summary.csv")
    if HAS_PANDAS:
        import pandas as pd
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    else:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    print(f"CSV保存: {csv_path} ({len(csv_data)} 件)")

    # dataset.yaml 自動生成
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# ElbowVision YOLOv8-Pose Dataset (自動生成)
path: {os.path.abspath(out_dir)}
train: images/train
val:   images/val

kpt_shape: [4, 3]
flip_idx: [0, 2, 1, 3]

nc: 1
names: [elbow]

# キーポイント定義
# 0: humerus_shaft       — 上腕骨幹部（近位端）
# 1: lateral_epicondyle  — 外側上顆
# 2: medial_epicondyle   — 内側上顆
# 3: forearm_shaft       — 前腕骨幹部（遠位端）
""")
    print(f"dataset.yaml 保存: {yaml_path}")

    train_imgs = [d for d in csv_data if d["split"] == "train"]
    val_imgs   = [d for d in csv_data if d["split"] == "val"]
    print(f"\n完了: train {len(train_imgs)} 枚 / val {len(val_imgs)} 枚")
    print(f"出力先: {out_dir}")
    print("\n次のステップ:")
    print("  python elbow-train/train_yolo_pose.py")


# ─── エントリポイント ──────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="ElbowVision DRR + YOLOラベル自動生成")
    parser.add_argument("--ct_dir",    default=os.path.join(base, "data/raw_dicom/ct/CT_001"),
                        help="CT DICOMフォルダ")
    parser.add_argument("--out_dir",   default=os.path.join(base, "data/yolo_dataset"),
                        help="出力先フォルダ")
    parser.add_argument("--size",      type=int, default=128, help="CTボリュームキューブサイズ")
    parser.add_argument("--img_size",  type=int, default=512, help="DRR画像サイズ（正方形）")
    parser.add_argument("--synthetic", action="store_true",
                        help="実CTなしで合成ファントムを使って動作確認")
    args = parser.parse_args()

    run_factory(
        ct_dir=args.ct_dir,
        out_dir=args.out_dir,
        size=args.size,
        out_img_size=(args.img_size, args.img_size),
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
