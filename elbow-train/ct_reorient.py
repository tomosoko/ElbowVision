"""
CT ボリューム 自動向き補正スクリプト（肘対応版）

【設計思想】
  DRRを正確に生成するには、スキャナ座標系ではなく
  「肘関節固有の解剖学的座標系」に基づいて投影する必要がある。

  患者CT・ファントムCTで向きがバラバラになる原因:
    1. インスキャン（HFS: Head First Supine）↔ アウトスキャン（FFS: Feet First Supine）
       → z軸の方向が180°反転する
    2. 前腕の回内・回外（内側向き・外側向き）
       → 上腕骨長軸まわりの回転
    3. 肘の傾き（テーブル上で腕が斜め）

  【座標系の定義手順（これが核心）】
  Step 1: 上腕骨長軸（Humeral Axis）を求める
           → 骨ボクセルのPCA第1主成分
           → この軸をZ軸に揃える

  Step 2: 経顆軸（Transepicondylar Axis）を求める
           外側上顆（Lateral Epicondyle）〜 内側上顆（Medial Epicondyle）
           を結ぶ線 = 肘の屈曲軸（Flexion Axis）
           → この軸をX軸（AP方向）に揃える

  Step 3: AP軸 × 長軸 = LAT軸（外積で自動決定）

  この3軸が決まると:
    - 長軸（Z）に垂直な面に投影 → 軸位像
    - AP軸（X）方向に投影 → 正面像（外反角が見える）
    - LAT軸（Y）方向に投影 → 側面像（屈曲角が見える）

使い方:
  # 向きの確認だけ
  python ct_reorient.py --input data/raw_dicom/ct/CT_001/ --preview

  # DRR生成（AP・LAT・回転バリエーション）
  python ct_reorient.py --input data/raw_dicom/ct/CT_001/ --output data/ct_drr/CT_001/

  # 経顆軸を手動で指定（自動検出がうまくいかない場合）
  python ct_reorient.py --input data/raw_dicom/ct/CT_001/ \\
    --epic_lat "120,80,200" --epic_med "120,80,310"
    # スライス番号, Y座標, X座標 を直接入力

  # 複数の回転角でバリエーション生成
  python ct_reorient.py --input data/raw_dicom/ct/CT_001/ \\
    --output data/ct_drr/CT_001/ --rotations 0,30,60,-30,-60
"""
import argparse
import os
import sys

import numpy as np

try:
    import pydicom
    import pydicom.config
    pydicom.config.enforce_valid_values = False
except ImportError:
    print("pip install pydicom を実行してください")
    sys.exit(1)

try:
    from scipy.ndimage import affine_transform, label as nd_label
    from scipy.spatial.transform import Rotation
except ImportError:
    print("pip install scipy を実行してください")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("pip install opencv-python-headless を実行してください")
    sys.exit(1)


# ─── DICOM シリーズ読み込み ────────────────────────────────────────────────────

def load_dicom_series(directory: str) -> tuple[np.ndarray, dict]:
    """
    DICOMシリーズをz座標でソートして3Dボリュームとして読み込む。
    HFS（インスキャン）・FFS（アウトスキャン）を自動検出する。
    """
    files = []
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
                files.append((z, path, ds))
        except Exception as e:
            print(f"[ct_reorient] DICOMスキップ {fname}: {e}", file=sys.stderr)
            continue

    if not files:
        raise ValueError(f"DICOMシリーズが見つかりません: {directory}")

    files.sort(key=lambda x: x[0])
    ds0 = files[0][2]

    patient_position = str(getattr(ds0, "PatientPosition", "HFS")).upper()
    is_feet_first    = "FFS" in patient_position or "FFP" in patient_position
    print(f"  PatientPosition: {patient_position}  "
          f"({'アウトスキャン(FFS) → z軸反転が必要' if is_feet_first else 'インスキャン(HFS)'})")

    iop = None
    if hasattr(ds0, "ImageOrientationPatient"):
        iop = [float(v) for v in ds0.ImageOrientationPatient]

    print(f"  {len(files)} スライスを読み込み中...")
    slices = []
    for _, path, _ in files:
        ds = pydicom.dcmread(path)
        pixel = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slices.append(pixel * slope + intercept)

    volume = np.stack(slices, axis=0)   # (Z, Y, X)

    pixel_spacing = [float(v) for v in ds0.PixelSpacing]
    z_spacing     = abs(float(files[1][0]) - float(files[0][0])) if len(files) > 1 \
                    else float(getattr(ds0, "SliceThickness", 1.0))

    return volume, {
        "pixel_spacing":    pixel_spacing,
        "z_spacing":        z_spacing,
        "shape":            volume.shape,
        "patient_position": patient_position,
        "is_feet_first":    is_feet_first,
        "iop":              iop,
    }


# ─── Step 1 : インスキャン/アウトスキャン補正 ──────────────────────────────────

def correct_scan_direction(volume: np.ndarray, info: dict) -> np.ndarray:
    """
    アウトスキャン（FFS）はz方向が逆になっているため反転して統一する。
    これをやらないと後のPCAで主軸の向きが逆になる可能性がある。
    """
    if info["is_feet_first"]:
        print("  アウトスキャン検出 → z軸を反転")
        return volume[::-1].copy()
    return volume


# ─── Step 2 : 上腕骨長軸の検出（PCA） ────────────────────────────────────────

def detect_humeral_axis(volume: np.ndarray,
                        hu_threshold: float,
                        voxel_spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    骨ボクセルのPCAで上腕骨長軸（第1主成分）を求める。

    【改善点】前腕骨汚染の排除
      肘CTには上腕骨・橈骨・尺骨が混在する。全ボクセルでPCAすると
      前腕の屈曲方向に軸が引っ張られて誤差が生じる（屈曲30°で~10°誤差）。

      対策: 顆部Z位置（骨横断面積が最大のスライス）でボリュームを分割し、
            近位側・遠位側それぞれでPCAを実行。
            寄与率が高い側（= より直線的 = シャフト側）を上腕骨長軸として採用する。
    """
    mask   = volume > hu_threshold
    n_bone = int(mask.sum())
    print(f"  骨ボクセル数: {n_bone:,} ({n_bone / mask.size * 100:.1f}%)")

    if n_bone < 200:
        print(f"  警告: 骨ボクセルが少なすぎます。--hu を下げてください（現在: {hu_threshold}）")
        return np.array([1.0, 0.0, 0.0]), np.array(volume.shape, dtype=float) / 2

    dz, dy, dx = voxel_spacing
    nz         = volume.shape[0]

    # 全体重心（回転の支点として使う）
    all_coords    = np.array(np.where(mask)).T.astype(float)
    centroid_vx   = all_coords.mean(axis=0)

    # 顆部Z位置 = 骨横断面積が最大のスライス
    slice_areas = mask.sum(axis=(1, 2))
    condyle_z   = int(np.argmax(slice_areas))
    print(f"  顆部Z位置: {condyle_z} スライス")

    def _pca(slice_range):
        """指定Zスライス範囲の骨ボクセルでPCAし (軸, 寄与率) を返す"""
        sub = np.zeros_like(mask)
        sub[slice_range] = mask[slice_range]
        coords = np.array(np.where(sub)).T.astype(float)
        if len(coords) < 100:
            return None, 0.0
        if len(coords) > 50000:
            coords = coords[np.random.choice(len(coords), 50000, replace=False)]
        coords_mm   = coords * np.array([dz, dy, dx])
        centroid_mm = coords_mm.mean(axis=0)
        cov         = np.cov((coords_mm - centroid_mm).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_max = np.argmax(eigenvalues)
        return eigenvectors[:, idx_max], eigenvalues[idx_max] / eigenvalues.sum()

    axis_prox, score_prox = _pca(slice(0, condyle_z))
    axis_dist, score_dist = _pca(slice(condyle_z + 1, nz))

    # 寄与率が高い側（直線的 = シャフト）を採用
    shaft_is_proximal = True   # シャフトが顆部より近位（z < condyle_z）側かどうか
    if axis_prox is not None and score_prox >= score_dist:
        humeral_axis, explained = axis_prox, score_prox
        shaft_is_proximal = True
        print(f"  近位側をシャフトとして採用（寄与率 {score_prox:.1%} vs 遠位 {score_dist:.1%}）")
    elif axis_dist is not None:
        humeral_axis, explained = axis_dist, score_dist
        shaft_is_proximal = False
        print(f"  遠位側をシャフトとして採用（寄与率 {score_dist:.1%} vs 近位 {score_prox:.1%}）")
    else:
        # フォールバック: 全ボクセル
        coords_mm   = all_coords * np.array([dz, dy, dx])
        centroid_mm = coords_mm.mean(axis=0)
        cov         = np.cov((coords_mm - centroid_mm).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_max      = np.argmax(eigenvalues)
        humeral_axis = eigenvectors[:, idx_max]
        explained    = eigenvalues[idx_max] / eigenvalues.sum()
        print("  警告: 近位・遠位ともに骨ボクセル不足 → 全ボクセルで PCA")

    # ── 符号の正規化（Sign Canonicalization）────────────────────────────────
    # PCA は軸の方向を決めるが符号は保証しない。そのままだと患者ごとに
    # DRR が上下逆になるため、「上腕骨長軸は常にシャフト方向（近位）を向く」
    # と定義して符号を強制する。
    #   shaft_is_proximal=True  → シャフトは condyle_z より低い z インデックス側
    #                              = 近位方向 = z 減少方向 → axis[0] < 0 に揃える
    #   shaft_is_proximal=False → シャフトは condyle_z より高い z インデックス側
    #                              → axis[0] > 0 に揃える
    if shaft_is_proximal:
        if humeral_axis[0] > 0:
            humeral_axis = -humeral_axis
            print("  符号補正: 近位（シャフト）方向に反転（axis[0] を負に）")
    else:
        if humeral_axis[0] < 0:
            humeral_axis = -humeral_axis
            print("  符号補正: 近位（シャフト）方向に反転（axis[0] を正に）")

    print(f"  上腕骨長軸（PCA）: [{humeral_axis[0]:+.3f}, {humeral_axis[1]:+.3f}, {humeral_axis[2]:+.3f}]")
    print(f"  第1主成分 寄与率: {explained:.1%}")

    z_axis   = np.array([1.0, 0.0, 0.0])
    dot      = abs(float(np.dot(humeral_axis / np.linalg.norm(humeral_axis), z_axis)))
    tilt_deg = float(np.degrees(np.arccos(np.clip(dot, 0, 1))))
    print(f"  スキャナz軸からの傾き: {tilt_deg:.1f}°")

    return humeral_axis, centroid_vx


# ─── Step 3 : 経顆軸の検出（AP/LAT基準面の定義） ────────────────────────────

def detect_transepicondylar_axis(volume: np.ndarray,
                                  humeral_axis: np.ndarray,
                                  centroid: np.ndarray,
                                  hu_threshold: float,
                                  voxel_spacing: tuple,
                                  epic_lat_manual: np.ndarray = None,
                                  epic_med_manual: np.ndarray = None) -> np.ndarray:
    """
    経顆軸（外側上顆〜内側上顆を結ぶ線）を推定する。

    【手動指定が最も確実】
      CT画像上で外側上顆・内側上顆の座標を読み取り、
      --epic_lat と --epic_med オプションで渡す。

    【自動検出（粗い近似）】
      上腕骨長軸と垂直な断面（顆部レベル）で骨の「横幅」が最大になる
      スライスを見つけ、そのスライスでの骨幅の方向を経顆軸とする。
      精度はやや低いが、方向の大まかな推定には使える。

    返り値: 経顆軸ベクトル（正規化済み, [z,y,x]系）
    """
    if epic_lat_manual is not None and epic_med_manual is not None:
        axis = epic_lat_manual - epic_med_manual
        axis = axis / np.linalg.norm(axis)
        print(f"  経顆軸（手動）: [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
        return axis

    # 自動: 2段階アプローチで経顆軸を推定
    #
    # 【設計方針】
    #   手法A（condyle PCA）: 顆部領域の骨ボクセルを上腕骨長軸に垂直な平面に投影し PCA
    #     → 屈曲 0°〜60° 程度まで有効。寄与率で品質を判定できる。
    #
    #   手法B（外積法）: 上腕骨長軸 × 前腕骨軸 = 経顆軸
    #     → 上顆と前腕骨が連結しているため連結成分では分離できない高屈曲時に有効。
    #        condyle PCA の寄与率が低い（前腕骨汚染が大きい）ときに自動切替。
    print("  経顆軸を自動検出中...")
    mask   = volume > hu_threshold
    nz, ny, nx = volume.shape
    dz, dy, dx = voxel_spacing
    h = humeral_axis / np.linalg.norm(humeral_axis)

    # 顆部領域を特定（骨横断面積が最大付近）
    slice_areas = mask.sum(axis=(1, 2)).astype(float)
    max_area    = slice_areas.max()
    if max_area == 0:
        print("  警告: 骨ボクセルが見つかりません。デフォルト軸を使用します。")
        return np.array([0.0, 0.0, 1.0])

    condyle_slices = np.where(slice_areas >= max_area * 0.70)[0]
    if len(condyle_slices) == 0:
        condyle_slices = np.array([int(np.argmax(slice_areas))])
    condyle_z = int(condyle_slices.mean())
    print(f"  顆部領域: z={condyle_slices[0]}〜{condyle_slices[-1]} (中心={condyle_z})")

    # ── 手法A: condyle PCA ──────────────────────────────────────────────────
    sub_mask = np.zeros_like(mask)
    sub_mask[condyle_slices] = mask[condyle_slices]

    # 3D 連結成分解析 → 最大 blob（上顆本体）のみ使用
    labeled, n_components = nd_label(sub_mask)
    if n_components > 1:
        comp_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
        largest_label = int(np.argmax(comp_sizes)) + 1
        sub_mask = (labeled == largest_label)
        print(f"  連結成分: {n_components} → 最大blob を使用 (他 {n_components-1} 成分を除去)")

    coords = np.array(np.where(sub_mask)).T.astype(float)
    condyle_explained = 0.0
    axis_condyle = None
    if len(coords) >= 10:
        coords_mm   = coords * np.array([dz, dy, dx])
        centroid_mm = coords_mm.mean(axis=0)
        proj        = (coords_mm - centroid_mm) @ h
        coords_perp = (coords_mm - centroid_mm) - np.outer(proj, h)
        cov         = np.cov(coords_perp.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_max = np.argmax(eigenvalues)
        axis_condyle    = eigenvectors[:, idx_max] / np.linalg.norm(eigenvectors[:, idx_max])
        condyle_explained = eigenvalues[idx_max] / eigenvalues.sum()
        print(f"  手法A condyle PCA 寄与率: {condyle_explained:.1%}")

    # ── 手法B: 上腕骨長軸 × 前腕骨軸（外積法）──────────────────────────────
    # 顆部より遠位側 = 前腕骨のみ（PCAで前腕長軸を取得）
    axis_cross = None
    dist_mask  = np.zeros_like(mask)
    dist_mask[condyle_z + 1:] = mask[condyle_z + 1:]
    dist_coords = np.array(np.where(dist_mask)).T.astype(float)
    if len(dist_coords) >= 100:
        if len(dist_coords) > 50000:
            dist_coords = dist_coords[np.random.choice(len(dist_coords), 50000, replace=False)]
        dc_mm = dist_coords * np.array([dz, dy, dx])
        dc_mm -= dc_mm.mean(axis=0)
        cov   = np.cov(dc_mm.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        forearm_axis = eigenvectors[:, np.argmax(eigenvalues)]
        cross = np.cross(h, forearm_axis / np.linalg.norm(forearm_axis))
        if np.linalg.norm(cross) > 0.1:   # 前腕が上腕と平行でない場合のみ有効
            axis_cross = cross / np.linalg.norm(cross)
            dot_fa = abs(float(np.dot(h, forearm_axis / np.linalg.norm(forearm_axis))))
            flex_est = float(np.degrees(np.arccos(np.clip(dot_fa, 0, 1))))
            print(f"  手法B 推定屈曲角: {flex_est:.1f}°")

    # ── 手法選択: condyle PCA 寄与率 ≥ 0.75 → 手法A, それ以外 → 手法B ────
    QUALITY_THRESHOLD = 0.75
    if axis_condyle is not None and condyle_explained >= QUALITY_THRESHOLD:
        axis = axis_condyle
        print(f"  採用: condyle PCA（寄与率 {condyle_explained:.1%} ≥ {QUALITY_THRESHOLD:.0%}）")
    elif axis_cross is not None:
        axis = axis_cross
        print(f"  採用: 外積法（condyle PCA 寄与率 {condyle_explained:.1%} < {QUALITY_THRESHOLD:.0%} のため切替）")
    elif axis_condyle is not None:
        axis = axis_condyle
        print(f"  採用: condyle PCA（外積法が使用不可のためフォールバック）")
    else:
        axis = np.array([0.0, 0.0, 1.0])
        print("  警告: 全手法が失敗。デフォルト軸を使用します。")

    print(f"  経顆軸: [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
    print(f"  ※ 精度が低い場合は --epic_lat, --epic_med で手動指定してください")

    return axis


# ─── Step 4 : 解剖学的座標系の構築と回転 ──────────────────────────────────────

def build_anatomical_rotation(humeral_axis: np.ndarray,
                               transepicondylar_axis: np.ndarray) -> np.ndarray:
    """
    肘の解剖学的座標系を構築して、ワールド座標系に変換する回転行列を返す。

    定義:
      Z軸 = 上腕骨長軸（humeral_axis）→ 長軸方向
      X軸 = 経顆軸（transepicondylar_axis）→ AP方向（外反角が見える方向）
      Y軸 = Z × X の外積（外積で自動決定）→ LAT方向（屈曲角が見える方向）

    出力: 3×3 回転行列
      この行列でボリュームを変換すると解剖学的AP/LATの向きが揃う。
    """
    # 各軸を正規化
    z_hat = humeral_axis / np.linalg.norm(humeral_axis)

    # 経顆軸をhumeral_axisと直交する成分だけ残す（グラム・シュミット）
    x_raw = transepicondylar_axis / np.linalg.norm(transepicondylar_axis)
    x_hat = x_raw - np.dot(x_raw, z_hat) * z_hat
    x_hat = x_hat / np.linalg.norm(x_hat)

    # Y軸（LAT方向）= Z × X
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    # 回転行列（列に基底ベクトルを並べる）
    rot_mat = np.column_stack([z_hat, x_hat, y_hat])  # (z,y,x)系なのでZYXの順に基底

    print(f"  解剖学的座標系:")
    print(f"    Z（長軸）= [{z_hat[0]:+.3f}, {z_hat[1]:+.3f}, {z_hat[2]:+.3f}]")
    print(f"    X（AP方向・経顆軸）= [{x_hat[0]:+.3f}, {x_hat[1]:+.3f}, {x_hat[2]:+.3f}]")
    print(f"    Y（LAT方向）= [{y_hat[0]:+.3f}, {y_hat[1]:+.3f}, {y_hat[2]:+.3f}]")

    return rot_mat


def apply_rotation(volume: np.ndarray,
                   rot_mat: np.ndarray,
                   centroid: np.ndarray) -> np.ndarray:
    """ボリュームを重心周りに回転行列で変換する"""
    inv_rot = rot_mat.T   # 直交行列なので転置 = 逆行列
    offset  = centroid - inv_rot @ centroid
    return affine_transform(
        volume, matrix=inv_rot, offset=offset,
        output_shape=volume.shape, order=1, mode="constant", cval=-1000.0,
    )


# ─── Step 5 : 骨軸まわり回転（回内外バリエーション） ───────────────────────────

def rotate_around_long_axis(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    上腕骨長軸（z方向=[1,0,0]）まわりに回転させて
    回内外バリエーションを生成する。
    """
    if abs(angle_deg) < 0.5:
        return volume
    center  = np.array(volume.shape, dtype=float) / 2
    rot     = Rotation.from_euler("x", angle_deg, degrees=True)
    inv_rot = rot.as_matrix().T
    offset  = center - inv_rot @ center
    return affine_transform(
        volume, matrix=inv_rot, offset=offset,
        output_shape=volume.shape, order=1, mode="constant", cval=-1000.0,
    )


# ─── DRR 生成 ──────────────────────────────────────────────────────────────

def generate_drr(volume: np.ndarray, projection_axis: int, clahe: bool = True) -> np.ndarray:
    """
    簡易DRR（線減衰係数の積算）。
    projection_axis: 0=軸位, 1=AP（冠状）, 2=LAT（矢状）
    """
    mu  = np.clip((volume + 1000) / 1000, 0, None)
    drr = mu.sum(axis=projection_axis).astype(np.float32)
    mn, mx = drr.min(), drr.max()
    drr = ((drr - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
    if clahe:
        engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        drr = engine.apply(drr)
    return drr


# ─── ASCII プレビュー ─────────────────────────────────────────────────────────

def _axis_arrow(angle_deg: float, length: int = 5) -> str:
    """角度からASCII矢印を生成（0°=右, 90°=上, など）"""
    arrows = {
        (-22.5,  22.5): "→",
        ( 22.5,  67.5): "↗",
        ( 67.5, 112.5): "↑",
        (112.5, 157.5): "↖",
        (157.5, 180.0): "←",
        (-180., -157.5): "←",
        (-157.5,-112.5): "↙",
        (-112.5, -67.5): "↓",
        (-67.5,  -22.5): "↘",
    }
    for (lo, hi), arrow in arrows.items():
        if lo <= angle_deg < hi:
            return arrow
    return "•"


def ascii_preview(volume: np.ndarray,
                  humeral_axis: np.ndarray,
                  trans_axis: np.ndarray,
                  voxel_spacing: tuple,
                  hu_threshold: float,
                  width: int = 60,
                  height: int = 20) -> None:
    """
    --preview 時にターミナルでボリュームの断面と検出軸を ASCII で表示する。

    表示内容:
      [左] 側面投影 (ZY 平面): 上腕骨長軸の向き確認
      [右] 顆部断面 (YX 平面): 経顆軸の向き確認
    """
    mask = volume > hu_threshold
    nz, ny, nx = volume.shape
    dz, dy, dx = voxel_spacing

    # ─ 左パネル: ZY 側面投影（X方向に最大値投影） ─────────────────────────
    side = mask.any(axis=2).astype(float)     # (Z, Y)
    panel_w = width // 2 - 2
    panel_h = height

    # リサイズ（ブロック平均）
    def resample(img2d, out_h, out_w):
        h, w = img2d.shape
        row_idx = (np.arange(out_h) * h / out_h).astype(int)
        col_idx = (np.arange(out_w) * w / out_w).astype(int)
        return img2d[np.ix_(row_idx, col_idx)]

    side_small   = resample(side,  panel_h, panel_w)
    # 顆部断面: 骨面積が最大のスライス
    condyle_z    = int(mask.sum(axis=(1, 2)).argmax())
    front        = mask[condyle_z].astype(float)              # (Y, X)
    front_small  = resample(front, panel_h, panel_w)

    # 軸の方向を画面座標に変換（Z↓, Y→）
    #   side panel:  横=Y軸, 縦=Z軸
    #   front panel: 横=X軸, 縦=Y軸
    h_zyx = humeral_axis / (np.linalg.norm(humeral_axis) + 1e-9)
    t_zyx = trans_axis   / (np.linalg.norm(trans_axis)   + 1e-9)

    # 上腕骨長軸を ZY 平面に投影 → 角度
    import math
    h_angle = math.degrees(math.atan2(-h_zyx[0], h_zyx[1]))   # ZY面: 右=Y, 上=-Z
    t_angle = math.degrees(math.atan2(-t_zyx[1], t_zyx[2]))   # YX面: 右=X, 上=-Y

    def render_panel(grid, label, axis_angle, axis_name):
        lines = []
        lines.append(f"  {label}")
        lines.append("  +" + "-" * grid.shape[1] + "+")
        cy, cx = grid.shape[0] // 2, grid.shape[1] // 2
        for r in range(grid.shape[0]):
            row = ""
            for c in range(grid.shape[1]):
                if r == cy and c == cx:
                    row += _axis_arrow(axis_angle)
                elif grid[r, c] > 0:
                    row += "█"
                else:
                    row += "·"
            lines.append("  |" + row + "|")
        lines.append("  +" + "-" * grid.shape[1] + "+")
        lines.append(f"  {axis_name}: {_axis_arrow(axis_angle)}")
        return lines

    left_lines  = render_panel(side_small,  "側面 (ZY) — 上腕骨長軸", h_angle, "上腕骨長軸")
    right_lines = render_panel(front_small, f"顆部断面 z={condyle_z} (YX) — 経顆軸", t_angle, "経顆軸")

    max_rows = max(len(left_lines), len(right_lines))
    left_lines  += [""] * (max_rows - len(left_lines))
    right_lines += [""] * (max_rows - len(right_lines))

    col_w = panel_w + 6
    print()
    for l, r in zip(left_lines, right_lines):
        print(f"{l:<{col_w}}{r}")
    print()


# ─── メイン ──────────────────────────────────────────────────────────────────

def parse_coords(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split(",")])


def main():
    parser = argparse.ArgumentParser(description="CT肘ボリューム 自動向き補正・DRR生成")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--hu",        type=float, default=300.0,
                        help="骨HU閾値（デフォルト: 300）")
    parser.add_argument("--preview",   action="store_true",
                        help="向き確認のみ（保存なし）")
    parser.add_argument("--rotations", default="0",
                        help="回内外バリエーション角度（例: 0,30,60,-30,-60）")
    parser.add_argument("--size",      type=int, default=512)
    parser.add_argument("--epic_lat",  default=None,
                        help="外側上顆の座標（z,y,x） 例: 120,80,200")
    parser.add_argument("--epic_med",  default=None,
                        help="内側上顆の座標（z,y,x） 例: 120,80,310")
    args = parser.parse_args()

    rotation_angles = [float(a) for a in args.rotations.split(",")]
    epic_lat = parse_coords(args.epic_lat) if args.epic_lat else None
    epic_med = parse_coords(args.epic_med) if args.epic_med else None

    print("=== CT 解剖学的向き補正・DRR生成 ===")
    print(f"入力: {args.input}")
    print(f"骨閾値: HU > {args.hu}")
    print()

    # Step 1: 読み込み + スキャン方向補正
    print("[Step 1] DICOMシリーズ読み込み・スキャン方向補正...")
    volume, info = load_dicom_series(args.input)
    volume = correct_scan_direction(volume, info)
    voxel_spacing = (info["z_spacing"], info["pixel_spacing"][0], info["pixel_spacing"][1])
    print(f"  形状: {info['shape']}, ボクセル: {voxel_spacing[0]:.2f}×{voxel_spacing[1]:.3f}×{voxel_spacing[2]:.3f} mm")

    # Step 2: 上腕骨長軸検出
    print("\n[Step 2] 上腕骨長軸を PCA で検出...")
    humeral_axis, centroid = detect_humeral_axis(volume, args.hu, voxel_spacing)

    # Step 3: 経顆軸検出（AP/LAT基準面の定義）
    print("\n[Step 3] 経顆軸（AP/LAT基準）を検出...")
    trans_axis = detect_transepicondylar_axis(
        volume, humeral_axis, centroid, args.hu, voxel_spacing, epic_lat, epic_med
    )

    if args.preview:
        print("\n[Preview モード] 座標系定義のみ確認")
        build_anatomical_rotation(humeral_axis, trans_axis)
        ascii_preview(volume, humeral_axis, trans_axis, voxel_spacing, args.hu)
        print("要約:")
        print(f"  スキャン: {'アウトスキャン' if info['is_feet_first'] else 'インスキャン'}")
        print(f"  長軸: {humeral_axis.round(3)}")
        print(f"  経顆軸: {trans_axis.round(3)}")
        print("\n  → --epic_lat/--epic_med で経顆軸を手動指定するとAP/LATの精度が上がります")
        return

    # Step 4: 解剖学的座標系で回転補正
    print("\n[Step 4] 解剖学的座標系に整列...")
    rot_mat = build_anatomical_rotation(humeral_axis, trans_axis)
    volume_aligned = apply_rotation(volume, rot_mat, centroid)

    # Step 5: DRR生成
    print("\n[Step 5] DRR 生成・保存...")
    output_dir = args.output or args.input.rstrip("/") + "_drr"
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for rot_deg in rotation_angles:
        vol_rot = rotate_around_long_axis(volume_aligned, rot_deg)
        rot_tag = f"rot{rot_deg:+.0f}".replace("+", "p").replace("-", "m") if rot_deg != 0 else "rot0"

        drr_ap  = generate_drr(vol_rot, projection_axis=1)  # AP（経顆軸方向に投影）
        drr_lat = generate_drr(vol_rot, projection_axis=2)  # LAT（直交方向に投影）

        drr_ap  = cv2.resize(drr_ap,  (args.size, args.size), interpolation=cv2.INTER_AREA)
        drr_lat = cv2.resize(drr_lat, (args.size, args.size), interpolation=cv2.INTER_AREA)

        ap_path  = os.path.join(output_dir, f"drr_ap_{rot_tag}.png")
        lat_path = os.path.join(output_dir, f"drr_lat_{rot_tag}.png")
        cv2.imwrite(ap_path,  drr_ap)
        cv2.imwrite(lat_path, drr_lat)
        saved.extend([ap_path, lat_path])
        print(f"  {rot_tag}: AP → {os.path.basename(ap_path)},  LAT → {os.path.basename(lat_path)}")

    print(f"\n完了: {len(saved)} 枚保存 → {output_dir}")
    print()
    print("【経顆軸の精度が悪い場合】")
    print("  CTビューアで外側上顆・内側上顆のボクセル座標を確認して手動指定:")
    print(f"  python ct_reorient.py --input {args.input} \\")
    print(f"    --epic_lat <z,y,x> --epic_med <z,y,x>")


if __name__ == "__main__":
    main()
