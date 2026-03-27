"""
ElbowSynth — 肘CT から DRR（擬似X線）を自動生成する訓練データ作成スクリプト

OsteoVision の OsteoSynth/yolo_pose_factory.py を肘用に移植。

【目的】
  CTボリュームから様々な前腕回旋角度・肘屈曲角度で DRR を生成し、
  各画像に「理想ポジショニングからのズレ量」を自動でラベル付けする。
  LabelStudio による手動アノテーションは不要。

【CT向き自動検出】
  DICOM の ImageOrientationPatient タグから CT ボリュームの向きを検出し、
  必ず下記の「正準方向」に揃えてから処理する。
  6 方向（±PD / ±AP / ±ML）すべてに対応。

【正準方向（canonical orientation）】
  axis 0 = PD: 近位(proximal/肩側) → 遠位(distal/手側)  ← 腕の長軸
  axis 1 = AP: 前方(anterior) → 後方(posterior)
  axis 2 = ML: 内側(medial)   → 外側(lateral)

  AP  DRR: AP方向(axis1)に透視投影 → 画像 shape (PD, ML) rows=PD, cols=ML
  LAT DRR: ML方向(axis2)に透視投影 → 画像 shape (PD, AP) rows=PD, cols=AP
  SID: 1000mm（100cm）デフォルト。--sid オプションで変更可能

【最適ポジショニング基準（側面像）】
  - 肘屈曲: 90°
  - 内外側上顆: 完全に重なる（superimposed）
  - これからのズレ量を訓練ラベルとして使用

【使い方（複数シリーズ・ファントム対応）】
  cd /Users/kohei/develop/Dev/vision/ElbowVision
  source elbow-api/venv/bin/activate

  # 単一CTシリーズ
  python elbow-train/elbow_synth.py \
    --ct_dir  data/raw_dicom/ct/ \
    --out_dir data/yolo_dataset/ \
    --laterality L \
    --series_nums 12 \
    --base_flexions 90

  # 3ポジションCT（180°/135°/90°）を同一ディレクトリから一括処理
  python elbow-train/elbow_synth.py \
    --ct_dir  "/path/to/dicom_root" \
    --out_dir data/yolo_dataset/ \
    --laterality L \
    --series_nums 4,8,12 \
    --base_flexions 180,135,90 \
    --hu_min -200 --hu_max 1000 \
    --n_ap  120 --n_lat 360

【ファントムのウィンドウ推奨値】
  実骨:   --hu_min -400 --hu_max 1500（デフォルト）
  ファントム: --hu_min -200 --hu_max 1000

【出力】
  data/yolo_dataset/
  ├── images/train/   ← DRR画像 (PNG)
  ├── images/val/
  ├── labels/train/   ← YOLOキーポイントラベル (.txt)
  ├── labels/val/
  └── dataset_summary.csv  ← 各画像の回旋ズレ・屈曲角
"""

import argparse
import csv
import math
import os
import random

import cv2
import numpy as np
import pydicom
from scipy.ndimage import zoom, affine_transform

try:
    import yaml
except ImportError:
    yaml = None


# ─── CT ボリューム読み込み ──────────────────────────────────────────────────────

def load_ct_slices(dicom_dir: str, series_num: int = None):
    """
    DICOMフォルダ（サブディレクトリ含む）からスライス群を読み込み、Z軸順にソートして返す。

    series_num: SeriesNumber で絞り込む（複数シリーズ混在ディレクトリ用）
    """
    dcm_paths = []
    for root, _dirs, files in os.walk(dicom_dir):
        for f in sorted(files):
            if f.lower().endswith(('.dcm', '.dicom')):
                dcm_paths.append(os.path.join(root, f))

    if not dcm_paths:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # シリーズ番号でフィルタ
    if series_num is not None:
        filtered = []
        for p in dcm_paths:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if int(getattr(ds, 'SeriesNumber', -1)) == series_num:
                filtered.append(p)
        if not filtered:
            raise ValueError(f"SeriesNumber={series_num} が {dicom_dir} に見つかりません")
        dcm_paths = filtered
        print(f"  シリーズ {series_num}: {len(dcm_paths)} スライス")

    slices = [pydicom.dcmread(p) for p in dcm_paths]

    if hasattr(slices[0], 'ImagePositionPatient'):
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    elif hasattr(slices[0], 'InstanceNumber'):
        slices.sort(key=lambda s: int(s.InstanceNumber))

    return slices


def detect_laterality(slices, fallback: str = 'R') -> str:
    """
    DICOMタグから左右（'R' or 'L'）を検出する。

    参照タグ（優先順）:
      1. ImageLaterality (0020,0062) — シリーズ単位の左右
      2. Laterality       (0020,0060) — スタディ単位の左右
      3. fallback                     — タグなし時のデフォルト（'R'）

    左右の違いがMLフリップに影響する:
      右腕: X+(患者左) = lateral → ML flip if sign > 0
      左腕: X+(患者左) = medial  → ML flip if sign < 0
    """
    ref = slices[0]
    for tag in ('ImageLaterality', 'Laterality'):
        val = getattr(ref, tag, None)
        if val is not None:
            laterality = str(val).strip().upper()
            if laterality in ('R', 'L'):
                print(f"  左右検出 ({tag}): {laterality}腕")
                return laterality
    print(f"  左右タグなし → デフォルト: {fallback}腕 (--laterality で上書き可)")
    return fallback


# ─── CT 向き検出・正準化 ────────────────────────────────────────────────────────
#
# 患者座標系（DICOM LPS）:
#   X+ = Left（患者の左）
#   Y+ = Posterior（患者の後方）
#   Z+ = Superior（患者の頭側 = 腕の近位側）
#
# ボリューム軸とDICOMの対応:
#   axis 0 (depth)  = スライス積み上げ方向  ← cross(row, col) で決まる
#   axis 1 (height) = DICOM の col 方向
#   axis 2 (width)  = DICOM の row 方向
#
# patient_axis コード: 0=LR(X), 1=AP(Y), 2=SI(Z)

def parse_volume_to_patient_mapping(slices):
    """
    ImageOrientationPatient から各ボリューム軸の患者座標方向を返す。

    戻り値:
      mapping: [(patient_axis, sign), ...] length=3, index = volume axis 0/1/2
        patient_axis: 0=LR, 1=AP, 2=SI
        sign: +1 なら軸増加 = 患者座標+方向、-1 なら逆
    """
    ref = slices[0]
    iop = getattr(ref, 'ImageOrientationPatient', None)

    if iop is None:
        # タグなし → デフォルト: 軸アライン axial スキャンと仮定
        # axis0=SI(Z), axis1=AP(Y), axis2=LR(X)
        return [(2, 1), (1, 1), (0, 1)]

    row_cos = np.array([float(v) for v in iop[:3]])   # axis2 (width)  方向
    col_cos = np.array([float(v) for v in iop[3:]])   # axis1 (height) 方向
    slc_cos = np.cross(row_cos, col_cos)               # axis0 (depth)  方向

    mapping = []
    for cosines in [slc_cos, col_cos, row_cos]:   # axis 0, 1, 2 の順
        dom = int(np.argmax(np.abs(cosines)))
        sign = 1 if cosines[dom] > 0 else -1
        mapping.append((dom, sign))

    return mapping


def reorient_volume_canonical(volume: np.ndarray, mapping: list, laterality: str = 'R'):
    """
    ボリュームを正準方向（PD, AP, ML）に揃える。

      target axis 0 = SI/PD  (patient_axis=2, flip if sign>0 so proximal→low_index)
      target axis 1 = AP     (patient_axis=1, flip if sign<0 so anterior→low_index)
      target axis 2 = LR/ML  (patient_axis=0, flip if sign<0 so medial→low_index)

    戻り値:
      canonical: 正準方向のボリューム (float32)
      transpose_order: [old_ax_for_new0, old_ax_for_new1, old_ax_for_new2]
      flip_map: [bool, bool, bool] 各新軸で反転したか
    """
    # 目標患者軸（正準 axis 0/1/2 にそれぞれ何の患者軸を入れるか）
    target_patient_axes = [2, 1, 0]   # SI, AP, LR
    # 反転ルール: 正準方向の「増加方向」が患者座標の何に相当するか
    # SI(Z): Z+はsuperior/proximal。正準axis0はproximal→distal。
    #        sign>0 = Z+が低indexに来る = proximal側が低index → flip不要
    #        sign<0 = Z-（inferior）が低index = distal側が低index → flip必要? 違う。
    #        sign>0 = 低indexが proximal = 正準と同じ → flip不要... いや:
    #        正準axis0は proximal(index 0) → distal(index max)
    #        Z+はsuperior/proximalなので sign>0 = index増加 = proximal→ ... = Z+方向 = proximal増加?
    #        → index 0 が proximal = 正準通り → NO flip
    #        sign<0 = index増加 = Z- = distal方向 = index 0 が distal → flip必要
    #   正しい: flip when sign < 0? 違う、コメント(行131)が flip if sign>0 と言っている...
    #   DICOM Z+はsuperior(proximal)。sign>0 = 低index側がsuperior/proximal = 正準通り(proximal at low)
    #   → sign>0 ならflip不要、sign<0 ならflip必要。しかし元コードは sign>0でflip。
    #   ここは元コードが正しい: CT scanの積み上げ順がsuperior→inferiorなら最初のスライスが近位側。
    #   この場合Z+（近位）がaxis0の低indexに来る(sign>0)→正準 proximal→distal と逆→flip必要。
    # AP(Y): Y+はposterior。正準axis1はanterior→posterior。
    #        sign>0 = index増加 = Y+ = posterior方向 = anterior→posterior = 正準通り → NO flip
    #        sign<0 = flip必要
    # ML(X): X+はLeft。腕の向きにより Left=medial or lateral。
    #        正準axis2はmedial→lateral。
    #        右腕: X+(Left) = lateral → index増加=lateral → flip必要 (sign>0)
    #        左腕: X+(Left) = medial  → index増加=medial  → flip必要 (sign<0)
    ml_flip = (lambda sign: sign > 0) if laterality == 'R' else (lambda sign: sign < 0)
    flip_rules = {2: lambda sign: sign > 0,   # SI: low indexが近位側(Z+/proximal) → flip
                  1: lambda sign: sign < 0,   # AP: anterior→posterior (Y+) → flip if sign<0
                  0: ml_flip}                 # ML: 左右腕で反転ルールが異なる

    transpose_order = []
    flip_map = []
    used = set()

    for target_pa in target_patient_axes:
        found = False
        for old_ax, (pa, sign) in enumerate(mapping):
            if pa == target_pa and old_ax not in used:
                transpose_order.append(old_ax)
                flip_map.append(flip_rules[target_pa](sign))
                used.add(old_ax)
                found = True
                break
        if not found:
            raise ValueError(
                f"CT向き検出失敗: patient_axis={target_pa} に対応するボリューム軸が見つかりません。"
                f" mapping={mapping}"
            )

    canonical = np.transpose(volume, transpose_order)
    for new_ax, do_flip in enumerate(flip_map):
        if do_flip:
            canonical = np.flip(canonical, axis=new_ax).copy()

    return canonical.astype(np.float32), transpose_order, flip_map


def transform_landmarks_canonical(landmarks_norm: dict,
                                   transpose_order: list,
                                   flip_map: list) -> dict:
    """
    ボリューム再配向後にランドマーク正規化座標を更新する。

    landmarks_norm: {name: (n_axis0, n_axis1, n_axis2)} 元ボリュームの正規化座標
    """
    updated = {}
    for name, coords in landmarks_norm.items():
        # 転置: 新 axis i の座標 = 元の axis transpose_order[i] の座標
        new_coords = [coords[transpose_order[i]] for i in range(3)]
        # 反転
        for i, do_flip in enumerate(flip_map):
            if do_flip:
                new_coords[i] = 1.0 - new_coords[i]
        updated[name] = tuple(new_coords)
    return updated


def load_ct_volume(dicom_dir: str, target_size: int = 128, laterality: str = None,
                   series_num: int = None, hu_min: float = -400, hu_max: float = 1500):
    """
    DICOM フォルダから CT ボリュームを読み込み、正準方向に揃えて返す。

    laterality : 'R'（右腕）/ 'L'（左腕）/ None（DICOMタグから自動検出）
    series_num : SeriesNumber で絞り込む（省略時は全スライスを使用）
    hu_min/max : HU ウィンドウ（ファントム推奨: -200〜1000、実骨: -400〜1500）

    戻り値:
      volume       : (PD, AP, ML) float32 配列（0〜1正規化）
      voxel_spacing: (pd_mm, ap_mm, ml_mm) 各方向の実寸法（概算）
      laterality   : 'R' or 'L'
      voxel_mm_eff : 実効ボクセルサイズ（mm/voxel）
    """
    slices = load_ct_slices(dicom_dir, series_num=series_num)
    if laterality is None:
        laterality = detect_laterality(slices)
    else:
        print(f"  左右（指定値）: {laterality}腕")
    mapping = parse_volume_to_patient_mapping(slices)

    ref = slices[0]
    ps = getattr(ref, 'PixelSpacing', [1.0, 1.0])
    y_mm, x_mm = float(ps[0]), float(ps[1])

    if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
        z_mm = abs(
            float(slices[1].ImagePositionPatient[2]) -
            float(slices[0].ImagePositionPatient[2])
        )
    else:
        z_mm = float(getattr(ref, 'SliceThickness', 1.0))

    # スタック → HU変換（RescaleSlope / RescaleIntercept を適用）
    def to_hu(s):
        arr = s.pixel_array.astype(np.float32)
        slope     = float(getattr(s, 'RescaleSlope',     1.0))
        intercept = float(getattr(s, 'RescaleIntercept', 0.0))
        return arr * slope + intercept

    stack = np.stack([to_hu(s) for s in slices])
    nz_orig, ny_orig, nx_orig = stack.shape

    # 等方ボクセルにリサンプリング
    scale = (z_mm / x_mm, 1.0, 1.0)
    volume = zoom(stack, scale, order=1)

    # target_size にリサイズ
    sf = target_size / max(volume.shape)
    volume = zoom(volume, sf, order=1)

    # 実効ボクセルサイズ（mm/voxel）: 物理最大寸法 / target_size
    phys_max_mm = max(nz_orig * z_mm, ny_orig * y_mm, nx_orig * x_mm)
    voxel_mm_eff = phys_max_mm / target_size

    # HU ウィンドウ → 0〜1 正規化
    print(f"  HU window: {hu_min} 〜 {hu_max}")
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # 正準方向に揃える（6方向 × 左右腕 対応）
    canonical, t_order, f_map = reorient_volume_canonical(volume, mapping, laterality)

    print(f"  CT orientation: {['LR','AP','SI'][mapping[0][0]]}(axis0) "
          f"{['LR','AP','SI'][mapping[1][0]]}(axis1) "
          f"{['LR','AP','SI'][mapping[2][0]]}(axis2)")
    print(f"  Canonical shape (PD,AP,ML): {canonical.shape}")
    print(f"  Transpose order: {t_order}, Flips: {f_map}")

    return canonical, (z_mm, y_mm, x_mm), laterality, voxel_mm_eff


# ─── キーポイント定義（正準方向 PD/AP/ML 座標） ────────────────────────────────
#
# 正規化座標 (n_PD, n_AP, n_ML):
#   n_PD: 0=肩側(近位) / 1=手側(遠位)
#   n_AP: 0=前方       / 1=後方
#   n_ML: 0=内側       / 1=外側
#
# ← CT 撮影後、ITK-SNAP 等で実測してここを更新する

DEFAULT_LANDMARKS_NORMALIZED = {
    "humerus_shaft":      (0.05, 0.50, 0.49),   # 上腕骨幹部（近位）
    "lateral_epicondyle": (0.30, 0.50, 0.80),   # 外側上顆（ML方向 外側寄り）
    "medial_epicondyle":  (0.30, 0.50, 0.19),   # 内側上顆（ML方向 内側寄り）
    "forearm_shaft":      (0.55, 0.50, 0.49),   # 前腕骨幹部（遠位）
    "radial_head":        (0.38, 0.43, 0.70),   # 橈骨頭（外側前方）
    "olecranon":          (0.30, 0.63, 0.49),   # 肘頭（後方）
    "joint_center":       (0.30, 0.50, 0.49),   # 関節中心（回転軸）
}


# ─── 回転行列 ──────────────────────────────────────────────────────────────────
#
# 正準ボリューム軸: [PD=axis0, AP=axis1, ML=axis2]
#
# rotation_matrix_x: axis0(PD)を固定し axis1/axis2(AP/ML)が回転
#   → 前腕の長軸まわりの回旋（pronation/supination）に使用
#
# rotation_matrix_z: axis2(ML)を固定し axis0/axis1(PD/AP)が回転
#   → 肘の屈曲・伸展（flexion/extension）に使用
#
# rotation_matrix_y: axis1(AP)を固定し axis0/axis2が回転
#   → 内反・外反（varus/valgus）調整用（拡張用途）

def rotation_matrix_x(deg: float) -> np.ndarray:
    """axis0(PD)まわりの回転 → 前腕回旋（pronation/supination）"""
    r = math.radians(deg)
    return np.array([
        [1, 0,            0           ],
        [0, math.cos(r), -math.sin(r) ],
        [0, math.sin(r),  math.cos(r) ],
    ])


def rotation_matrix_y(deg: float) -> np.ndarray:
    """axis1(AP)まわりの回転 → 内反・外反（参考用）"""
    r = math.radians(deg)
    return np.array([
        [ math.cos(r), 0, math.sin(r)],
        [ 0,           1, 0          ],
        [-math.sin(r), 0, math.cos(r)],
    ])


def rotation_matrix_z(deg: float) -> np.ndarray:
    """axis2(ML)まわりの回転 → 肘屈曲・伸展（flexion/extension）"""
    r = math.radians(deg)
    return np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r),  math.cos(r), 0],
        [0,            0,           1],
    ])


# ─── DRR 生成（透視投影コーンビーム + CLAHE） ─────────────────────────────────
#
# 正準ボリューム volume[PD, AP, ML] から:
#   AP  DRR: AP方向(axis1)に透視投影 → 画像 shape (PD, ML) rows=近位→遠位, cols=内側→外側
#   LAT DRR: ML方向(axis2)に透視投影 → 画像 shape (PD, AP) rows=近位→遠位, cols=前→後
#
# 透視投影ジオメトリ（AP像の例）:
#   ソース (point source) が AP = -D_s の位置（ボリューム前端より手前）
#   検出器が AP = D（ボリューム後端）
#   SID = D_s + D [voxels]
#   各検出器ピクセル (h_det, w_det) に対して、
#   ソースからの逆レイで volume 座標を求めて積分する。
#
#   逆マッピング (深度 d での volume 座標):
#     h_vol = H/2 + (h_det - H/2) * (D_s + d) / SID_vox
#     w_vol = W/2 + (w_det - W/2) * (D_s + d) / SID_vox

def generate_drr(volume: np.ndarray, axis: str = "AP",
                  sid_mm: float = 1000.0, voxel_mm: float = 1.0) -> np.ndarray:
    """
    正準方向ボリュームから透視投影（コーンビーム）DRRを生成する。

    【Beer-Lambert指数減衰モデル】
      実際のX線はI = I_0 * exp(-∫μ(x)dx) で減衰する。
      CTの正規化値（0〜1）をμ（線減弱係数）として扱い、
      指数減衰で透過強度を計算 → 実X線に近い濃淡を再現。

      骨（高μ）: 白く映る（X線が透過しにくい）
      軟部組織（低μ）: 暗く映る
      空気（μ≈0）: 黒

    axis:
      "AP"  — 前後方向投影（正面像）: AP方向(axis1)に透視投影
      "LAT" — 内外側方向投影（側面像）: ML方向(axis2)に透視投影

    sid_mm  : X線管焦点〜検出器間距離（mm）。実撮影条件に合わせる（デフォルト 1000mm）
    voxel_mm: 正準ボリュームの実効ボクセルサイズ（mm/voxel）

    戻り値: (H, W) uint8 画像
    """
    from scipy.ndimage import map_coordinates as _mc

    NP, NA, NM = volume.shape

    if axis == "AP":
        H, W, D = NP, NM, NA
    else:
        H, W, D = NP, NA, NM

    SID_vox = sid_mm / voxel_mm
    D_s = max(SID_vox - D, 1.0)

    # 検出器グリッド (H, W)
    hh, ww = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32), indexing='ij')

    d_vals = np.arange(D, dtype=np.float32)
    inv_m  = (D_s + d_vals) / SID_vox

    inv_m3 = inv_m[None, None, :]
    h_vol  = H / 2.0 + (hh[:, :, None] - H / 2.0) * inv_m3
    w_vol  = W / 2.0 + (ww[:, :, None] - W / 2.0) * inv_m3
    d_vol  = np.tile(d_vals[None, None, :], (H, W, 1))

    if axis == "AP":
        coords = np.stack([h_vol.ravel(), d_vol.ravel(), w_vol.ravel()], axis=0)
    else:
        coords = np.stack([h_vol.ravel(), w_vol.ravel(), d_vol.ravel()], axis=0)

    samples = _mc(volume, coords, order=1, mode='constant', cval=0.0)
    mu_volume = samples.reshape(H, W, D)

    # ── Beer-Lambert指数減衰 ──
    # μスケーリング: CT正規化値(0〜1)を実効的な線減弱係数にスケール
    # 線積分値 = sum(μ) * voxel_mm で、投影深度D個のボクセルを積分する。
    # 骨(μ≈0.6) × D(≈90) = 54 だと exp(-54)≈0 で真っ白になるため、
    # voxel_mm ではなく固定の正規化係数を使い、
    # 線積分値が bone で 2〜4 程度になるよう調整する。
    line_integral = mu_volume.sum(axis=2)
    # 正規化: 最大線積分が target_max_attenuation になるようスケール
    li_max = line_integral.max() + 1e-8
    target_max_attenuation = 2.5  # 骨で exp(-2.5)≈0.082、空気で exp(0)=1
    line_integral = line_integral * (target_max_attenuation / li_max)
    # I = I_0 * exp(-∫μ dx) → 透過強度
    transmission = np.exp(-line_integral)

    # ── 散乱線シミュレーション（簡易版） ──
    scatter_fraction = 0.15  # 散乱線割合（15%: 実X線に近い値）
    intensity = (1.0 - scatter_fraction) * transmission + scatter_fraction

    # ── ヒール効果（X線管からの距離による強度低下の簡易近似） ──
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32), indexing='ij')
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    max_dist_sq = cy ** 2 + cx ** 2 + 1e-8
    heel_effect = 1.0 - 0.05 * (dist_sq / max_dist_sq)
    intensity = intensity * heel_effect

    # ── X線画像変換（透過→表示） ──
    # DR(デジタルX線)は骨が白い（反転表示）
    img = 1.0 - intensity  # 反転: 骨=白, 空気=黒

    # ── 背景マスク（空気部分を完全に黒くする） ──
    # 線積分がほぼ0の領域（空気のみ通過）は黒にする
    air_threshold = 0.02  # 反転後にこれ以下は背景
    bg_mask = img < air_threshold

    # ── ウィンドウ処理（骨のダイナミックレンジに最適化） ──
    # 背景以外のピクセルで統計を取る
    fg_pixels = img[~bg_mask]
    if len(fg_pixels) > 0:
        p_low = np.percentile(fg_pixels, 2)
        p_high = np.percentile(fg_pixels, 99.5)
    else:
        p_low, p_high = 0.0, 1.0
    img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
    img[bg_mask] = 0.0  # 背景は完全黒

    # ── ガンマ補正（レントゲンフィルムの特性曲線を模倣） ──
    gamma = 1.0  # リニア（実X線に近い）
    img = np.power(img + 1e-8, gamma)
    img[bg_mask] = 0.0

    # ── 軽いガウシアンブラー（X線の焦点ぼけを再現） ──
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0.5)

    # CLAHE（局所的なコントラスト強調、控えめに）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img_u8)


# ─── ボリューム回転 + キーポイント変換 ────────────────────────────────────────

def rotate_volume_and_landmarks(
    volume: np.ndarray,
    landmarks_norm: dict,
    forearm_rotation_deg: float,
    flexion_deg: float,
    base_flexion: float = 90.0,
    valgus_deg: float = 0.0,
) -> tuple:
    """
    正準方向ボリュームに対して、前腕回旋と肘屈曲を適用する。

    【解剖学的分離回転】
      肘の屈曲/伸展は関節中心（joint_center）を境に:
        - 上腕骨（近位側）: 固定
        - 前腕骨（遠位側）: joint_center を中心に回転
      これにより実際の肘関節運動に近いDRRが生成される。

      前腕回旋（pronation/supination）も同様に前腕部分のみに適用。

    forearm_rotation_deg:
      理想位からのズレ量（°）。0=理想、+方向=回外、-方向=回内
      前腕長軸(PD=axis0)まわりの回転 → rotation_matrix_x を使用

    flexion_deg:
      目標肘屈曲角（°）。90=LAT理想位、180=AP伸展位。

    base_flexion:
      CT撮影時の実際の肘屈曲角（°）。
      回転量 = flexion_deg - base_flexion として計算。
      デフォルト 90.0（従来互換）。

    戻り値:
      rotated_volume   : 回転後のボリューム (float32)
      rotated_landmarks: 回転後のキーポイント座標（正規化）
    """
    pd, ap, ml = volume.shape

    # ── valgus（内反・外反）: ポジショニング誤差のため全体ボリュームに適用 ──
    if abs(valgus_deg) > 1e-6:
        R_val = rotation_matrix_y(valgus_deg)
        R_val_inv = R_val.T
        vol_center = np.array([pd / 2.0, ap / 2.0, ml / 2.0])
        offset_val = vol_center - R_val_inv @ vol_center
        volume = affine_transform(
            volume, R_val_inv, offset=offset_val, order=1, mode='constant', cval=0.0
        )
        # ランドマークも同じ回転を適用
        rotated_lm_val = {}
        for name, (nPD, nAP, nML) in landmarks_norm.items():
            p = np.array([nPD * pd, nAP * ap, nML * ml]) - vol_center
            p_rot = R_val @ p + vol_center
            rotated_lm_val[name] = (p_rot[0] / pd, p_rot[1] / ap, p_rot[2] / ml)
        landmarks_norm = rotated_lm_val

    # 回転軸: joint_center（解剖学的関節中心）を優先
    if "joint_center" in landmarks_norm:
        jc = landmarks_norm["joint_center"]
        center = np.array([jc[0] * pd, jc[1] * ap, jc[2] * ml])
    else:
        center = np.array([pd / 2, ap / 2, ml / 2])

    # 前腕長軸(PD=axis0)まわりの前腕回旋
    R_rot  = rotation_matrix_x(forearm_rotation_deg)
    # ML軸(axis2)まわりの肘屈曲（CT実ポジションからの差分で回転）
    R_flex = rotation_matrix_z(flexion_deg - base_flexion)
    R = R_flex @ R_rot
    R_inv = R.T

    # ── 前腕分離回転: joint_center より遠位（PD > joint_center）だけを回転 ──
    # 上腕側はそのまま保持し、前腕側だけ関節中心を軸に回転させる
    joint_pd_vox = int(center[0])  # 関節中心のPDインデックス

    # 遷移帯: 関節中心付近で急にマスクが切れるとアーティファクトが出るため、
    # ±blend_half ボクセルの範囲でスムーズにブレンドする
    blend_half = max(2, int(pd * 0.03))  # ボリュームの3%幅

    # 出力ボリューム: 上腕側は元のまま
    rotated = volume.copy()

    # 前腕部分だけ回転したボリュームを生成
    offset_forearm = center - R_inv @ center
    forearm_rotated = affine_transform(
        volume, R_inv, offset=offset_forearm, order=1, mode='constant', cval=0.0
    )

    # ブレンドマスク: 0=上腕(元のまま), 1=前腕(回転後)
    blend_mask = np.zeros(pd, dtype=np.float32)
    for i in range(pd):
        if i >= joint_pd_vox + blend_half:
            blend_mask[i] = 1.0
        elif i > joint_pd_vox - blend_half:
            blend_mask[i] = (i - (joint_pd_vox - blend_half)) / (2 * blend_half)

    # 3Dブレンドマスクに拡張して適用
    blend_3d = blend_mask[:, None, None]  # (PD, 1, 1) → ブロードキャスト
    rotated = (1.0 - blend_3d) * volume + blend_3d * forearm_rotated

    # キーポイント座標変換
    # 上腕側ランドマーク（joint_center以上）は固定、前腕側は回転
    humerus_landmarks = {"humerus_shaft"}
    rotated_lm = {}
    for name, (nPD, nAP, nML) in landmarks_norm.items():
        if name in humerus_landmarks:
            # 上腕側: 固定
            rotated_lm[name] = (nPD, nAP, nML)
        else:
            # 前腕側 + 関節中心付近: 関節中心を軸に回転
            p = np.array([nPD * pd, nAP * ap, nML * ml]) - center
            p_rot = R @ p + center
            rotated_lm[name] = (
                p_rot[0] / pd,
                p_rot[1] / ap,
                p_rot[2] / ml,
            )

    return rotated.astype(np.float32), rotated_lm


# ─── YOLO ラベル生成 ────────────────────────────────────────────────────────────
#
# 正準座標 (n_PD, n_AP, n_ML) から透視投影 2D 座標を算出:
#
#   AP  DRR: ソースが AP 軸前方。投影方向の深度 = n_AP * NA
#     px = 0.5 + (n_ML - 0.5) * SID_vox / (D_s + n_AP * NA)
#     py = 0.5 + (n_PD - 0.5) * SID_vox / (D_s + n_AP * NA)
#
#   LAT DRR: ソースが ML 軸前方。投影方向の深度 = n_ML * NM
#     px = 0.5 + (n_AP - 0.5) * SID_vox / (D_s + n_ML * NM)
#     py = 0.5 + (n_PD - 0.5) * SID_vox / (D_s + n_ML * NM)


def _project_kp_perspective(n_PD: float, n_depth: float, n_lateral: float,
                              vol_depth: int, D_s: float, SID_vox: float):
    """
    1 キーポイントを透視投影して正規化 2D 座標 (px, py) を返す。

    n_depth  : 投影軸方向の正規化座標 (AP像: n_AP, LAT像: n_ML)
    n_lateral: 横軸方向の正規化座標   (AP像: n_ML, LAT像: n_AP)
    """
    mag = SID_vox / max(D_s + n_depth * vol_depth, 1e-6)
    px = max(0.0, min(1.0, 0.5 + (n_lateral - 0.5) * mag))
    py = max(0.0, min(1.0, 0.5 + (n_PD     - 0.5) * mag))
    return px, py


def make_yolo_label(landmarks_norm: dict, axis: str, img_h: int, img_w: int,
                    vol_shape: tuple = (128, 128, 128),
                    sid_mm: float = 1000.0, voxel_mm: float = 1.0) -> str:
    """
    YOLO Pose フォーマットのラベル文字列を生成する（透視投影対応）。

    フォーマット:
      class cx cy w h  kp0x kp0y kp0v  kp1x kp1y kp1v  ...
    """
    kp_order = [
        "humerus_shaft",      # 0: 上腕骨幹部
        "lateral_epicondyle", # 1: 外側上顆
        "medial_epicondyle",  # 2: 内側上顆
        "forearm_shaft",      # 3: 前腕骨幹部
        "radial_head",        # 4: 橈骨頭（前腕回旋の直接指標）
        "olecranon",          # 5: 肘頭（LAT屈曲角・AP/LAT判定に必要）
    ]

    NP, NA, NM = vol_shape
    SID_vox = sid_mm / voxel_mm

    kp_2d = []
    for name in kp_order:
        n_PD, n_AP, n_ML = landmarks_norm[name]
        if axis == "AP":
            D_s = max(SID_vox - NA, 1.0)
            px, py = _project_kp_perspective(n_PD, n_AP, n_ML, NA, D_s, SID_vox)
        else:
            D_s = max(SID_vox - NM, 1.0)
            px, py = _project_kp_perspective(n_PD, n_ML, n_AP, NM, D_s, SID_vox)
        kp_2d.append((px, py))

    # バウンディングボックス
    xs = [p[0] for p in kp_2d]
    ys = [p[1] for p in kp_2d]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    bw = max(0.05, min(1.0, (max(xs) - min(xs)) + 0.15))
    bh = max(0.05, min(1.0, (max(ys) - min(ys)) + 0.15))
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))

    # visibility:
    #   AP像:
    #     olecranon は後方に隠れる（前後方向投影なので見えにくい）→ occluded (1)
    #   LAT像:
    #     medial_epicondyle は外側上顆に重なる → occluded (1)
    #     radial_head は橈骨頭が尺骨と重なる場合あり → occluded (1)
    kp_visibility = []
    for name in kp_order:
        if axis == "AP" and name == "olecranon":
            kp_visibility.append(1)   # AP像: 肘頭は後方に隠れる
        elif axis == "LAT" and name in ("medial_epicondyle", "radial_head"):
            kp_visibility.append(1)   # LAT像: 内側上顆・橈骨頭は重なる
        else:
            kp_visibility.append(2)   # visible

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for (px, py), vis in zip(kp_2d, kp_visibility):
        parts.append(f"{px:.6f} {py:.6f} {vis}")

    return " ".join(parts)


# ─── ランドマーク自動検出 ──────────────────────────────────────────────────────

def auto_detect_landmarks(volume: np.ndarray, bone_threshold: float = None,
                          laterality: str = 'R') -> dict:
    """
    正準方向CT ボリューム (PD, AP, ML) からランドマーク座標を自動検出する。

    正規化済みボリューム（0〜1）を前提とし、高輝度領域を骨として扱う。
    ファントム素材の場合も最高密度領域を骨と見なすため概ね機能する。

    アルゴリズム:
      1. Otsu法で骨閾値を自動決定（bone_threshold=None の場合）
      2. PD方向の各スライスでML幅を計算 → 中央60%で最大 = 上顆レベル
      3. 上顆レベルでML最大/最小位置 → 外側/内側上顆
      4. 上顆レベルから近位15% / 遠位20%の骨重心 → 上腕骨/前腕骨幹部
         （PD固定値ではなく上顆位置からの相対位置）

    戻り値: DEFAULT_LANDMARKS_NORMALIZED と同じ形式の dict
    """
    pd_size, ap_size, ml_size = volume.shape

    # Otsu法で骨閾値を自動決定（ファントム・実骨の素材差に対応）
    if bone_threshold is None:
        flat = volume.flatten()
        # Otsu: ヒストグラムで二値化閾値を自動計算
        hist, bin_edges = np.histogram(flat, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()
        w0, w1, mu0, mu1 = 0.0, 1.0, 0.0, float((hist * bin_centers).sum() / total)
        best_var, best_thresh = 0.0, 0.5
        for i in range(len(hist)):
            w0 += hist[i] / total
            w1 = 1.0 - w0
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (mu0 * (w0 * total - hist[i]) + bin_centers[i] * hist[i]) / (w0 * total) if w0 * total > 0 else 0
            mu1 = (float((hist[i+1:] * bin_centers[i+1:]).sum()) / (w1 * total)) if w1 * total > 0 else 0
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > best_var:
                best_var, best_thresh = var, bin_centers[i]
        bone_threshold = best_thresh
        print(f"  骨閾値 (Otsu自動): {bone_threshold:.3f}")
    else:
        print(f"  骨閾値 (指定値): {bone_threshold:.3f}")

    bone_mask = volume > bone_threshold

    # ── 上顆レベル検出: ML幅が最大のPDスライス ────────────────────────────────
    ml_widths = np.zeros(pd_size)
    for pd_i in range(pd_size):
        ml_proj = bone_mask[pd_i].any(axis=0)   # (AP, ML) → (ML,)
        ml_idx  = np.where(ml_proj)[0]
        if len(ml_idx) >= 2:
            ml_widths[pd_i] = ml_idx.max() - ml_idx.min()

    # 近位・遠位端の外れ値を除くため中央60%に絞る
    pd_s = int(pd_size * 0.30)
    pd_e = int(pd_size * 0.70)
    joint_pd_idx  = pd_s + int(ml_widths[pd_s:pd_e].argmax())
    joint_pd_norm = joint_pd_idx / pd_size

    # ── 外側/内側上顆 ─────────────────────────────────────────────────────────
    joint_slice = bone_mask[joint_pd_idx]        # (AP, ML)
    ml_proj     = joint_slice.any(axis=0)        # (ML,)
    ap_proj     = joint_slice.any(axis=1)        # (AP,)
    ml_idx      = np.where(ml_proj)[0]
    ap_idx      = np.where(ap_proj)[0]

    if len(ml_idx) >= 2:
        # 正準axis2: medial(0) → lateral(1)
        # 右腕: reorient後 ML+ = lateral → ml_idx.max() が外側上顆
        # 左腕: reorient後 ML+ = lateral（reorient_volume_canonicalで統一済み）
        # → 左右腕ともに ml_idx.max() = lateral, ml_idx.min() = medial で一致
        lat_ml = float(ml_idx.max()) / ml_size   # 外側上顆（ML方向 最大 = lateral）
        med_ml = float(ml_idx.min()) / ml_size   # 内側上顆（ML方向 最小 = medial）
    else:
        print("  ⚠ 上顆検出失敗: デフォルト値を使用")
        lat_ml, med_ml = 0.65, 0.35

    epic_ap = float(ap_idx.mean()) / ap_size if len(ap_idx) > 0 else 0.50

    # ── 上腕骨幹部・前腕骨幹部: 上顆レベルからの相対位置で決定 ─────────────
    # PD固定値（0.20, 0.75）ではなくファントムサイズに依存しない相対指定
    def shaft_centroid(pd_norm: float):
        sl = bone_mask[int(pd_size * pd_norm)]   # (AP, ML)
        yx = np.where(sl)
        if len(yx[0]) > 0:
            return float(yx[0].mean()) / ap_size, float(yx[1].mean()) / ml_size
        return 0.50, 0.50

    # 上腕骨幹部: 上顆レベルより近位25%（ただし最低5%の余裕を確保）
    hum_pd_norm  = max(0.05, joint_pd_norm - 0.25)
    # 前腕骨幹部: 上顆レベルより遠位25%（ただし最大95%まで）
    fore_pd_norm = min(0.95, joint_pd_norm + 0.25)

    hum_ap,  hum_ml  = shaft_centroid(hum_pd_norm)
    fore_ap, fore_ml = shaft_centroid(fore_pd_norm)

    # ── 橈骨頭（radial_head）: 上顆レベルより少し遠位、外側前方 ──────────────
    # 外顆レベルから遠位5〜15%のスライスで、ML外側 × AP前半の骨重心
    rh_pd_norm = min(0.95, joint_pd_norm + 0.08)
    rh_sl = bone_mask[int(pd_size * rh_pd_norm)]   # (AP, ML)
    # 外側半（lat_ml以上）かつ前半（AP < 0.5）に絞る
    rh_mask = rh_sl.copy()
    rh_mask[:, :int(ml_size * lat_ml * 0.8)] = False   # 内側を除外
    rh_mask[int(ap_size * 0.55):, :] = False            # 後方を除外
    rh_yx = np.where(rh_mask)
    if len(rh_yx[0]) > 0:
        rh_ap = float(rh_yx[0].mean()) / ap_size
        rh_ml = float(rh_yx[1].mean()) / ml_size
    else:
        rh_ap, rh_ml = 0.40, lat_ml   # フォールバック: 外顆と同じML位置・前方AP

    # ── 肘頭（olecranon）: 上顆レベル付近、後方 ──────────────────────────────
    # 上顆レベルのスライスで AP後半（60〜100%）の骨重心
    ol_pd_norm = joint_pd_norm
    ol_sl = bone_mask[int(pd_size * ol_pd_norm)]   # (AP, ML)
    ol_mask = ol_sl.copy()
    ol_mask[:int(ap_size * 0.55), :] = False   # 前方を除外
    ol_yx = np.where(ol_mask)
    if len(ol_yx[0]) > 0:
        ol_ap = float(ol_yx[0].mean()) / ap_size
        ol_ml = float(ol_yx[1].mean()) / ml_size
    else:
        ol_ap, ol_ml = 0.80, (lat_ml + med_ml) / 2   # フォールバック: 後方・中央

    landmarks = {
        "humerus_shaft":      (hum_pd_norm,   hum_ap,  hum_ml),
        "lateral_epicondyle": (joint_pd_norm, epic_ap, lat_ml),
        "medial_epicondyle":  (joint_pd_norm, epic_ap, med_ml),
        "forearm_shaft":      (fore_pd_norm,  fore_ap, fore_ml),
        "radial_head":        (rh_pd_norm,    rh_ap,   rh_ml),
        "olecranon":          (ol_pd_norm,    ol_ap,   ol_ml),
        "joint_center":       (joint_pd_norm, epic_ap, (lat_ml + med_ml) / 2),
    }

    print(f"  自動検出結果 ({laterality}腕):")
    print(f"    上顆レベル    PD={joint_pd_norm:.2f}  外側上顆 ML={lat_ml:.2f}  内側上顆 ML={med_ml:.2f}")
    print(f"    上腕骨幹部   PD={hum_pd_norm:.2f}  AP={hum_ap:.2f}  ML={hum_ml:.2f}")
    print(f"    前腕骨幹部   PD={fore_pd_norm:.2f}  AP={fore_ap:.2f}  ML={fore_ml:.2f}")
    print(f"    橈骨頭       PD={rh_pd_norm:.2f}  AP={rh_ap:.2f}  ML={rh_ml:.2f}")
    print(f"    肘頭         PD={ol_pd_norm:.2f}  AP={ol_ap:.2f}  ML={ol_ml:.2f}")

    return landmarks


# ─── メイン生成ループ ──────────────────────────────────────────────────────────

def generate_dataset(args):
    laterality = getattr(args, 'laterality', None)
    sid_mm     = getattr(args, 'sid',        1000.0)
    hu_min     = getattr(args, 'hu_min',     -400.0)
    hu_max     = getattr(args, 'hu_max',     1500.0)

    # 複数シリーズのパース
    series_nums_raw   = getattr(args, 'series_nums',   None)
    base_flexions_raw = getattr(args, 'base_flexions', None)

    if series_nums_raw:
        series_nums   = [int(x)   for x in series_nums_raw.split(',')]
        if base_flexions_raw:
            base_flexions = [float(x) for x in base_flexions_raw.split(',')]
        else:
            # base_flexion 未指定時: series 数に関わらず 90° をデフォルトに
            base_flexions = [90.0] * len(series_nums)
        if len(series_nums) != len(base_flexions):
            raise ValueError("--series_nums と --base_flexions の要素数が一致しません")
    else:
        series_nums   = [None]
        base_flexions = [90.0]

    # ── 全シリーズを読み込む ──
    volumes_data = []
    for sn, bf in zip(series_nums, base_flexions):
        label_str = f"シリーズ {sn}" if sn is not None else "（シリーズ全体）"
        print(f"\nLoading CT volume {label_str}  base_flexion={bf}°  ...")
        vol, _, lat, vox_mm = load_ct_volume(
            args.ct_dir, laterality=laterality,
            series_num=sn, hu_min=hu_min, hu_max=hu_max,
            target_size=args.target_size,
        )
        print(f"  実効ボクセルサイズ: {vox_mm:.2f} mm/voxel, SID: {sid_mm:.0f} mm")
        print("  ランドマーク自動検出中...")
        lm = auto_detect_landmarks(vol, laterality=lat)
        volumes_data.append({
            'volume':       vol,
            'landmarks':    lm,
            'laterality':   lat,
            'voxel_mm':     vox_mm,
            'base_flexion': bf,
        })

    print(f"\n  (手動上書きする場合は DEFAULT_LANDMARKS_NORMALIZED を編集してください)")

    # AP: base_flexion >= 150° のボリュームのみ使用（伸展位のCT）
    ap_vols  = [v for v in volumes_data if v['base_flexion'] >= 150.0]
    if not ap_vols:
        # 閾値内がなければ最も伸展したものだけ使用
        ap_vols = [max(volumes_data, key=lambda x: x['base_flexion'])]

    # LAT: base_flexion <= 120° のボリュームのみ使用（屈曲位のCT）
    lat_vols = [v for v in volumes_data if v['base_flexion'] <= 120.0]
    if not lat_vols:
        lat_vols = [min(volumes_data, key=lambda x: x['base_flexion'])]

    print(f"\n  AP使用シリーズ:  base_flexion={[v['base_flexion'] for v in ap_vols]}")
    print(f"  LAT使用シリーズ: base_flexion={[v['base_flexion'] for v in lat_vols]}")

    out_dir = args.out_dir
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # train/val分割比率（YAML設定対応）
    val_ratio = 1.0 - getattr(args, 'train_val_split', 0.85)

    summary_rows = []
    img_idx = 0

    def apply_domain_aug(drr_bgr: np.ndarray) -> np.ndarray:
        """DRR画像に実X線らしさを付与するaugmentation。--domain_aug 指定時のみ使用。"""
        img = drr_bgr.copy().astype(np.float32)
        # 1. ガウスノイズ（X線量子ノイズ相当）
        noise = np.random.normal(0, random.uniform(3, 12), img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)
        # 2. コントラスト・輝度ランダム変動
        alpha = random.uniform(0.75, 1.25)   # コントラスト
        beta  = random.uniform(-20, 20)      # 輝度
        img = np.clip(alpha * img + beta, 0, 255)
        # 3. ガンマ補正（軟部組織コントラスト変動）
        gamma = random.uniform(0.7, 1.4)
        inv_gamma = 1.0 / gamma
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
        img = lut[img.astype(np.uint8)]
        # 4. 軽微なガウスブラー（焦点ボケ相当）
        if random.random() < 0.4:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)
        # 5. ヒストグラムマッチング（実X線の輝度分布に寄せる、50%の確率で適用）
        if random.random() < 0.5:
            img = _histogram_match_to_real(img.astype(np.uint8)).astype(np.float32)
        return img.astype(np.uint8)

    # 実X線の参照ヒストグラム（累積分布関数）をキャッシュ
    _real_xray_cdfs = {}
    real_xray_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "real_xray", "images")
    for _rxf in ["008_AP.png", "008_LAT.png"]:
        _rxp = os.path.join(real_xray_dir, _rxf)
        if os.path.exists(_rxp):
            _rx = cv2.imread(_rxp, cv2.IMREAD_GRAYSCALE)
            _h = cv2.calcHist([_rx], [0], None, [256], [0, 256]).ravel()
            _cdf = _h.cumsum()
            _cdf = _cdf / (_cdf[-1] + 1e-8)
            _real_xray_cdfs[_rxf] = _cdf
    _real_cdf_list = list(_real_xray_cdfs.values()) if _real_xray_cdfs else []

    def _histogram_match_to_real(img_u8: np.ndarray) -> np.ndarray:
        """DRRのヒストグラムを実X線画像にマッチング"""
        if not _real_cdf_list:
            return img_u8
        ref_cdf = random.choice(_real_cdf_list)
        src_hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).ravel()
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / (src_cdf[-1] + 1e-8)
        lut = np.zeros(256, dtype=np.uint8)
        for s in range(256):
            lut[s] = np.argmin(np.abs(ref_cdf - src_cdf[s]))
        return lut[img_u8]

    def save_sample(drr, label_str, meta: dict, split: str):
        nonlocal img_idx
        fname = f"elbow_{img_idx:05d}"
        out_img = apply_domain_aug(drr) if args.domain_aug else drr
        cv2.imwrite(os.path.join(out_dir, "images", split, f"{fname}.png"), out_img)
        with open(os.path.join(out_dir, "labels", split, f"{fname}.txt"), "w") as f:
            f.write(label_str)
        summary_rows.append({"filename": f"{fname}.png", "split": split, **meta})
        img_idx += 1

    # ── AP 像生成 ──
    # base_flexion が最大のボリュームをサイクルで使用（最も伸展した CT が AP に最適）
    if "AP" in args.views:
        print(f"\nGenerating {args.n_ap} AP DRRs...")
        for i in range(args.n_ap):
            vd  = ap_vols[i % len(ap_vols)]
            rotation_err = random.uniform(-25.0, 25.0)
            valgus_deg   = random.uniform(-10.0, 10.0)
            # 目標屈曲角: CT実ポジション周辺 ±10°（AP範囲 150〜180°にクランプ）
            flex_center  = max(150.0, min(180.0, vd['base_flexion']))
            flexion      = max(150.0, min(180.0, random.uniform(flex_center - 10.0, flex_center + 10.0)))

            rot_vol, rot_lm = rotate_volume_and_landmarks(
                vd['volume'], vd['landmarks'], rotation_err, flexion,
                base_flexion=vd['base_flexion'], valgus_deg=valgus_deg,
            )
            drr     = generate_drr(rot_vol, axis="AP", sid_mm=sid_mm, voxel_mm=vd['voxel_mm'])
            drr_bgr = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)

            label = make_yolo_label(rot_lm, "AP", drr.shape[0], drr.shape[1],
                                    vol_shape=rot_vol.shape, sid_mm=sid_mm, voxel_mm=vd['voxel_mm'])
            split = "val" if i < int(args.n_ap * val_ratio) else "train"
            save_sample(drr_bgr, label, {
                "view_type":          "AP",
                "rotation_error_deg": round(rotation_err, 2),
                "flexion_deg":        round(flexion, 2),
                "base_flexion":       vd['base_flexion'],
                "carrying_angle":     0.0,
                "valgus_deg":         round(valgus_deg, 2),
            }, split)
    else:
        print("\nSkipping AP DRR generation (--views does not include AP)")

    # ── LAT 像生成 ──
    # base_flexion が最小のボリュームをサイクルで使用（最も屈曲した CT が LAT に最適）
    if "LAT" in args.views:
        print(f"Generating {args.n_lat} LAT DRRs...")
        for i in range(args.n_lat):
            vd  = lat_vols[i % len(lat_vols)]
            rotation_err = random.uniform(-30.0, 30.0)
            valgus_deg   = random.uniform(-8.0, 8.0)
            # 目標屈曲角: CT実ポジション周辺 ±20°（LAT範囲 60〜120°にクランプ）
            flex_center  = max(60.0, min(120.0, vd['base_flexion']))
            flexion      = max(60.0, min(120.0, random.uniform(flex_center - 20.0, flex_center + 20.0)))

            rot_vol, rot_lm = rotate_volume_and_landmarks(
                vd['volume'], vd['landmarks'], rotation_err, flexion,
                base_flexion=vd['base_flexion'], valgus_deg=valgus_deg,
            )
            drr     = generate_drr(rot_vol, axis="LAT", sid_mm=sid_mm, voxel_mm=vd['voxel_mm'])
            drr_bgr = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)

            label = make_yolo_label(rot_lm, "LAT", drr.shape[0], drr.shape[1],
                                    vol_shape=rot_vol.shape, sid_mm=sid_mm, voxel_mm=vd['voxel_mm'])
            split = "val" if i < int(args.n_lat * val_ratio) else "train"
            save_sample(drr_bgr, label, {
                "view_type":          "LAT",
                "rotation_error_deg": round(rotation_err, 2),
                "flexion_deg":        round(flexion, 2),
                "base_flexion":       vd['base_flexion'],
                "carrying_angle":     0.0,
                "valgus_deg":         round(valgus_deg, 2),
            }, split)
    else:
        print("Skipping LAT DRR generation (--views does not include LAT)")

    # dataset.yaml 生成
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("nc: 1\nnames: [elbow_joint]\n")
        # kpt_shape: [6, 3] — 6キーポイント × (x, y, visibility)
        # flip_idx: 水平フリップ時に lateral(1) ↔ medial(2) を入れ替え
        #   0=humerus_shaft, 1=lateral_epicondyle, 2=medial_epicondyle,
        #   3=forearm_shaft, 4=radial_head, 5=olecranon
        f.write("kpt_shape: [6, 3]\nflip_idx: [0, 2, 1, 3, 4, 5]\n")

    # CSV サマリー（YOLOラベル兼用）
    csv_path = os.path.join(out_dir, "dataset_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # ConvNeXt 訓練用ラベルCSV（角度回帰ターゲット）
    # カラム: filename, split, view_type, rotation_error_deg, flexion_deg, carrying_angle
    convnext_csv_path = os.path.join(out_dir, "convnext_labels.csv")
    convnext_fields = ["filename", "split", "view_type",
                       "rotation_error_deg", "flexion_deg", "carrying_angle",
                       "valgus_deg"]
    with open(convnext_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=convnext_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n完了: {img_idx}枚生成")
    print(f"  画像: {out_dir}/images/")
    print(f"  ラベル: {out_dir}/labels/")
    print(f"  サマリー: {csv_path}")
    print(f"  ConvNeXtラベル: {convnext_csv_path}")


# ─── エントリポイント ──────────────────────────────────────────────────────────

def _load_config_yaml(config_path: str) -> dict:
    """YAML設定ファイルを読み込む。PyYAMLが必要。"""
    if yaml is None:
        raise ImportError(
            "PyYAML が必要です: pip install pyyaml\n"
            "  --config を使わない場合は従来のCLI引数を使用してください。"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_config_to_namespace(ns: argparse.Namespace, config: dict):
    """YAML設定の値をNamespaceに反映する（CLI引数が未指定の場合のみ上書き）。"""
    # YAML key -> argparse dest のマッピング
    key_map = {
        'ct_dir':         'ct_dir',
        'out_dir':        'out_dir',
        'n_ap':           'n_ap',
        'n_lat':          'n_lat',
        'laterality':     'laterality',
        'sid':            'sid',
        'series_nums':    'series_nums',
        'base_flexions':  'base_flexions',
        'target_size':    'target_size',
        'domain_aug':     'domain_aug',
        'hu_min':         'hu_min',
        'hu_max':         'hu_max',
        'train_val_split': 'train_val_split',
        'views':          'views',
    }
    for yaml_key, attr_name in key_map.items():
        if yaml_key in config and not hasattr(ns, f'_cli_{attr_name}'):
            val = config[yaml_key]
            # laterality: YAMLではリスト [R, L] だが、CLI互換のため最初の値を使用
            if yaml_key == 'laterality' and isinstance(val, list):
                val = val[0] if val else None
            setattr(ns, attr_name, val)


def main():
    parser = argparse.ArgumentParser(description="ElbowSynth: 肘CT → DRR自動生成（6方向CT対応）")
    parser.add_argument("--config",       default=None,
                        help="YAML設定ファイルパス（configs/dataset_config.yaml）。CLI引数で上書き可能")
    parser.add_argument("--ct_dir",       default=None,
                        help="CT DICOMフォルダ（サブディレクトリも再帰検索）")
    parser.add_argument("--out_dir",      default=None, help="出力先フォルダ")
    parser.add_argument("--n_ap",         type=int, default=None, help="AP DRR生成枚数")
    parser.add_argument("--n_lat",        type=int, default=None, help="LAT DRR生成枚数")
    parser.add_argument("--laterality",   choices=['R', 'L'], default=None,
                        help="左右腕の指定（R=右腕 / L=左腕）。省略時はDICOMタグから自動検出")
    parser.add_argument("--sid",          type=float, default=None,
                        help="X線管焦点〜検出器間距離 mm（デフォルト 1000.0 = 100cm）")
    parser.add_argument("--series_nums",  default=None,
                        help="使用する SeriesNumber（カンマ区切りで複数指定可）例: '4,8,12'")
    parser.add_argument("--base_flexions", default=None,
                        help="各シリーズのCT撮影時の肘屈曲角°（series_nums と同数）例: '180,135,90'")
    parser.add_argument("--target_size",  type=int,   default=None,
                        help="CT ボリュームの最大辺サイズ（px）。高いほど高解像度DRR生成（デフォルト: 128）")
    parser.add_argument("--domain_aug",   action="store_true", default=None,
                        help="実X線らしさを追加するドメイン適応augmentationを適用")
    parser.add_argument("--hu_min",       type=float, default=None,
                        help="HUウィンドウ下限（ファントム推奨: -200、デフォルト: -400）")
    parser.add_argument("--hu_max",       type=float, default=None,
                        help="HUウィンドウ上限（ファントム推奨: 1000、デフォルト: 1500）")
    parser.add_argument("--views",        default="AP,LAT",
                        help="生成するビュー（カンマ区切り: AP,LAT / AP / LAT）")
    args = parser.parse_args()

    # YAML設定の読み込み・マージ
    if args.config:
        config = _load_config_yaml(args.config)
        print(f"設定ファイル読み込み: {args.config}")
        _apply_config_to_namespace(args, config)

    # デフォルト値の適用（CLI/YAMLどちらも未指定の場合）
    defaults = {
        'n_ap': 120, 'n_lat': 360, 'sid': 1000.0,
        'target_size': 128, 'domain_aug': False,
        'hu_min': -400.0, 'hu_max': 1500.0,
    }
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    # 必須引数チェック
    if not args.ct_dir or not args.out_dir:
        parser.error("--ct_dir と --out_dir は必須です（CLI引数またはYAML設定で指定）")

    # train_val_split をargsに保持（generate_dataset で使用可能）
    if not hasattr(args, 'train_val_split'):
        args.train_val_split = 0.85

    # --views をリストにパース
    args.views = [v.strip() for v in args.views.split(",")]

    generate_dataset(args)


if __name__ == "__main__":
    main()
