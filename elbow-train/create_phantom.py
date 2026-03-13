#!/usr/bin/env python3
"""
create_phantom.py — 解剖学的肘CT ファントムを DICOM シリーズとして生成

座標系 (DICOM LPS):
  X+ = Left（患者の左）= 右腕の内側（medial）
  Y+ = Posterior（後方）
  Z+ = Superior（上方 = 近位/肩側）
  スライス: k=0（遠位/手側）→ k=179（近位/肩側）

生成構造:
  上腕骨（骨幹部 + 内外側上顆 + 滑車 + 小頭）
  橈骨（橈骨頭 + 骨幹部）
  尺骨（肘頭 + 骨幹部）
  軟部組織（筋肉 + 脂肪）

使い方:
  python elbow-train/create_phantom.py --out_dir data/raw_dicom/ct/ --laterality R
"""

import argparse
import datetime
import os

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter

# ─── ボリューム定義 ────────────────────────────────────────────────────────────

NX, NY, NZ = 256, 256, 180   # voxels
PX, PY, PZ = 0.5, 0.5, 1.0  # mm/voxel

HU = {
    'air':    -1000,
    'fat':      -80,
    'muscle':    50,
    'cancel':   350,   # 海綿骨（cancellous）
    'cortex':   800,   # 皮質骨（cortical）
    'marrow':   -80,   # 髄腔
}

CX, CY = NX // 2, NY // 2   # 画像中心 (128, 128)

# ─── 3D グリッド（グローバル） ─────────────────────────────────────────────────

_KK = _II = _JJ = None
_ii2 = _jj2 = None   # 2Dスライス用グリッド


def _init_grid():
    global _KK, _II, _JJ, _ii2, _jj2
    _KK, _II, _JJ = np.mgrid[0:NZ, 0:NY, 0:NX].astype(np.float32)
    _ii2, _jj2 = np.mgrid[0:NY, 0:NX].astype(np.float32)


# ─── マスク生成 ────────────────────────────────────────────────────────────────

def cyl(cj, ci, rj, ri, k0, k1):
    """Z軸方向の楕円柱マスク（3D）"""
    return (
        ((_JJ - cj) / rj) ** 2 + ((_II - ci) / ri) ** 2 <= 1.0
    ) & (_KK >= k0) & (_KK <= k1)


def ell(cj, ci, ck, rj, ri, rk):
    """楕円体マスク（3D）"""
    return (
        ((_JJ - cj) / rj) ** 2 +
        ((_II - ci) / ri) ** 2 +
        ((_KK - ck) / rk) ** 2
    ) <= 1.0


def ell2(cj, ci, rj, ri):
    """楕円マスク（2Dスライス用）"""
    return ((_jj2 - cj) / rj) ** 2 + ((_ii2 - ci) / ri) ** 2 <= 1.0


def shell(vol, outer, inner, cortex_hu, fill_hu):
    """皮質骨殻 + 内部充填"""
    vol[outer & ~inner] = cortex_hu
    vol[inner] = fill_hu


# ─── ファントム生成 ────────────────────────────────────────────────────────────

def build_phantom(laterality: str = 'R') -> np.ndarray:
    """
    HU値の3D配列（NZ, NY, NX）を生成する。

    右腕（R）: X+(j大) = patient Left = medial（内側）
    左腕（L）: X+(j大) = patient Left = lateral（外側）
               → s=-1 で内外側の位置を反転

    実寸（右腕の解剖：AP位）:
      骨幹部径 ~24mm（皮質3mm）
      内側上顆: 骨幹中心から内側24mm + 外側20mm、Z=64mm付近
      外側上顆: 骨幹中心から外側18mm + 外側16mm、Z=65mm付近
      橈骨頭径: ~22mm、外側28mm、Z=60-68mm
      尺骨肘頭: 後方38mm、Z=67mm付近
    """
    _init_grid()
    vol = np.full((NZ, NY, NX), HU['air'], dtype=np.float32)

    # 右腕: medial = X+ = j大 (s=+1)
    # 左腕: medial = X- = j小 (s=-1)
    s = 1 if laterality == 'R' else -1

    # ── 軟部組織（腕の輪郭） ──────────────────────────────────────────────────
    # 楕円断面: 左右92px=46mm, 前後76px=38mm
    vol[cyl(CX, CY, 92, 76, 0, NZ - 1)] = HU['muscle']
    # 皮下脂肪リング
    vol[cyl(CX, CY, 92, 76, 0, NZ - 1) & ~cyl(CX, CY, 80, 65, 0, NZ - 1)] = HU['fat']

    # ── 上腕骨幹部（k=82〜179: 近位 97mm） ───────────────────────────────────
    # 断面中心を少し前方（前腕筋との関係）
    hx, hy = CX, CY - 10

    shell(
        vol,
        cyl(hx, hy, 24, 22, 82, 179),   # 外径: 24x22 px = 12x11mm
        cyl(hx, hy, 16, 14, 82, 179),   # 髄腔: 16x14 px = 8x7mm
        HU['cortex'], HU['marrow']
    )

    # ── 上腕骨遠位フレア（k=60〜84: 骨幹→上顆への移行） ────────────────────
    # k が小さくなる（遠位へ）につれ、内外側に広がる
    for ki in range(60, 84):
        t = (83 - ki) / 23.0   # 0.0(k=83)→1.0(k=60): 遠位ほどt大

        # 内側（trochlea/滑車寄り）: s=+1なら j大方向に張り出す
        tj = CX + int(s * t * 26)
        tr_j = int(24 + t * 14)
        tr_i = int(22 + t * 4)
        sl = vol[ki]
        outer_t = ell2(tj, hy, tr_j, tr_i)
        inner_t = ell2(tj, hy, tr_j - 8, tr_i - 6)
        sl[outer_t & ~inner_t] = HU['cortex']
        sl[inner_t] = HU['cancel']

        # 外側（capitulum/小頭寄り）: s=+1なら j小方向に張り出す
        cj = CX - int(s * t * 20)
        cr_j = int(20 + t * 10)
        cr_i = int(20 + t * 4)
        outer_c = ell2(cj, hy - 2, cr_j, cr_i)
        inner_c = ell2(cj, hy - 2, cr_j - 7, cr_i - 5)
        sl[outer_c & ~inner_c] = HU['cortex']
        sl[inner_c] = HU['cancel']

    # ── 内側上顆（medial epicondyle）────────────────────────────────────────
    # 内側上顆: 大きく突出（臨床的に重要）
    # 右腕: j大方向(X+) = CX + 48 = 176
    mej = CX + s * 48
    shell(
        vol,
        ell(mej, CY + 6, 64, 22, 17, 19),   # 外径: 22x17x19px = 11x8.5x19mm
        ell(mej, CY + 6, 64, 13, 10, 12),
        HU['cortex'], HU['cancel']
    )

    # ── 外側上顆（lateral epicondyle）───────────────────────────────────────
    # 外側上顆: 内側より小さい
    # 右腕: j小方向(X-) = CX - 36 = 92
    lej = CX - s * 36
    shell(
        vol,
        ell(lej, CY - 4, 65, 17, 14, 16),
        ell(lej, CY - 4, 65, 10,  8, 10),
        HU['cortex'], HU['cancel']
    )

    # ── 滑車（trochlea）: 内側下端 ────────────────────────────────────────────
    trj = CX + s * 24
    shell(
        vol,
        ell(trj, CY + 10, 57, 30, 19, 16),
        ell(trj, CY + 10, 57, 20, 11,  9),
        HU['cortex'], HU['cancel']
    )

    # ── 小頭（capitulum）: 外側下端 ──────────────────────────────────────────
    caj = CX - s * 22
    shell(
        vol,
        ell(caj, CY - 6, 58, 22, 17, 14),
        ell(caj, CY - 6, 58, 13, 10,  8),
        HU['cortex'], HU['cancel']
    )

    # ── 橈骨（radius）── 外側（lateral）────────────────────────────────────
    # 右腕: j小方向 CX - s*28 = 100
    rj = CX - s * 28

    # 橈骨頭（disc状）: k=52〜68
    shell(
        vol,
        cyl(rj, CY - 8, 22, 22, 52, 68),
        cyl(rj, CY - 8, 14, 14, 52, 68),
        HU['cortex'], HU['cancel']
    )
    # 橈骨頸（細い）: k=44〜54
    shell(
        vol,
        cyl(rj, CY - 6, 11, 11, 44, 54),
        cyl(rj, CY - 6,  7,  7, 44, 54),
        HU['cortex'], HU['marrow']
    )
    # 橈骨幹部: k=0〜46（わずかに外側に傾く）
    shell(
        vol,
        cyl(rj - s * 4, CY - 4, 18, 16,  0, 46),
        cyl(rj - s * 4, CY - 4, 10,  9,  0, 46),
        HU['cortex'], HU['marrow']
    )

    # ── 尺骨（ulna）── 内側後方 ──────────────────────────────────────────────
    # 右腕: j大方向（内側）かつ後方 CX + s*14 = 142, CY+20=148
    uj = CX + s * 14
    ui = CY + 20

    # 肘頭（olecranon）: 大きな後方突起 k=58〜78
    shell(
        vol,
        ell(uj, CY + 38, 68, 17, 25, 22),
        ell(uj, CY + 38, 68, 10, 17, 14),
        HU['cortex'], HU['cancel']
    )
    # 尺骨近位部（鈎状突起含む）: k=50〜67
    shell(
        vol,
        cyl(uj, ui, 19, 17, 50, 67),
        cyl(uj, ui, 11, 10, 50, 67),
        HU['cortex'], HU['cancel']
    )
    # 尺骨幹部: k=0〜52
    shell(
        vol,
        cyl(uj + s * 2, ui, 17, 15,  0, 52),
        cyl(uj + s * 2, ui,  9,  8,  0, 52),
        HU['cortex'], HU['marrow']
    )

    # ── スムージング（骨の境界を滑らかに） ─────────────────────────────────
    vol = gaussian_filter(vol, sigma=0.8)

    return vol


# ─── DICOM 書き出し ───────────────────────────────────────────────────────────

def write_dicom_series(vol: np.ndarray, output_dir: str, laterality: str = 'R'):
    """
    3Dボリューム（NZ, NY, NX）をDICOMシリーズとして書き出す。

    画像ジオメトリ（標準アキシャル）:
      ImageOrientationPatient = [1,0,0, 0,1,0]
        → row方向（j増加）= X+（患者の左）
        → col方向（i増加）= Y+（患者の後方）
      Z: k=0 が遠位端（Z=0mm）、k増加で近位（Z+=Superior）
    """
    os.makedirs(output_dir, exist_ok=True)

    nz, ny, nx = vol.shape
    study_uid  = generate_uid()
    series_uid = generate_uid()
    now = datetime.datetime.now()

    print(f"  書き出し中: {nz} スライス → {output_dir}")
    for k in range(nz):
        z_pos = float(k) * PZ   # mm (Z+= proximal/superior)

        # ── ファイルメタ ──────────────────────────────────────────────────────
        file_meta = FileMetaDataset()
        sop_uid   = generate_uid()
        file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'  # CT Storage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID          = '1.2.840.10008.1.2.1'  # Explicit VR LE

        # ── データセット ──────────────────────────────────────────────────────
        fname = os.path.join(output_dir, f'slice_{k:04d}.dcm')
        ds = FileDataset(fname, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # 患者情報
        ds.PatientName = 'ElbowPhantom^Synthetic'
        ds.PatientID   = f'PHANTOM_{laterality}'
        ds.PatientSex  = 'O'
        ds.PatientAge  = '000Y'

        # スタディ
        ds.StudyDate        = now.strftime('%Y%m%d')
        ds.StudyTime        = now.strftime('%H%M%S')
        ds.StudyInstanceUID = study_uid
        ds.StudyDescription = f'Synthetic Elbow Phantom ({laterality} arm)'
        ds.AccessionNumber  = ''

        # シリーズ
        ds.Modality          = 'CT'
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber      = '1'
        ds.SeriesDescription = f'Elbow Phantom {laterality}'
        ds.BodyPartExamined  = 'ELBOW'
        ds.Laterality        = laterality
        ds.ImageLaterality   = laterality

        # インスタンス
        ds.SOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'
        ds.SOPInstanceUID = sop_uid
        ds.InstanceNumber = str(k + 1)

        # 画像ジオメトリ
        # row_cos=[1,0,0] (X+方向), col_cos=[0,1,0] (Y+方向)
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient    = [
            -(nx * PX / 2.0),   # X: 左端（患者の右端）
            -(ny * PY / 2.0),   # Y: 前端
            z_pos,              # Z: 現スライスの高さ（mm）
        ]
        ds.PixelSpacing   = [PY, PX]    # row/col spacing (mm)
        ds.SliceThickness = PZ
        ds.SliceLocation  = z_pos

        # 画像属性
        ds.Rows    = ny
        ds.Columns = nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated    = 16
        ds.BitsStored       = 16
        ds.HighBit          = 15
        ds.PixelRepresentation = 1   # signed int16

        # HUをそのまま格納（RescaleIntercept=0, RescaleSlope=1）
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope     = 1.0
        ds.RescaleType      = 'HU'
        ds.WindowCenter     = 400
        ds.WindowWidth      = 1800

        # ピクセルデータ（HU値をint16で格納）
        slice_data = np.clip(vol[k], -32768, 32767).astype(np.int16)
        ds.PixelData = slice_data.tobytes()
        ds.is_implicit_VR   = False
        ds.is_little_endian = True

        pydicom.dcmwrite(fname, ds)

        if k % 45 == 0 or k == nz - 1:
            print(f"    [{k+1}/{nz}] Z={z_pos:.0f}mm")

    print(f"  完了: {nz} スライス")


# ─── エントリポイント ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ElbowPhantomGenerator: 解剖学的肘CT ファントムを DICOM で生成'
    )
    parser.add_argument('--out_dir',    default='data/raw_dicom/ct/',
                        help='出力先フォルダ（デフォルト: data/raw_dicom/ct/）')
    parser.add_argument('--laterality', choices=['R', 'L'], default='R',
                        help='左右腕（R=右腕 / L=左腕）')
    args = parser.parse_args()

    print(f"ElbowPhantom 生成開始 ({args.laterality}腕)")
    print(f"  ボリューム: {NX}x{NY}x{NZ} voxels, {PX}x{PY}x{PZ} mm/voxel")
    print(f"  物理サイズ: {NX*PX:.0f}x{NY*PY:.0f}x{NZ*PZ:.0f} mm")

    print("\nボリューム構築中...")
    vol = build_phantom(args.laterality)
    print(f"  HU範囲: {vol.min():.0f} 〜 {vol.max():.0f}")

    print("\nDICOM書き出し中...")
    write_dicom_series(vol, args.out_dir, args.laterality)

    print(f"\n完了: {args.out_dir}")
    print("次のステップ:")
    print(f"  python elbow-train/elbow_synth.py \\")
    print(f"    --ct_dir {args.out_dir} \\")
    print(f"    --out_dir data/yolo_dataset/ \\")
    print(f"    --laterality {args.laterality}")


if __name__ == '__main__':
    main()
