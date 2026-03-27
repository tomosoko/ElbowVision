"""
ファントムCTのHU分布を解析し、骨/外殻/空気の閾値を特定するスクリプト
"""
import os
import sys
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_DIR = os.path.join(PROJECT_ROOT, "data/raw_dicom/ct")
OUT_DIR = os.path.join(PROJECT_ROOT, "results/domain_gap_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_hu_volume(dicom_dir, series_num=None):
    """DICOMからHUボリュームを読み込む（リサイズなし）"""
    dcm_paths = []
    for root, _, files in os.walk(dicom_dir):
        for f in sorted(files):
            if f.lower().endswith(('.dcm', '.dicom')):
                dcm_paths.append(os.path.join(root, f))

    if not dcm_paths:
        raise ValueError(f"No DICOM files in {dicom_dir}")

    slices = []
    for p in dcm_paths:
        ds = pydicom.dcmread(p)
        if series_num is not None and int(getattr(ds, 'SeriesNumber', -1)) != series_num:
            continue
        slices.append(ds)

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    def to_hu(s):
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, 'RescaleSlope', 1.0))
        intercept = float(getattr(s, 'RescaleIntercept', 0.0))
        return arr * slope + intercept

    volume = np.stack([to_hu(s) for s in slices])
    print(f"  Volume shape: {volume.shape}")
    print(f"  HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    return volume


def analyze_hu(volume, label=""):
    """HU分布をヒストグラムで解析"""
    flat = volume.flatten()

    # 空気(-1000付近)を除外した分布
    non_air = flat[flat > -500]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Phantom CT HU Distribution {label}", fontsize=14, fontweight="bold")

    # 全体ヒストグラム
    ax = axes[0, 0]
    ax.hist(flat, bins=500, range=(-1100, 2000), alpha=0.7, color="steelblue")
    ax.set_title("Full HU Histogram")
    ax.set_xlabel("HU")
    ax.set_ylabel("Voxel Count")
    ax.axvline(x=-200, color="red", linestyle="--", label="hu_min=-200 (current)")
    ax.axvline(x=1000, color="orange", linestyle="--", label="hu_max=1000 (current)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 空気除外・拡大ヒストグラム
    ax = axes[0, 1]
    ax.hist(non_air, bins=500, range=(-500, 2000), alpha=0.7, color="coral")
    ax.set_title("HU Histogram (air excluded, > -500)")
    ax.set_xlabel("HU")
    ax.set_ylabel("Voxel Count")
    ax.grid(True, alpha=0.3)

    # パーセンタイル表示
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pvals = np.percentile(non_air, percentiles)
    for p, v in zip(percentiles, pvals):
        if p in [5, 25, 50, 75, 95]:
            ax.axvline(x=v, color="green", linestyle=":", alpha=0.5)
            ax.text(v, ax.get_ylim()[1] * 0.9, f"P{p}={v:.0f}", fontsize=8, rotation=90)

    # 骨領域に絞ったヒストグラム（HU > 100）
    bone_region = flat[flat > 100]
    ax = axes[1, 0]
    if len(bone_region) > 0:
        ax.hist(bone_region, bins=300, range=(100, 2000), alpha=0.7, color="darkgreen")
        ax.set_title(f"Bone Region (HU > 100): {len(bone_region)} voxels ({100*len(bone_region)/len(flat):.1f}%)")
    else:
        ax.text(0.5, 0.5, "No voxels > 100 HU", transform=ax.transAxes, ha="center")
    ax.set_xlabel("HU")
    ax.set_ylabel("Voxel Count")
    ax.grid(True, alpha=0.3)

    # 外殻領域（-200〜200 HU付近 — プラスチック/ポリエチレン等）
    shell_region = flat[(flat > -200) & (flat < 200)]
    ax = axes[1, 1]
    if len(shell_region) > 0:
        ax.hist(shell_region, bins=200, range=(-200, 200), alpha=0.7, color="purple")
        ax.set_title(f"Shell Region (-200 < HU < 200): {len(shell_region)} voxels ({100*len(shell_region)/len(flat):.1f}%)")
    else:
        ax.text(0.5, 0.5, "No voxels in range", transform=ax.transAxes, ha="center")
    ax.set_xlabel("HU")
    ax.set_ylabel("Voxel Count")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, f"ct_hu_analysis{label}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存: {out_path}")

    # 統計サマリー
    print(f"\n  === HU Statistics {label} ===")
    print(f"  全ボクセル: {len(flat):,}")
    print(f"  非空気(>-500): {len(non_air):,} ({100*len(non_air)/len(flat):.1f}%)")
    for p, v in zip(percentiles, pvals):
        print(f"  P{p:2d}: {v:>8.1f} HU")

    # 素材推定
    print(f"\n  === 素材帯域推定 ===")
    air = flat[flat < -500]
    soft = flat[(flat >= -500) & (flat < 100)]
    bone = flat[flat >= 100]
    print(f"  空気   (< -500 HU): {len(air):>10,} voxels ({100*len(air)/len(flat):>5.1f}%)")
    print(f"  外殻等 (-500〜100): {len(soft):>10,} voxels ({100*len(soft)/len(flat):>5.1f}%)")
    print(f"  骨     (>= 100 HU): {len(bone):>10,} voxels ({100*len(bone)/len(flat):>5.1f}%)")

    return pvals


def main():
    print("ファントムCT HU分布解析")
    print("=" * 50)

    # シリーズ番号を探す
    dcm_paths = []
    for root, _, files in os.walk(CT_DIR):
        for f in sorted(files):
            if f.lower().endswith(('.dcm', '.dicom')):
                dcm_paths.append(os.path.join(root, f))

    if not dcm_paths:
        print(f"ERROR: DICOMファイルが見つかりません: {CT_DIR}")
        sys.exit(1)

    # 利用可能なシリーズを列挙
    series_set = set()
    for p in dcm_paths[:50]:  # 先頭50枚でシリーズを特定
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        sn = int(getattr(ds, 'SeriesNumber', -1))
        series_set.add(sn)

    print(f"  DICOM files: {len(dcm_paths)}")
    print(f"  Available series: {sorted(series_set)}")

    # 各シリーズを解析
    for sn in sorted(series_set):
        if sn < 0:
            continue
        print(f"\n{'='*50}")
        print(f"  Series {sn}")
        print(f"{'='*50}")
        try:
            vol = load_hu_volume(CT_DIR, series_num=sn)
            analyze_hu(vol, label=f"_series{sn}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
