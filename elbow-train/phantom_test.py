"""
合成肘ファントムで ct_reorient.py のランドマーク検出精度を評価するスクリプト

【検証内容】
  - 上腕骨長軸（PCA）検出精度
  - 経顆軸（外側上顆・内側上顆）検出精度
  - 屈曲角 0 / 30 / 60° で変化を観察

使い方:
  python elbow-train/phantom_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ct_reorient import detect_humeral_axis, detect_transepicondylar_axis


# ─── 合成ファントム生成 ─────────────────────────────────────────────────────────

def make_cylinder(shape, center_mm, axis, radius_mm, half_len_mm, voxel_mm):
    """軸・半径・長さで円柱マスクを生成（mm座標系）"""
    Z, Y, X = shape
    zm, ym, xm = voxel_mm
    gz, gy, gx = np.meshgrid(
        np.arange(Z) * zm, np.arange(Y) * ym, np.arange(X) * xm, indexing="ij"
    )
    cz, cy, cx = center_mm
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)

    dz, dy, dx = gz - cz, gy - cy, gx - cx
    proj = dz * a[0] + dy * a[1] + dx * a[2]
    pz, py, px = dz - proj * a[0], dy - proj * a[1], dx - proj * a[2]
    return (pz**2 + py**2 + px**2 < radius_mm**2) & (np.abs(proj) < half_len_mm)


def make_ellipsoid(shape, center_mm, semi_axes_mm, voxel_mm):
    """楕円体マスクを生成（mm座標系）"""
    Z, Y, X = shape
    zm, ym, xm = voxel_mm
    gz, gy, gx = np.meshgrid(
        np.arange(Z) * zm, np.arange(Y) * ym, np.arange(X) * xm, indexing="ij"
    )
    cz, cy, cx = center_mm
    rz, ry, rx = semi_axes_mm
    return ((gz - cz) / rz) ** 2 + ((gy - cy) / ry) ** 2 + ((gx - cx) / rx) ** 2 < 1.0


def create_elbow_phantom(flex_deg=30.0):
    """
    合成肘ファントムを生成する。

    座標系 (Z, Y, X):
      - 上腕骨長軸  = Z軸 (1, 0, 0)
      - 経顆軸      = X軸 (0, 0, 1)
      - AP方向      = Y軸 (0, 1, 0)

    前腕（橈骨・尺骨）は ZY 平面内で flex_deg だけ屈曲させる。
    """
    shape = (220, 160, 160)
    voxel_mm = (1.5, 0.5, 0.5)   # (dz, dy, dx) mm
    Z, Y, X = shape
    zm, ym, xm = voxel_mm
    ctr = np.array([Z * zm / 2, Y * ym / 2, X * xm / 2])

    volume = np.full(shape, -1000.0, dtype=np.float32)

    # 1) 上腕骨シャフト（Z軸平行）
    shaft_ctr = ctr + np.array([-35, 0, 0])
    volume[make_cylinder(shape, shaft_ctr, [1, 0, 0], radius_mm=9, half_len_mm=55, voxel_mm=voxel_mm)] = 600.0

    # 2) 上顆部（楕円体: Z方向に薄く・X方向に広い = 経顆軸）
    condyle_ctr = ctr + np.array([25, 0, 0])   # 遠位
    volume[make_ellipsoid(shape, condyle_ctr, semi_axes_mm=(12, 9, 24), voxel_mm=voxel_mm)] = 500.0

    # 外側上顆・内側上顆の真値（mm）
    epic_lat_mm = condyle_ctr + np.array([0, 0, +24])
    epic_med_mm = condyle_ctr + np.array([0, 0, -24])

    # 3) 前腕骨（ZY平面内で屈曲）
    flex_rad = np.radians(flex_deg)
    fa = np.array([np.cos(flex_rad), np.sin(flex_rad), 0.0])  # 前腕の向き

    # 橈骨（外側 +X）
    r_ctr = condyle_ctr + fa * 42 + np.array([0, 0, +6])
    volume[make_cylinder(shape, r_ctr, fa, radius_mm=5, half_len_mm=40, voxel_mm=voxel_mm)] = 550.0

    # 尺骨（内側 -X）
    u_ctr = condyle_ctr + fa * 42 + np.array([0, 0, -7])
    volume[make_cylinder(shape, u_ctr, fa, radius_mm=6, half_len_mm=40, voxel_mm=voxel_mm)] = 520.0

    gt = {
        "humeral_axis": np.array([1.0, 0.0, 0.0]),
        "transepicondylar_axis": np.array([0.0, 0.0, 1.0]),
        "condyle_center_mm": condyle_ctr,
        "epic_lat_mm": epic_lat_mm,
        "epic_med_mm": epic_med_mm,
        "flex_deg": flex_deg,
    }
    return volume, gt, voxel_mm


# ─── 評価ユーティリティ ────────────────────────────────────────────────────────

def angle_between(a, b):
    """2ベクトルのなす角（度, 0〜90°に正規化）"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cos = abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def grade(angle_deg):
    if angle_deg < 3:
        return "OK"
    if angle_deg < 10:
        return "WARN"
    return "FAIL"


# ─── テスト本体 ────────────────────────────────────────────────────────────────

def run_one(flex_deg):
    print(f"\n{'='*55}")
    print(f"  屈曲角 {flex_deg}°")
    print(f"{'='*55}")

    volume, gt, voxel_mm = create_elbow_phantom(flex_deg=flex_deg)
    hu_threshold = 300.0
    voxel_spacing = voxel_mm

    # --- 上腕骨長軸 ---
    print("\n[上腕骨長軸 PCA]")
    humeral_axis, centroid = detect_humeral_axis(volume, hu_threshold, voxel_spacing)
    err_h = angle_between(humeral_axis, gt["humeral_axis"])
    print(f"  検出: {humeral_axis.round(3)}  真値: {gt['humeral_axis']}")
    print(f"  誤差: {err_h:.1f}°  [{grade(err_h)}]")

    # --- 経顆軸 ---
    print("\n[経顆軸 自動検出]")
    trans_axis = detect_transepicondylar_axis(
        volume, humeral_axis, centroid, hu_threshold, voxel_spacing
    )
    err_t = angle_between(trans_axis, gt["transepicondylar_axis"])
    print(f"  検出: {trans_axis.round(3)}  真値: {gt['transepicondylar_axis']}")
    print(f"  誤差: {err_t:.1f}°  [{grade(err_t)}]")

    sign_ok = humeral_axis[0] < 0   # シャフトが近位（z < condyle_z）なので常に負のはず
    print(f"  符号チェック: axis[0]={humeral_axis[0]:+.3f} → {'OK（負）' if sign_ok else 'NG（正）'}")

    return {"flex_deg": flex_deg, "humeral_err": err_h, "trans_err": err_t,
            "humeral_axis": humeral_axis, "sign_ok": sign_ok}


def main():
    print("=== 合成肘ファントム 検出精度テスト ===")
    results = []
    for flex in [0, 30, 60, 75, 90]:
        results.append(run_one(flex))

    print(f"\n{'='*55}")
    print("  サマリ")
    print(f"{'='*55}")
    print(f"{'屈曲角':>6}  {'上腕骨長軸誤差':>14}  {'経顆軸誤差':>12}")
    print("-" * 40)
    for r in results:
        print(
            f"  {r['flex_deg']:>3}°  "
            f"{r['humeral_err']:>8.1f}° [{grade(r['humeral_err'])}]  "
            f"{r['trans_err']:>8.1f}° [{grade(r['trans_err'])}]"
        )
    sign_fails = [r for r in results if not r['sign_ok']]
    print(f"  符号の一貫性: {'全ケース OK' if not sign_fails else f'{len(sign_fails)} ケースで符号NG'}")
    print()
    fails = [r for r in results if grade(r['humeral_err']) == 'FAIL' or grade(r['trans_err']) == 'FAIL']
    warns = [r for r in results if grade(r['humeral_err']) == 'WARN' or grade(r['trans_err']) == 'WARN']
    if not fails and not warns:
        print("  精度: 全ケース OK")
    elif not fails:
        print(f"  精度: WARN {len(warns)} ケース（FAIL なし）")
    else:
        print(f"  精度: FAIL {len(fails)} ケース")


if __name__ == "__main__":
    main()
