"""
ct_reorient.py のユニットテスト

CT座標系変換・解剖学的軸検出のロジックをテストする。
実DICOMは不要で、NumPyで生成したボリュームで検証する。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "elbow-train"))
from ct_reorient import (
    correct_scan_direction,
    detect_humeral_axis,
    build_anatomical_rotation,
    apply_rotation,
    rotate_around_long_axis,
    generate_drr,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def _make_bone_volume(shape=(64, 48, 48), bone_hu=500.0, bg_hu=-500.0):
    """
    上腕骨を模した円柱状のボリュームを生成。
    z軸（axis=0）に沿って骨が伸びる。
    """
    nz, ny, nx = shape
    vol = np.full(shape, bg_hu, dtype=np.float32)
    cy, cx = ny // 2, nx // 2
    radius = min(ny, nx) // 6

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if (y - cy) ** 2 + (x - cx) ** 2 < radius ** 2:
                    vol[z, y, x] = bone_hu

    return vol


def _make_elbow_volume(shape=(80, 48, 48), bone_hu=500.0, bg_hu=-500.0):
    """
    上腕骨+前腕骨を模したボリューム（上半分:細い円柱, 中央:太い関節, 下半分:細い円柱）
    """
    nz, ny, nx = shape
    vol = np.full(shape, bg_hu, dtype=np.float32)
    cy, cx = ny // 2, nx // 2
    r_shaft = min(ny, nx) // 8
    r_joint = min(ny, nx) // 4

    for z in range(nz):
        dist_from_center = abs(z - nz // 2)
        r = r_joint if dist_from_center < nz // 10 else r_shaft

        for y in range(ny):
            for x in range(nx):
                if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                    vol[z, y, x] = bone_hu

    return vol


# ─── correct_scan_direction テスト ──────────────────────────────────────────


class TestCorrectScanDirection:
    def test_hfs_no_change(self):
        """HFS（インスキャン）ではボリュームが変化しない"""
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        info = {"is_feet_first": False}
        result = correct_scan_direction(vol, info)
        np.testing.assert_array_equal(result, vol)

    def test_ffs_reverses_z(self):
        """FFS（アウトスキャン）ではz軸が反転する"""
        vol = np.arange(24).reshape(4, 3, 2).astype(np.float32)
        info = {"is_feet_first": True}
        result = correct_scan_direction(vol, info)
        np.testing.assert_array_equal(result[0], vol[3])
        np.testing.assert_array_equal(result[3], vol[0])

    def test_double_flip_identity(self):
        """2回反転で元に戻る"""
        vol = np.random.rand(8, 4, 4).astype(np.float32)
        info = {"is_feet_first": True}
        result = correct_scan_direction(correct_scan_direction(vol, info), info)
        np.testing.assert_array_equal(result, vol)


# ─── detect_humeral_axis テスト ──────────────────────────────────────────────


class TestDetectHumeralAxis:
    def test_z_aligned_bone(self):
        """z方向に伸びる円柱の主軸はz方向に近い"""
        vol = _make_bone_volume(shape=(80, 32, 32))
        axis, centroid = detect_humeral_axis(
            vol, hu_threshold=0.0,
            voxel_spacing=(1.0, 1.0, 1.0),
        )
        # z方向成分が最も大きいはず
        assert abs(axis[0]) > abs(axis[1])
        assert abs(axis[0]) > abs(axis[2])

    def test_axis_is_unit_vector(self):
        """検出される軸は正規化されている（ほぼ単位ベクトル）"""
        vol = _make_bone_volume()
        axis, _ = detect_humeral_axis(vol, hu_threshold=0.0, voxel_spacing=(1.0, 1.0, 1.0))
        np.testing.assert_allclose(np.linalg.norm(axis), 1.0, atol=0.01)

    def test_centroid_within_volume(self):
        """重心がボリューム内にある"""
        shape = (64, 48, 48)
        vol = _make_bone_volume(shape=shape)
        _, centroid = detect_humeral_axis(vol, hu_threshold=0.0, voxel_spacing=(1.0, 1.0, 1.0))
        for i in range(3):
            assert 0 <= centroid[i] <= shape[i]

    def test_low_bone_voxels_fallback(self):
        """骨ボクセルが少ない場合のフォールバック"""
        vol = np.full((32, 32, 32), -500.0, dtype=np.float32)
        # 骨ボクセルを50個だけ配置（< 200）
        vol[10:12, 15:17, 15:20] = 500.0
        axis, centroid = detect_humeral_axis(vol, hu_threshold=0.0, voxel_spacing=(1.0, 1.0, 1.0))
        # フォールバック: [1,0,0], 中心
        np.testing.assert_array_equal(axis, [1.0, 0.0, 0.0])

    def test_anisotropic_spacing(self):
        """異方性ボクセルスペーシングでも正常に動作する"""
        vol = _make_bone_volume(shape=(40, 48, 48))
        axis, _ = detect_humeral_axis(
            vol, hu_threshold=0.0,
            voxel_spacing=(2.0, 0.5, 0.5),  # z方向が粗い
        )
        # mm空間ではz方向が40*2=80mmと長いので主軸はz方向
        assert abs(axis[0]) > 0.5

    def test_elbow_volume_with_joint(self):
        """関節部分が太いボリュームでもクラッシュしない"""
        vol = _make_elbow_volume()
        axis, centroid = detect_humeral_axis(
            vol, hu_threshold=0.0, voxel_spacing=(1.0, 1.0, 1.0)
        )
        assert axis.shape == (3,)
        assert np.all(np.isfinite(axis))


# ─── build_anatomical_rotation テスト ───────────────────────────────────────


class TestBuildAnatomicalRotation:
    def test_output_is_orthogonal(self):
        """出力の回転行列は直交行列"""
        h_axis = np.array([1.0, 0.0, 0.0])
        t_axis = np.array([0.0, 0.0, 1.0])
        rot = build_anatomical_rotation(h_axis, t_axis)
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        """行列式が1（回転行列であること）"""
        h_axis = np.array([1.0, 0.2, 0.1])
        t_axis = np.array([0.1, -0.1, 1.0])
        rot = build_anatomical_rotation(h_axis, t_axis)
        assert abs(np.linalg.det(rot) - 1.0) < 1e-10

    def test_gram_schmidt_orthogonalization(self):
        """経顆軸がhumeral_axisと直交でなくても正しく直交化される"""
        h_axis = np.array([1.0, 0.0, 0.0])
        t_axis = np.array([0.3, 0.1, 0.9])  # humeral_axisに完全に垂直ではない
        rot = build_anatomical_rotation(h_axis, t_axis)
        # 列ベクトル同士の内積が0（直交）
        np.testing.assert_allclose(rot[:, 0] @ rot[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(rot[:, 0] @ rot[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(rot[:, 1] @ rot[:, 2], 0.0, atol=1e-10)

    def test_identity_when_axes_aligned(self):
        """軸がすでに標準基底と一致する場合の動作確認"""
        h_axis = np.array([1.0, 0.0, 0.0])
        t_axis = np.array([0.0, 1.0, 0.0])
        rot = build_anatomical_rotation(h_axis, t_axis)
        # z_hat=[1,0,0], x_hat=[0,1,0], y_hat=cross([1,0,0],[0,1,0])=[0,0,1]
        expected = np.column_stack([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(rot, expected, atol=1e-10)


# ─── apply_rotation テスト ──────────────────────────────────────────────────


class TestApplyRotation:
    def test_identity_rotation_preserves(self):
        """単位回転ではボリュームが保持される"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        centroid = np.array([16.0, 16.0, 16.0])
        result = apply_rotation(vol, np.eye(3), centroid)
        np.testing.assert_allclose(result, vol, atol=1e-3)

    def test_output_shape_preserved(self):
        """回転後の形状が入力と同じ"""
        vol = _make_bone_volume(shape=(32, 24, 24))
        centroid = np.array([16.0, 12.0, 12.0])
        rot = build_anatomical_rotation(
            np.array([1.0, 0.1, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        result = apply_rotation(vol, rot, centroid)
        assert result.shape == vol.shape

    def test_finite_values(self):
        """回転後の値にNaN/Infがない"""
        vol = _make_bone_volume(shape=(32, 24, 24))
        centroid = np.array([16.0, 12.0, 12.0])
        rot = build_anatomical_rotation(
            np.array([0.8, 0.2, 0.1]),
            np.array([0.0, 0.1, 0.9]),
        )
        result = apply_rotation(vol, rot, centroid)
        assert np.all(np.isfinite(result))


# ─── rotate_around_long_axis テスト ──────────────────────────────────────────


class TestRotateAroundLongAxis:
    def test_zero_angle_no_change(self):
        """0度回転ではボリュームが変化しない"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        result = rotate_around_long_axis(vol, 0.0)
        np.testing.assert_array_equal(result, vol)

    def test_small_angle_no_change(self):
        """0.5度未満の回転も変化なし"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        result = rotate_around_long_axis(vol, 0.4)
        np.testing.assert_array_equal(result, vol)

    def test_rotation_changes_volume(self):
        """大きな回転角では値が変わる"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        result = rotate_around_long_axis(vol, 45.0)
        assert not np.array_equal(result, vol)

    def test_shape_preserved(self):
        """回転後の形状が保持される"""
        vol = _make_bone_volume(shape=(40, 30, 30))
        result = rotate_around_long_axis(vol, 30.0)
        assert result.shape == vol.shape

    def test_360_approx_identity(self):
        """360度回転はほぼ元に戻る（補間誤差あり）"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        result = rotate_around_long_axis(vol, 360.0)
        # 補間誤差があるため完全一致ではないが、骨の位置は同じはず
        bone_orig = (vol > 0).sum()
        bone_rot = (result > 0).sum()
        assert abs(bone_orig - bone_rot) / bone_orig < 0.1


# ─── generate_drr テスト ──────────────────────────────────────────────────────


class TestGenerateDRR:
    def test_ap_projection(self):
        """AP投影（axis=1）の出力形状が正しい"""
        vol = _make_bone_volume(shape=(64, 48, 48))
        drr = generate_drr(vol, projection_axis=1, clahe=False)
        assert drr.shape == (64, 48)  # (Z, X)
        assert drr.dtype == np.uint8

    def test_lat_projection(self):
        """LAT投影（axis=2）の出力形状が正しい"""
        vol = _make_bone_volume(shape=(64, 48, 48))
        drr = generate_drr(vol, projection_axis=2, clahe=False)
        assert drr.shape == (64, 48)  # (Z, Y)
        assert drr.dtype == np.uint8

    def test_axial_projection(self):
        """軸位投影（axis=0）の出力形状が正しい"""
        vol = _make_bone_volume(shape=(64, 48, 48))
        drr = generate_drr(vol, projection_axis=0, clahe=False)
        assert drr.shape == (48, 48)  # (Y, X)

    def test_with_clahe(self):
        """CLAHE適用時もクラッシュしない"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        drr = generate_drr(vol, projection_axis=1, clahe=True)
        assert drr.dtype == np.uint8
        assert drr.shape == (32, 32)

    def test_bone_brighter(self):
        """骨領域は背景より明るい"""
        vol = _make_bone_volume(shape=(32, 32, 32))
        drr = generate_drr(vol, projection_axis=1, clahe=False)
        center = drr[16, 16]
        corner = drr[0, 0]
        assert center > corner

    def test_empty_volume(self):
        """空ボリュームでもクラッシュしない"""
        vol = np.full((16, 16, 16), -1000.0, dtype=np.float32)
        drr = generate_drr(vol, projection_axis=1, clahe=False)
        assert drr.shape == (16, 16)


# ─── 統合テスト ──────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self):
        """座標系変換→回転→DRR生成の一連のパイプライン"""
        vol = _make_elbow_volume(shape=(64, 48, 48))
        info = {"is_feet_first": False}

        # 1. スキャン方向補正
        vol = correct_scan_direction(vol, info)

        # 2. 上腕骨軸検出
        axis, centroid = detect_humeral_axis(
            vol, hu_threshold=0.0, voxel_spacing=(1.0, 1.0, 1.0)
        )

        # 3. 解剖学的座標系構築 + 回転
        t_axis = np.array([0.0, 0.0, 1.0])  # 仮の経顆軸
        rot = build_anatomical_rotation(axis, t_axis)
        vol_rotated = apply_rotation(vol, rot, centroid)

        # 4. DRR生成
        drr_ap = generate_drr(vol_rotated, projection_axis=1, clahe=True)
        drr_lat = generate_drr(vol_rotated, projection_axis=2, clahe=True)

        assert drr_ap.dtype == np.uint8
        assert drr_lat.dtype == np.uint8
        assert drr_ap.max() > 50  # 骨の投影がある

    def test_rotation_variation_pipeline(self):
        """回内外バリエーション生成のパイプライン"""
        vol = _make_bone_volume(shape=(48, 32, 32))

        drrs = []
        for angle in [0, 30, -30]:
            rotated = rotate_around_long_axis(vol, angle)
            drr = generate_drr(rotated, projection_axis=1, clahe=False)
            drrs.append(drr)

        # 異なる回転角ではDRRが異なるはず
        assert not np.array_equal(drrs[0], drrs[1])
        assert not np.array_equal(drrs[0], drrs[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
