"""elbow_synth.py の改善部分のテスト"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))
from elbow_synth import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotate_volume_and_landmarks,
    generate_drr,
    transform_landmarks_canonical,
)


class TestRotationMatrices:
    """回転行列の基本性質テスト"""

    def test_identity_at_zero(self):
        """0度回転は単位行列"""
        for func in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R = func(0.0)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonality(self):
        """回転行列は直交行列（R^T R = I）"""
        for deg in [15, 45, 90, 135, 180, -30]:
            for func in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = func(deg)
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        """回転行列の行列式は1"""
        for deg in [30, 90, -45]:
            for func in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = func(deg)
                assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestForearmSeparation:
    """前腕分離回転のテスト"""

    def _make_test_volume_and_landmarks(self):
        """テスト用ボリュームとランドマークを生成"""
        # 64x64x64 の簡易ボリューム（上腕部と前腕部に骨を模した値を配置）
        vol = np.zeros((64, 64, 64), dtype=np.float32)
        # 上腕骨（PD=5〜30, 中央付近）
        vol[5:30, 28:36, 28:36] = 0.5
        # 前腕骨（PD=35〜60, 中央付近）
        vol[35:60, 28:36, 28:36] = 0.5

        landmarks = {
            "joint_center": (0.5, 0.5, 0.5),  # PD=32: ちょうど中央
            "humerus_shaft": (0.15, 0.5, 0.5),  # PD=9.6: 上腕
            "lateral_epicondyle": (0.5, 0.5, 0.8),
            "medial_epicondyle": (0.5, 0.5, 0.2),
            "forearm_shaft": (0.75, 0.5, 0.5),  # PD=48: 前腕
        }
        return vol, landmarks

    def test_no_rotation_preserves_volume(self):
        """0度回転でボリュームが変化しない"""
        vol, lm = self._make_test_volume_and_landmarks()
        rotated_vol, rotated_lm = rotate_volume_and_landmarks(
            vol, lm, forearm_rotation_deg=0, flexion_deg=90, base_flexion=90
        )
        # 差分回転が0なのでほぼ同一（ブレンド処理のため完全一致ではない）
        np.testing.assert_allclose(rotated_vol, vol, atol=1e-5)

    def test_humerus_landmark_fixed(self):
        """上腕ランドマーク（humerus_shaft）は回転しても固定"""
        vol, lm = self._make_test_volume_and_landmarks()
        _, rotated_lm = rotate_volume_and_landmarks(
            vol, lm, forearm_rotation_deg=0, flexion_deg=120, base_flexion=90
        )
        assert rotated_lm["humerus_shaft"] == lm["humerus_shaft"]

    def test_forearm_landmark_moves(self):
        """前腕ランドマーク（forearm_shaft）は屈曲で移動する"""
        vol, lm = self._make_test_volume_and_landmarks()
        _, rotated_lm = rotate_volume_and_landmarks(
            vol, lm, forearm_rotation_deg=0, flexion_deg=120, base_flexion=90
        )
        # flexion 30度分回転するので前腕ランドマークは元と異なるはず
        assert rotated_lm["forearm_shaft"] != lm["forearm_shaft"]

    def test_upper_arm_voxels_preserved(self):
        """上腕部分のボクセル値は回転後もほぼ保持される"""
        vol, lm = self._make_test_volume_and_landmarks()
        rotated_vol, _ = rotate_volume_and_landmarks(
            vol, lm, forearm_rotation_deg=0, flexion_deg=120, base_flexion=90
        )
        # 上腕部（PD=5〜25, ブレンド帯より十分上腕側）
        upper_orig = vol[5:25, :, :]
        upper_rot = rotated_vol[5:25, :, :]
        np.testing.assert_allclose(upper_rot, upper_orig, atol=1e-5)

    def test_blend_mask_smooth(self):
        """ブレンド帯のボクセル値が0〜1の間でスムーズに遷移する"""
        vol, lm = self._make_test_volume_and_landmarks()
        rotated_vol, _ = rotate_volume_and_landmarks(
            vol, lm, forearm_rotation_deg=0, flexion_deg=150, base_flexion=90
        )
        # 回転後のボリュームにNaN/Infがないことを確認
        assert np.all(np.isfinite(rotated_vol))

    def test_large_flexion_range(self):
        """大きな屈曲角でもクラッシュしない"""
        vol, lm = self._make_test_volume_and_landmarks()
        for flex in [0, 45, 90, 135, 180]:
            rotated_vol, rotated_lm = rotate_volume_and_landmarks(
                vol, lm, forearm_rotation_deg=0, flexion_deg=flex, base_flexion=90
            )
            assert rotated_vol.shape == vol.shape
            assert np.all(np.isfinite(rotated_vol))


class TestGenerateDRR:
    """DRR生成のテスト"""

    def _make_simple_volume(self):
        """テスト用の簡易ボリューム（骨を模した円柱）"""
        vol = np.zeros((64, 48, 48), dtype=np.float32)
        # 中央に円柱状の骨
        for i in range(64):
            for j in range(48):
                for k in range(48):
                    if (j - 24) ** 2 + (k - 24) ** 2 < 100:
                        vol[i, j, k] = 0.5
        return vol

    def test_ap_drr_shape(self):
        """AP DRRの出力形状が正しい (PD, ML)"""
        vol = self._make_simple_volume()
        drr = generate_drr(vol, axis="AP")
        assert drr.shape == (64, 48)
        assert drr.dtype == np.uint8

    def test_lat_drr_shape(self):
        """LAT DRRの出力形状が正しい (PD, AP)"""
        vol = self._make_simple_volume()
        drr = generate_drr(vol, axis="LAT")
        assert drr.shape == (64, 48)
        assert drr.dtype == np.uint8

    def test_drr_has_contrast(self):
        """DRRにコントラストがある（全黒/全白ではない）"""
        vol = self._make_simple_volume()
        for axis in ["AP", "LAT"]:
            drr = generate_drr(vol, axis=axis)
            assert drr.max() > drr.min() + 50, f"{axis} DRR has no contrast"

    def test_background_is_dark(self):
        """背景（骨がない領域）は骨より暗い"""
        vol = self._make_simple_volume()
        drr = generate_drr(vol, axis="AP")
        # 角のピクセルは骨の中央より暗いはず
        corners = [drr[0, 0], drr[0, -1], drr[-1, 0], drr[-1, -1]]
        center_val = drr[32, 24]
        for c in corners:
            assert c < center_val, \
                f"Background ({c}) should be darker than bone ({center_val})"

    def test_bone_region_is_bright(self):
        """骨がある領域は明るい"""
        vol = self._make_simple_volume()
        drr = generate_drr(vol, axis="AP")
        # 中央付近は骨があるので明るいはず
        center_val = drr[32, 24]
        assert center_val > 100, f"Bone region too dark: {center_val}"

    def test_empty_volume(self):
        """空のボリュームでもクラッシュしない"""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        drr = generate_drr(vol, axis="AP")
        assert drr.shape == (32, 32)

    def test_beer_lambert_physics(self):
        """Beer-Lambert: 厚い骨はより白く映る"""
        # 薄い骨
        vol_thin = np.zeros((64, 48, 48), dtype=np.float32)
        vol_thin[20:44, 22:26, 20:28] = 0.5  # AP方向に4voxel厚

        # 厚い骨
        vol_thick = np.zeros((64, 48, 48), dtype=np.float32)
        vol_thick[20:44, 12:36, 20:28] = 0.5  # AP方向に24voxel厚

        drr_thin = generate_drr(vol_thin, axis="AP")
        drr_thick = generate_drr(vol_thick, axis="AP")

        # 厚い骨の中央は薄い骨より明るいはず（骨=白）
        thin_center = float(drr_thin[32, 24])
        thick_center = float(drr_thick[32, 24])
        assert thick_center >= thin_center, \
            f"Thick bone ({thick_center}) should be >= thin bone ({thin_center})"


class TestTransformLandmarksCanonical:
    """transform_landmarks_canonical のテスト."""

    def _lm(self, x, y, z):
        return {"A": (x, y, z)}

    def test_no_transform_returns_same(self):
        """転置[0,1,2]・反転なしなら元の値を返す."""
        lm = {"pt": (0.1, 0.2, 0.3)}
        result = transform_landmarks_canonical(lm, [0, 1, 2], [False, False, False])
        assert result["pt"] == (0.1, 0.2, 0.3)

    def test_flip_axis0(self):
        """axis0を反転: n → 1-n."""
        lm = {"pt": (0.2, 0.5, 0.8)}
        result = transform_landmarks_canonical(lm, [0, 1, 2], [True, False, False])
        assert abs(result["pt"][0] - 0.8) < 1e-10
        assert abs(result["pt"][1] - 0.5) < 1e-10
        assert abs(result["pt"][2] - 0.8) < 1e-10

    def test_transpose_order(self):
        """転置: new_axis_i = 元axis_transpose[i]."""
        lm = {"pt": (0.1, 0.5, 0.9)}
        # transpose [2, 0, 1]: new_i0 = old_2, new_i1 = old_0, new_i2 = old_1
        result = transform_landmarks_canonical(lm, [2, 0, 1], [False, False, False])
        assert abs(result["pt"][0] - 0.9) < 1e-10
        assert abs(result["pt"][1] - 0.1) < 1e-10
        assert abs(result["pt"][2] - 0.5) < 1e-10

    def test_multiple_landmarks(self):
        """複数ランドマークが全て変換される."""
        lm = {"A": (0.2, 0.3, 0.4), "B": (0.6, 0.7, 0.8)}
        result = transform_landmarks_canonical(lm, [0, 1, 2], [False, False, True])
        assert len(result) == 2
        assert abs(result["A"][2] - 0.6) < 1e-10
        assert abs(result["B"][2] - 0.2) < 1e-10

    def test_flip_all_axes(self):
        """全軸反転: (x,y,z) → (1-x, 1-y, 1-z)."""
        lm = {"pt": (0.2, 0.3, 0.4)}
        result = transform_landmarks_canonical(lm, [0, 1, 2], [True, True, True])
        assert abs(result["pt"][0] - 0.8) < 1e-10
        assert abs(result["pt"][1] - 0.7) < 1e-10
        assert abs(result["pt"][2] - 0.6) < 1e-10

    def test_empty_landmarks(self):
        """空の辞書は空を返す."""
        result = transform_landmarks_canonical({}, [0, 1, 2], [False, False, False])
        assert result == {}

    def test_transpose_then_flip(self):
        """転置後に反転が正しく適用される."""
        lm = {"pt": (0.1, 0.5, 0.9)}
        # transpose [2, 1, 0]: new = (0.9, 0.5, 0.1), then flip axis0: (0.1, 0.5, 0.1)
        result = transform_landmarks_canonical(lm, [2, 1, 0], [True, False, False])
        assert abs(result["pt"][0] - 0.1) < 1e-10
        assert abs(result["pt"][1] - 0.5) < 1e-10
        assert abs(result["pt"][2] - 0.1) < 1e-10


class TestEndToEnd:
    """DRR生成のE2Eテスト（回転→DRR→品質チェック）"""

    def test_rotated_drr_valid(self):
        """回転後のボリュームからDRR生成が正常に動作する"""
        vol = np.zeros((64, 64, 64), dtype=np.float32)
        vol[10:55, 28:36, 28:36] = 0.5

        landmarks = {
            "joint_center": (0.5, 0.5, 0.5),
            "humerus_shaft": (0.15, 0.5, 0.5),
            "forearm_shaft": (0.75, 0.5, 0.5),
        }

        rotated_vol, _ = rotate_volume_and_landmarks(
            vol, landmarks, forearm_rotation_deg=10, flexion_deg=120, base_flexion=90
        )
        drr_ap = generate_drr(rotated_vol, axis="AP")
        drr_lat = generate_drr(rotated_vol, axis="LAT")

        assert drr_ap.dtype == np.uint8
        assert drr_lat.dtype == np.uint8
        assert drr_ap.max() > 50  # 骨がある
        assert drr_lat.max() > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
