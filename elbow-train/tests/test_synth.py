"""
ElbowSynth ユニットテスト — DRR生成パイプラインの各関数をテスト
"""
import math
import sys
import os

import numpy as np
import pytest

# elbow-train ディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from elbow_synth import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    generate_drr,
    auto_detect_landmarks,
    make_yolo_label,
    _project_kp_perspective,
    rotate_volume_and_landmarks,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def make_synthetic_volume(size: int = 64) -> np.ndarray:
    """テスト用の骨模擬ボリューム (PD, AP, ML) を生成。
    中央に円柱状の高密度領域（骨）を配置し、関節部分を太くする。"""
    vol = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2
    radius_shaft = size // 8
    radius_joint = size // 4

    for pd in range(size):
        # 関節レベル（中央付近）は太く
        dist_from_center = abs(pd - center)
        if dist_from_center < size // 8:
            r = radius_joint
        else:
            r = radius_shaft

        for ap in range(size):
            for ml in range(size):
                d = math.sqrt((ap - center) ** 2 + (ml - center) ** 2)
                if d < r:
                    vol[pd, ap, ml] = 0.7  # 骨密度
    return vol


@pytest.fixture(scope="module")
def synth_volume():
    return make_synthetic_volume(64)


@pytest.fixture(scope="module")
def synth_landmarks(synth_volume):
    return auto_detect_landmarks(synth_volume, laterality='R')


# ─── 回転行列テスト ──────────────────────────────────────────────────────────


class TestRotationMatrices:
    def test_identity_at_0_degrees(self):
        for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R = fn(0.0)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            for deg in [30, 45, 90, -60, 180]:
                R = fn(deg)
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10,
                                           err_msg=f"{fn.__name__}({deg})")

    def test_determinant_is_one(self):
        for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            for deg in [15, 90, 270]:
                R = fn(deg)
                assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_360_returns_identity(self):
        for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R = fn(360.0)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_inverse_rotation(self):
        for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R_pos = fn(45.0)
            R_neg = fn(-45.0)
            np.testing.assert_allclose(R_pos @ R_neg, np.eye(3), atol=1e-10)


# ─── DRR生成テスト ───────────────────────────────────────────────────────────


class TestGenerateDRR:
    def test_ap_output_shape(self, synth_volume):
        drr = generate_drr(synth_volume, axis="AP")
        NP, NA, NM = synth_volume.shape
        assert drr.shape == (NP, NM)

    def test_lat_output_shape(self, synth_volume):
        drr = generate_drr(synth_volume, axis="LAT")
        NP, NA, NM = synth_volume.shape
        assert drr.shape == (NP, NA)

    def test_output_dtype_uint8(self, synth_volume):
        for axis in ["AP", "LAT"]:
            drr = generate_drr(synth_volume, axis=axis)
            assert drr.dtype == np.uint8

    def test_bone_brighter_than_air(self, synth_volume):
        """骨がある中央部は空気（端）より明るいはず"""
        drr = generate_drr(synth_volume, axis="AP")
        h, w = drr.shape
        center_region = drr[h // 4:3 * h // 4, w // 4:3 * w // 4]
        edge_region = drr[:5, :5]  # 左上コーナー（空気）
        assert center_region.mean() > edge_region.mean()

    def test_empty_volume_uniform_output(self):
        """空ボリュームでは均一な出力（骨構造なし）"""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        drr = generate_drr(vol, axis="AP")
        # 骨がないのでCLAHE前のガンマ補正後はほぼ均一
        # CLAHEがコントラスト引き伸ばしするが、標準偏差は骨ありより小さい
        vol_bone = make_synthetic_volume(32)
        drr_bone = generate_drr(vol_bone, axis="AP")
        assert drr.std() <= drr_bone.std() + 10

    def test_sid_parameter_accepted(self, synth_volume):
        """SIDパラメータで結果が変わることを確認"""
        drr_1000 = generate_drr(synth_volume, axis="AP", sid_mm=1000.0)
        drr_500 = generate_drr(synth_volume, axis="AP", sid_mm=500.0)
        # SIDが異なれば拡大率が変わるので画像が異なる
        assert not np.array_equal(drr_1000, drr_500)


# ─── ランドマーク自動検出テスト ──────────────────────────────────────────────


class TestAutoDetectLandmarks:
    REQUIRED_KEYS = [
        "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
        "forearm_shaft", "radial_head", "olecranon", "joint_center",
    ]

    def test_returns_all_landmarks(self, synth_landmarks):
        for key in self.REQUIRED_KEYS:
            assert key in synth_landmarks, f"'{key}' が欠落"

    def test_landmarks_are_normalized(self, synth_landmarks):
        """全座標が0〜1の範囲"""
        for name, (pd, ap, ml) in synth_landmarks.items():
            assert 0.0 <= pd <= 1.0, f"{name} PD={pd}"
            assert 0.0 <= ap <= 1.0, f"{name} AP={ap}"
            assert 0.0 <= ml <= 1.0, f"{name} ML={ml}"

    def test_humerus_proximal_to_forearm(self, synth_landmarks):
        """上腕骨は前腕骨より近位（PD値が小さい）"""
        hum_pd = synth_landmarks["humerus_shaft"][0]
        fore_pd = synth_landmarks["forearm_shaft"][0]
        assert hum_pd < fore_pd

    def test_lateral_epicondyle_more_lateral(self, synth_landmarks):
        """外側上顆は内側上顆よりML値が大きい（正準方向: medial→lateral）"""
        lat_ml = synth_landmarks["lateral_epicondyle"][2]
        med_ml = synth_landmarks["medial_epicondyle"][2]
        assert lat_ml > med_ml

    def test_joint_center_between_epicondyles(self, synth_landmarks):
        """関節中心は外側・内側上顆の中間"""
        lat_ml = synth_landmarks["lateral_epicondyle"][2]
        med_ml = synth_landmarks["medial_epicondyle"][2]
        jc_ml = synth_landmarks["joint_center"][2]
        expected = (lat_ml + med_ml) / 2
        assert abs(jc_ml - expected) < 0.01


# ─── YOLOラベル生成テスト ────────────────────────────────────────────────────


class TestMakeYoloLabel:
    def test_label_format(self, synth_landmarks):
        label = make_yolo_label(synth_landmarks, "AP", 256, 256)
        parts = label.split()
        # class + cx,cy,w,h + 6 keypoints * 3 values = 1 + 4 + 18 = 23
        assert len(parts) == 23
        assert parts[0] == "0"  # class id

    def test_all_values_numeric(self, synth_landmarks):
        for axis in ["AP", "LAT"]:
            label = make_yolo_label(synth_landmarks, axis, 256, 256)
            for val in label.split():
                float(val)  # パース可能であること

    def test_bbox_in_range(self, synth_landmarks):
        label = make_yolo_label(synth_landmarks, "AP", 256, 256)
        parts = label.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < bw <= 1.0
        assert 0.0 < bh <= 1.0

    def test_keypoint_coords_in_range(self, synth_landmarks):
        label = make_yolo_label(synth_landmarks, "AP", 256, 256)
        parts = label.split()
        for i in range(6):
            px = float(parts[5 + i * 3])
            py = float(parts[6 + i * 3])
            vis = int(parts[7 + i * 3])
            assert 0.0 <= px <= 1.0, f"kp{i} px={px}"
            assert 0.0 <= py <= 1.0, f"kp{i} py={py}"
            assert vis in [0, 1, 2], f"kp{i} vis={vis}"

    def test_ap_olecranon_occluded(self, synth_landmarks):
        """AP像ではolecranon(kp5)がoccluded(1)"""
        label = make_yolo_label(synth_landmarks, "AP", 256, 256)
        parts = label.split()
        olecranon_vis = int(parts[5 + 5 * 3 + 2])  # kp5のvisibility
        assert olecranon_vis == 1

    def test_lat_medial_epicondyle_occluded(self, synth_landmarks):
        """LAT像ではmedial_epicondyle(kp2)がoccluded(1)"""
        label = make_yolo_label(synth_landmarks, "LAT", 256, 256)
        parts = label.split()
        med_epi_vis = int(parts[5 + 2 * 3 + 2])  # kp2のvisibility
        assert med_epi_vis == 1


# ─── 透視投影テスト ──────────────────────────────────────────────────────────


class TestPerspectiveProjection:
    def test_center_maps_to_center(self):
        """ボリューム中心のキーポイントは画像中心に投影される"""
        px, py = _project_kp_perspective(0.5, 0.5, 0.5, 128, 872.0, 1000.0)
        assert abs(px - 0.5) < 0.01
        assert abs(py - 0.5) < 0.01

    def test_magnification_increases_offset(self):
        """中心から離れたポイントは拡大率により更に外側に投影される"""
        px, _ = _project_kp_perspective(0.5, 0.0, 0.7, 128, 872.0, 1000.0)
        assert px > 0.7  # 拡大効果

    def test_output_clamped_to_01(self):
        """極端な座標でも0-1にクランプされる"""
        px, py = _project_kp_perspective(0.5, 0.0, 10.0, 128, 872.0, 1000.0)
        assert 0.0 <= px <= 1.0
        assert 0.0 <= py <= 1.0


# ─── ボリューム回転テスト ────────────────────────────────────────────────────


class TestRotateVolumeAndLandmarks:
    def test_zero_rotation_preserves_volume(self, synth_volume, synth_landmarks):
        rot_vol, rot_lm = rotate_volume_and_landmarks(
            synth_volume, synth_landmarks,
            forearm_rotation_deg=0.0, flexion_deg=90.0, base_flexion=90.0
        )
        assert rot_vol.shape == synth_volume.shape
        # ゼロ回転なら元とほぼ同じ
        np.testing.assert_allclose(rot_vol, synth_volume, atol=0.05)

    def test_rotation_changes_volume(self, synth_volume, synth_landmarks):
        rot_vol, _ = rotate_volume_and_landmarks(
            synth_volume, synth_landmarks,
            forearm_rotation_deg=30.0, flexion_deg=90.0, base_flexion=90.0
        )
        # 回転後は元と異なるはず
        assert not np.allclose(rot_vol, synth_volume, atol=0.01)

    def test_landmarks_remain_normalized(self, synth_volume, synth_landmarks):
        _, rot_lm = rotate_volume_and_landmarks(
            synth_volume, synth_landmarks,
            forearm_rotation_deg=15.0, flexion_deg=120.0, base_flexion=90.0
        )
        for name, (pd, ap, ml) in rot_lm.items():
            assert 0.0 <= pd <= 1.0, f"{name} PD={pd}"
            assert 0.0 <= ap <= 1.0, f"{name} AP={ap}"
            assert 0.0 <= ml <= 1.0, f"{name} ML={ml}"

    def test_flexion_changes_forearm_position(self, synth_volume, synth_landmarks):
        """屈曲角を変えると前腕の位置が変わる"""
        _, lm_90 = rotate_volume_and_landmarks(
            synth_volume, synth_landmarks,
            forearm_rotation_deg=0.0, flexion_deg=90.0, base_flexion=90.0
        )
        _, lm_120 = rotate_volume_and_landmarks(
            synth_volume, synth_landmarks,
            forearm_rotation_deg=0.0, flexion_deg=120.0, base_flexion=90.0
        )
        fore_90 = lm_90["forearm_shaft"]
        fore_120 = lm_120["forearm_shaft"]
        # 屈曲角が異なれば前腕位置も異なる
        diff = sum((a - b) ** 2 for a, b in zip(fore_90, fore_120))
        assert diff > 0.001
