"""elbow_synth.py の make_yolo_label / _project_kp_perspective テスト."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))
from elbow_synth import make_yolo_label, _project_kp_perspective


def _center_landmarks():
    """全ランドマークがボリューム中心 (0.5, 0.5, 0.5) にある辞書."""
    kp_names = [
        "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
        "forearm_shaft", "radial_head", "olecranon",
    ]
    return {name: (0.5, 0.5, 0.5) for name in kp_names}


def _spread_landmarks():
    """ランドマークが縦に広がった辞書 (PD方向に分散)."""
    return {
        "humerus_shaft":      (0.1, 0.5, 0.5),
        "lateral_epicondyle": (0.45, 0.5, 0.45),
        "medial_epicondyle":  (0.45, 0.5, 0.55),
        "forearm_shaft":      (0.9, 0.5, 0.5),
        "radial_head":        (0.55, 0.5, 0.45),
        "olecranon":          (0.55, 0.5, 0.55),
    }


class TestProjectKpPerspective:
    """_project_kp_perspective の単体テスト."""

    def test_returns_tuple_of_two_floats(self):
        px, py = _project_kp_perspective(0.5, 0.5, 0.5, 128, 872.0, 1000.0)
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_center_point_stays_at_center(self):
        """中心点 (0.5, 0.5, 0.5) は (0.5, 0.5) に投影される."""
        px, py = _project_kp_perspective(0.5, 0.5, 0.5, 128, 872.0, 1000.0)
        assert abs(px - 0.5) < 1e-6
        assert abs(py - 0.5) < 1e-6

    def test_output_clipped_to_0_1(self):
        """出力は [0, 1] にクリップされる."""
        # 極端な座標
        px, py = _project_kp_perspective(0.0, 0.0, 0.0, 128, 872.0, 1000.0)
        assert 0.0 <= px <= 1.0
        assert 0.0 <= py <= 1.0

    def test_higher_n_pd_gives_higher_py(self):
        """n_PD が大きいほど py が大きい（上から下へ）."""
        _, py1 = _project_kp_perspective(0.3, 0.5, 0.5, 128, 872.0, 1000.0)
        _, py2 = _project_kp_perspective(0.7, 0.5, 0.5, 128, 872.0, 1000.0)
        assert py2 > py1

    def test_higher_n_lateral_gives_higher_px(self):
        """n_lateral が大きいほど px が大きい."""
        px1, _ = _project_kp_perspective(0.5, 0.5, 0.3, 128, 872.0, 1000.0)
        px2, _ = _project_kp_perspective(0.5, 0.5, 0.7, 128, 872.0, 1000.0)
        assert px2 > px1


class TestMakeYoloLabel:
    """make_yolo_label の単体テスト."""

    def test_returns_string(self):
        """戻り値は文字列."""
        lm = _center_landmarks()
        result = make_yolo_label(lm, "AP", 512, 512)
        assert isinstance(result, str)

    def test_starts_with_class_zero(self):
        """ラベルは class 0 で始まる."""
        result = make_yolo_label(_center_landmarks(), "AP", 512, 512)
        assert result.startswith("0 ")

    def test_field_count(self):
        """フィールド数: 1 (class) + 4 (bbox) + 6×3 (kp) = 23."""
        result = make_yolo_label(_spread_landmarks(), "AP", 512, 512)
        fields = result.split()
        assert len(fields) == 23

    def test_bbox_values_in_range(self):
        """bbox の cx,cy,bw,bh はすべて 0〜1 の範囲."""
        result = make_yolo_label(_spread_landmarks(), "AP", 512, 512)
        fields = result.split()
        cx, cy, bw, bh = float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])
        for v in (cx, cy, bw, bh):
            assert 0.0 <= v <= 1.0

    def test_kp_visibility_ap_olecranon_occluded(self):
        """AP像: olecranon (index 5) の visibility = 1 (occluded)."""
        result = make_yolo_label(_spread_landmarks(), "AP", 512, 512)
        fields = result.split()
        # kp_i の visibility は fields[5 + i*3 + 2]
        olecranon_vis = int(fields[5 + 5 * 3 + 2])  # index 5
        assert olecranon_vis == 1

    def test_kp_visibility_ap_others_visible(self):
        """AP像: olecranon 以外は visibility = 2 (visible)."""
        result = make_yolo_label(_spread_landmarks(), "AP", 512, 512)
        fields = result.split()
        for i in range(5):  # index 0〜4
            vis = int(fields[5 + i * 3 + 2])
            assert vis == 2, f"kp index {i} should be visible (2), got {vis}"

    def test_kp_visibility_lat_medial_epicondyle_occluded(self):
        """LAT像: medial_epicondyle (index 2) の visibility = 1."""
        result = make_yolo_label(_spread_landmarks(), "LAT", 512, 512)
        fields = result.split()
        med_vis = int(fields[5 + 2 * 3 + 2])
        assert med_vis == 1

    def test_kp_visibility_lat_radial_head_occluded(self):
        """LAT像: radial_head (index 4) の visibility = 1."""
        result = make_yolo_label(_spread_landmarks(), "LAT", 512, 512)
        fields = result.split()
        rh_vis = int(fields[5 + 4 * 3 + 2])
        assert rh_vis == 1

    def test_kp_coords_in_range(self):
        """全キーポイントの x,y 座標は 0〜1."""
        result = make_yolo_label(_spread_landmarks(), "AP", 512, 512)
        fields = result.split()
        for i in range(6):
            px = float(fields[5 + i * 3])
            py = float(fields[5 + i * 3 + 1])
            assert 0.0 <= px <= 1.0, f"kp {i} px={px} out of range"
            assert 0.0 <= py <= 1.0, f"kp {i} py={py} out of range"

    def test_view_type_overrides_visibility(self):
        """view_type が axis と異なる場合、view_type で visibility を決める."""
        # axis=AP だが view_type=LAT → LAT のルールで visibility 決定
        result_lat = make_yolo_label(_spread_landmarks(), "AP", 512, 512, view_type="LAT")
        fields = result_lat.split()
        med_vis = int(fields[5 + 2 * 3 + 2])  # medial_epicondyle
        assert med_vis == 1
