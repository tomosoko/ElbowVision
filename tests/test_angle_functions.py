"""elbow_synth.py の角度計算関数テスト (compute_carrying_angle / compute_flexion_angle)."""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-train'))
from elbow_synth import compute_carrying_angle, compute_flexion_angle


def _landmarks(hs, jc, fs):
    """テスト用ランドマーク辞書を作る。各値は (PD, AP, ML) のタプル。"""
    return {
        "humerus_shaft":  hs,
        "joint_center":   jc,
        "forearm_shaft":  fs,
    }


class TestComputeCarryingAngle:
    """compute_carrying_angle — AP像のcarrying angle（冠状面）テスト."""

    def test_returns_float(self):
        """戻り値は float."""
        lm = _landmarks((0.2, 0.5, 0.5), (0.5, 0.5, 0.5), (0.8, 0.5, 0.5))
        result = compute_carrying_angle(lm)
        assert isinstance(result, float)

    def test_straight_arm_returns_zero(self):
        """上腕・前腕が同一PD軸上 (ML変化なし) → 0°."""
        lm = _landmarks((0.2, 0.5, 0.5), (0.5, 0.5, 0.5), (0.8, 0.5, 0.5))
        angle = compute_carrying_angle(lm)
        assert abs(angle) < 1e-6

    def test_valgus_positive(self):
        """外反(valgus): 前腕が外側へ → 正の角度."""
        # 上腕: 下向き(PD↑), 前腕: 下向き+外側(ML増加) → valgus正
        lm = _landmarks(
            (0.1, 0.5, 0.5),   # humerus_shaft
            (0.5, 0.5, 0.5),   # joint_center
            (0.9, 0.5, 0.6),   # forearm_shaft: ML+0.1 → valgus
        )
        angle = compute_carrying_angle(lm)
        assert angle > 0

    def test_varus_negative(self):
        """内反(varus): 前腕が内側へ → 負の角度."""
        lm = _landmarks(
            (0.1, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.9, 0.5, 0.4),   # ML-0.1 → varus
        )
        angle = compute_carrying_angle(lm)
        assert angle < 0

    def test_output_in_degrees(self):
        """出力は度単位 (-180〜180)."""
        lm = _landmarks((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.6))
        angle = compute_carrying_angle(lm)
        assert -180.0 <= angle <= 180.0

    def test_antisymmetric_valgus_varus(self):
        """valgusとvarusは反対符号で同じ絶対値."""
        lm_val = _landmarks((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.6))
        lm_var = _landmarks((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.4))
        assert abs(compute_carrying_angle(lm_val) + compute_carrying_angle(lm_var)) < 1e-9

    def test_90_degree_valgus(self):
        """前腕が真横 (PD変化なし, ML増加) → +90°."""
        lm = _landmarks(
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 1.0),   # 前腕が真横
        )
        angle = compute_carrying_angle(lm)
        assert abs(angle - 90.0) < 1e-6


class TestComputeFlexionAngle:
    """compute_flexion_angle — LAT像の屈曲角（矢状面）テスト."""

    def test_returns_float(self):
        """戻り値は float."""
        lm = _landmarks((0.2, 0.5, 0.5), (0.5, 0.5, 0.5), (0.8, 0.5, 0.5))
        result = compute_flexion_angle(lm)
        assert isinstance(result, float)

    def test_full_extension_zero_degrees(self):
        """完全伸展: 上腕・前腕が同一直線 → 0°（ベクトル間夾角=0）."""
        lm = _landmarks((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.5))
        angle = compute_flexion_angle(lm)
        assert abs(angle - 0.0) < 1e-6

    def test_right_angle_flexion_90_degrees(self):
        """90°屈曲: 前腕が上腕に対して垂直 → 90°."""
        lm = _landmarks(
            (0.1, 0.5, 0.5),   # 上腕: 上→下
            (0.5, 0.5, 0.5),
            (0.5, 0.9, 0.5),   # 前腕: 下→奥方向(AP) → 90°
        )
        angle = compute_flexion_angle(lm)
        assert abs(angle - 90.0) < 1e-6

    def test_zero_length_humerus_returns_180(self):
        """上腕ベクトルが0 (hs==jc) → フォールバックで180°."""
        lm = _landmarks(
            (0.5, 0.5, 0.5),   # hs == jc
            (0.5, 0.5, 0.5),
            (0.9, 0.5, 0.5),
        )
        angle = compute_flexion_angle(lm)
        assert abs(angle - 180.0) < 1e-6

    def test_output_in_degrees(self):
        """出力は度単位 (0〜180)."""
        lm = _landmarks((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.6))
        angle = compute_flexion_angle(lm)
        assert 0.0 <= angle <= 180.0

    def test_symmetric_flexion(self):
        """前腕が上腕の鏡像 → 同じ屈曲角."""
        lm1 = _landmarks(
            (0.1, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.5, 0.9, 0.5),  # 前腕: AP方向+
        )
        lm2 = _landmarks(
            (0.1, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.5, 0.1, 0.5),  # 前腕: AP方向- (鏡像)
        )
        assert abs(compute_flexion_angle(lm1) - compute_flexion_angle(lm2)) < 1e-9
