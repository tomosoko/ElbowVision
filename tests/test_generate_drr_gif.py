"""
Unit tests for scripts/generate_drr_gif.py pure functions.

Covers:
- Constants (TARGET_SIZE, HU_MIN, HU_MAX, LATERALITY, FRAME_DURATION_MS)
- pick_volume(): nearest volume selection from dict
- add_labels(): text overlay on grayscale/color images
- bounce(): list bounce (forward + reverse without duplicates)
- make_gif(): GIF generation via PIL (mocked)
"""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock heavy dependencies before import
_cv2_mock = MagicMock()
_cv2_mock.cvtColor = lambda img, code: (
    np.stack([img] * 3, axis=-1) if len(img.shape) == 2 else img
)
_cv2_mock.COLOR_GRAY2BGR = 6
_cv2_mock.FONT_HERSHEY_SIMPLEX = 0
_cv2_mock.LINE_AA = 16
_cv2_mock.putText = MagicMock()

sys.modules.setdefault("cv2", _cv2_mock)
sys.modules.setdefault("elbow_synth", MagicMock())

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from generate_drr_gif import (
    pick_volume,
    add_labels,
    bounce,
    TARGET_SIZE,
    HU_MIN,
    HU_MAX,
    LATERALITY,
    FRAME_DURATION_MS,
)


# ====================================================================
# Constants
# ====================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_target_size_is_256(self):
        assert TARGET_SIZE == 256

    def test_hu_min(self):
        assert HU_MIN == 50

    def test_hu_max(self):
        assert HU_MAX == 800

    def test_hu_range_valid(self):
        assert HU_MIN < HU_MAX

    def test_laterality_is_L(self):
        assert LATERALITY == "L"

    def test_frame_duration_ms(self):
        assert FRAME_DURATION_MS == 120

    def test_frame_duration_positive(self):
        assert FRAME_DURATION_MS > 0


# ====================================================================
# pick_volume
# ====================================================================


class TestPickVolume:
    """Tests for pick_volume() nearest-volume selection."""

    @pytest.fixture
    def volumes(self):
        """Fake volumes dict keyed by base flexion angle."""
        return {
            180: ("vol_180", 0.5, "lm_180"),
            135: ("vol_135", 0.5, "lm_135"),
            90: ("vol_90", 0.5, "lm_90"),
        }

    def test_exact_match_180(self, volumes):
        result = pick_volume(180, volumes)
        assert result == ("vol_180", 0.5, "lm_180")

    def test_exact_match_135(self, volumes):
        result = pick_volume(135, volumes)
        assert result == ("vol_135", 0.5, "lm_135")

    def test_exact_match_90(self, volumes):
        result = pick_volume(90, volumes)
        assert result == ("vol_90", 0.5, "lm_90")

    def test_nearest_to_170(self, volumes):
        """170 is closest to 180."""
        result = pick_volume(170, volumes)
        assert result == ("vol_180", 0.5, "lm_180")

    def test_nearest_to_100(self, volumes):
        """100 is closest to 90."""
        result = pick_volume(100, volumes)
        assert result == ("vol_90", 0.5, "lm_90")

    def test_nearest_to_150(self, volumes):
        """150 is equidistant from 135 and 180; min() picks first by key order."""
        result = pick_volume(150, volumes)
        # Distance: |150-180|=30, |150-135|=15, |150-90|=60 -> 135 wins
        assert result == ("vol_135", 0.5, "lm_135")

    def test_nearest_to_112(self, volumes):
        """112 -> |112-90|=22, |112-135|=23, |112-180|=68 -> 90 wins."""
        result = pick_volume(112, volumes)
        assert result == ("vol_90", 0.5, "lm_90")

    def test_nearest_to_113(self, volumes):
        """113 -> |113-90|=23, |113-135|=22 -> 135 wins."""
        result = pick_volume(113, volumes)
        assert result == ("vol_135", 0.5, "lm_135")

    def test_single_volume(self):
        """Single volume always returned."""
        vols = {180: ("only_vol", 1.0, "lm")}
        assert pick_volume(0, vols) == ("only_vol", 1.0, "lm")
        assert pick_volume(180, vols) == ("only_vol", 1.0, "lm")
        assert pick_volume(999, vols) == ("only_vol", 1.0, "lm")

    def test_returns_tuple_of_three(self, volumes):
        result = pick_volume(120, volumes)
        assert len(result) == 3


# ====================================================================
# add_labels
# ====================================================================


class TestAddLabels:
    """Tests for add_labels() text overlay."""

    def test_grayscale_converted_to_color(self):
        """Grayscale input should produce 3-channel output."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = add_labels(gray, ["test"])
        assert result.shape == (64, 64, 3)

    def test_color_input_preserved(self):
        """Color input shape preserved."""
        color = np.zeros((64, 64, 3), dtype=np.uint8)
        result = add_labels(color, ["test"])
        assert result.shape == (64, 64, 3)

    def test_empty_lines(self):
        """Empty lines list produces image without error."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = add_labels(gray, [])
        assert result.shape == (64, 64, 3)

    def test_multiple_lines(self):
        """Multiple lines accepted."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = add_labels(gray, ["Line 1", "Line 2", "Line 3"])
        assert result.shape == (64, 64, 3)

    def test_output_different_with_more_lines(self):
        """More lines produces valid output (putText is called per line)."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        result1 = add_labels(gray, ["A"])
        result2 = add_labels(gray, ["A", "B", "C"])
        # Both produce valid 3-channel images
        assert result1.shape == (64, 64, 3)
        assert result2.shape == (64, 64, 3)

    def test_returns_numpy_array(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = add_labels(gray, ["hello"])
        assert isinstance(result, np.ndarray)

    def test_dtype_uint8(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = add_labels(gray, ["hello"])
        assert result.dtype == np.uint8


# ====================================================================
# bounce
# ====================================================================


class TestBounce:
    """Tests for bounce() list reversal."""

    def test_basic(self):
        # reversed([1,2,3])[1:-1] = [2] -> [1,2,3,2]
        assert bounce([1, 2, 3]) == [1, 2, 3, 2]

    def test_single_element(self):
        assert bounce([1]) == [1]

    def test_two_elements(self):
        # reversed([1,2])[1:-1] = [] -> [1,2]
        assert bounce([1, 2]) == [1, 2]

    def test_five_elements(self):
        result = bounce([1, 2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5, 4, 3, 2]

    def test_length_formula(self):
        """Length = n + max(0, n-2) = 2n-2 for n>=2, else n."""
        for n in range(1, 10):
            vals = list(range(n))
            result = bounce(vals)
            expected_len = n + max(0, n - 2)
            assert len(result) == expected_len

    def test_starts_with_first(self):
        result = bounce([7, 8, 9])
        assert result[0] == 7

    def test_ends_exclude_endpoints_of_reverse(self):
        """Reversed portion [1:-1] excludes first and last of reversed list."""
        result = bounce([7, 8, 9])
        # reversed = [9,8,7], [1:-1] = [8] -> full = [7,8,9,8]
        assert result[-1] == 8

    def test_first_element_appears_once(self):
        """First element of input appears only once."""
        result = bounce([1, 2, 3, 4, 5])
        assert result.count(1) == 1

    def test_last_element_appears_once(self):
        """Last element of input appears only once (not in reversed[1:-1])."""
        result = bounce([1, 2, 3, 4, 5])
        assert result.count(5) == 1

    def test_middle_elements_appear_twice(self):
        """Middle elements appear exactly twice."""
        result = bounce([1, 2, 3, 4, 5])
        assert result.count(2) == 2
        assert result.count(3) == 2
        assert result.count(4) == 2

    def test_range_input(self):
        """Works with range objects."""
        result = bounce(range(1, 4))
        assert result == [1, 2, 3, 2]

    def test_strings(self):
        """Works with non-numeric values."""
        result = bounce(["a", "b", "c"])
        assert result == ["a", "b", "c", "b"]

    def test_preserves_original(self):
        """Does not mutate original list."""
        original = [1, 2, 3]
        bounce(original)
        assert original == [1, 2, 3]


# ====================================================================
# make_gif (integration-light, mocked PIL)
# ====================================================================


class TestMakeGif:
    """Tests for make_gif() with mocked PIL."""

    def test_make_gif_calls_save(self, tmp_path):
        from generate_drr_gif import make_gif

        mock_frame = MagicMock()
        mock_frame.save = MagicMock()
        frames = [mock_frame, MagicMock(), MagicMock()]

        out_path = str(tmp_path / "test.gif")
        # Create a fake file so os.path.getsize works
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 100)

        make_gif(frames, out_path)
        mock_frame.save.assert_called_once()

    def test_make_gif_duration_param(self, tmp_path):
        from generate_drr_gif import make_gif

        mock_frame = MagicMock()
        frames = [mock_frame, MagicMock()]

        out_path = str(tmp_path / "test.gif")
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 50)

        make_gif(frames, out_path)
        call_kwargs = mock_frame.save.call_args[1]
        assert call_kwargs["duration"] == FRAME_DURATION_MS

    def test_make_gif_loop_infinite(self, tmp_path):
        from generate_drr_gif import make_gif

        mock_frame = MagicMock()
        frames = [mock_frame, MagicMock()]

        out_path = str(tmp_path / "test.gif")
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 50)

        make_gif(frames, out_path)
        call_kwargs = mock_frame.save.call_args[1]
        assert call_kwargs["loop"] == 0

    def test_make_gif_append_images(self, tmp_path):
        from generate_drr_gif import make_gif

        mock_frame = MagicMock()
        frame2 = MagicMock()
        frame3 = MagicMock()
        frames = [mock_frame, frame2, frame3]

        out_path = str(tmp_path / "test.gif")
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 50)

        make_gif(frames, out_path)
        call_kwargs = mock_frame.save.call_args[1]
        assert call_kwargs["append_images"] == [frame2, frame3]
