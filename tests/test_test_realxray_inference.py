"""Unit tests for scripts/test_realxray_inference.py pure functions.

Tests cover:
  - Constants (KP_NAMES, KP_COLORS_BGR, KP_COLORS_RGB)
  - build_drr_cdf: CDF construction from image directory
  - histogram_match: LUT-based histogram matching
  - preprocess_xray: CLAHE + histogram matching pipeline
  - draw_keypoints: keypoint visualization
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

# ── Mock heavy dependencies before import ────────────────────────────────────

_cv2_real = sys.modules.get("cv2")

_cv2_mock = MagicMock()
_cv2_mock.INTER_AREA = 3
_cv2_mock.COLOR_GRAY2BGR = 8
_cv2_mock.FONT_HERSHEY_SIMPLEX = 0
_cv2_mock.LINE_AA = 16
_cv2_mock.IMREAD_GRAYSCALE = 0

_matplotlib_mock = MagicMock()
_matplotlib_pyplot_mock = MagicMock()
_matplotlib_patches_mock = MagicMock()

# Save and force mock injection for import
_saved_modules = {}
for mod_name, mock_obj in [
    ("cv2", _cv2_mock),
    ("matplotlib", _matplotlib_mock),
    ("matplotlib.pyplot", _matplotlib_pyplot_mock),
    ("matplotlib.patches", _matplotlib_patches_mock),
]:
    _saved_modules[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = mock_obj

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import test_realxray_inference as mod

# Restore all original modules after import so other tests aren't broken
for mod_name, original in _saved_modules.items():
    if original is not None:
        sys.modules[mod_name] = original
    elif mod_name in sys.modules:
        del sys.modules[mod_name]


def _make_calcHist(imgs, ch, mask, sz, rng):
    """Simulate cv2.calcHist for grayscale uint8."""
    hist = np.zeros(256, dtype=np.float32)
    for v in imgs[0].ravel():
        hist[v] += 1
    return hist.reshape(-1, 1)


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_kp_names_length(self):
        self.assertEqual(len(mod.KP_NAMES), 6)

    def test_kp_names_contents(self):
        expected = [
            "humerus_shaft",
            "lateral_epicondyle",
            "medial_epicondyle",
            "forearm_shaft",
            "radial_head",
            "olecranon",
        ]
        self.assertEqual(mod.KP_NAMES, expected)

    def test_kp_colors_bgr_length(self):
        self.assertEqual(len(mod.KP_COLORS_BGR), 6)

    def test_kp_colors_bgr_are_3_tuples(self):
        for c in mod.KP_COLORS_BGR:
            self.assertEqual(len(c), 3)
            for v in c:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 255)

    def test_kp_colors_rgb_length(self):
        self.assertEqual(len(mod.KP_COLORS_RGB), 6)

    def test_kp_colors_rgb_bgr_consistency(self):
        """RGB should be BGR channel-swapped."""
        for (b, g, r), (r2, g2, b2) in zip(mod.KP_COLORS_BGR, mod.KP_COLORS_RGB):
            self.assertEqual((r, g, b), (r2, g2, b2))


# ── histogram_match ──────────────────────────────────────────────────────────


class TestHistogramMatch(unittest.TestCase):
    """Tests for histogram_match()."""

    @patch.object(mod, "cv2")
    def test_identity_when_cdfs_match(self, mock_cv2):
        """If source and reference CDF are the same, output ~ input."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (64, 64), dtype=np.uint8)

        hist = np.zeros(256, dtype=np.float64)
        for v in img.ravel():
            hist[v] += 1
        cdf = hist.cumsum()
        cdf /= cdf[-1] + 1e-8

        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, cdf)
        diff = np.abs(result.astype(int) - img.astype(int))
        self.assertLessEqual(diff.mean(), 1.0)

    @patch.object(mod, "cv2")
    def test_output_shape_matches_input(self, mock_cv2):
        img = np.zeros((32, 32), dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.shape, img.shape)

    @patch.object(mod, "cv2")
    def test_output_dtype_uint8(self, mock_cv2):
        img = np.full((16, 16), 128, dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.dtype, np.uint8)

    @patch.object(mod, "cv2")
    def test_uniform_ref_cdf_constant_input(self, mock_cv2):
        """Constant image matched to any CDF stays single-valued."""
        img = np.full((64, 64), 100, dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(len(np.unique(result)), 1)

    @patch.object(mod, "cv2")
    def test_dark_ref_cdf_shifts_dark(self, mock_cv2):
        """CDF concentrated at low values should darken output."""
        rng = np.random.RandomState(0)
        img = rng.randint(100, 200, (32, 32), dtype=np.uint8)
        ref_cdf = np.ones(256, dtype=np.float64)
        ref_cdf[:50] = np.linspace(0, 1, 50)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertLessEqual(result.max(), 50)

    @patch.object(mod, "cv2")
    def test_bright_ref_cdf_shifts_bright(self, mock_cv2):
        """CDF that stays 0 until end should brighten output."""
        rng = np.random.RandomState(1)
        img = rng.randint(50, 150, (32, 32), dtype=np.uint8)
        ref_cdf = np.zeros(256, dtype=np.float64)
        ref_cdf[200:] = np.linspace(0, 1, 56)
        ref_cdf[-1] = 1.0
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertGreaterEqual(result.min(), 200)


# ── build_drr_cdf ────────────────────────────────────────────────────────────


class TestBuildDrrCdf(unittest.TestCase):
    """Tests for build_drr_cdf()."""

    def test_returns_none_for_empty_dir(self):
        """No images found -> None."""
        with patch("os.path.isdir", return_value=False):
            result = mod.build_drr_cdf("/nonexistent", n_samples=10)
        self.assertIsNone(result)

    @patch.object(mod, "cv2")
    def test_returns_array_of_256(self, mock_cv2):
        fake_img = np.full((64, 64), 128, dtype=np.uint8)

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.return_value = fake_img
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=lambda p: "train" in p), \
             patch("os.listdir", return_value=["img001.png", "img002.png"]):
            result = mod.build_drr_cdf("/fake/dir", n_samples=10)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (256,))

    @patch.object(mod, "cv2")
    def test_cdf_monotonically_nondecreasing(self, mock_cv2):
        fake_img = np.random.RandomState(42).randint(0, 256, (32, 32), dtype=np.uint8)

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.return_value = fake_img
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=lambda p: "train" in p), \
             patch("os.listdir", return_value=["a.png"]):
            cdf = mod.build_drr_cdf("/fake", n_samples=10)

        self.assertTrue(np.all(np.diff(cdf) >= 0))

    @patch.object(mod, "cv2")
    def test_cdf_ends_near_one(self, mock_cv2):
        fake_img = np.full((16, 16), 200, dtype=np.uint8)

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.return_value = fake_img
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=lambda p: "val" in p), \
             patch("os.listdir", return_value=["x.png"]):
            cdf = mod.build_drr_cdf("/fake", n_samples=5)

        self.assertAlmostEqual(cdf[-1], 1.0, places=5)

    @patch.object(mod, "cv2")
    def test_skips_none_images(self, mock_cv2):
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.calcHist.side_effect = _make_calcHist

        def mock_imread(path, flag):
            if "bad" in path:
                return None
            return np.full((8, 8), 100, dtype=np.uint8)

        mock_cv2.imread.side_effect = mock_imread

        with patch("os.path.isdir", side_effect=lambda p: "train" in p), \
             patch("os.listdir", return_value=["good.png", "bad.png"]):
            cdf = mod.build_drr_cdf("/fake", n_samples=10)

        self.assertIsNotNone(cdf)

    @patch.object(mod, "cv2")
    def test_both_train_val_splits(self, mock_cv2):
        checked = set()

        def mock_isdir(path):
            for split in ["train", "val"]:
                if path.endswith(split):
                    checked.add(split)
            return True

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.return_value = np.full((8, 8), 128, dtype=np.uint8)
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=mock_isdir), \
             patch("os.listdir", return_value=["img.png"]):
            mod.build_drr_cdf("/fake", n_samples=5)

        self.assertIn("train", checked)
        self.assertIn("val", checked)

    @patch.object(mod, "cv2")
    def test_n_samples_limits_files(self, mock_cv2):
        files = [f"img{i:04d}.png" for i in range(100)]
        read_paths = []

        def mock_imread(path, flag):
            read_paths.append(path)
            return np.full((8, 8), 50, dtype=np.uint8)

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.side_effect = mock_imread
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=lambda p: "train" in p), \
             patch("os.listdir", return_value=files):
            mod.build_drr_cdf("/fake", n_samples=10)

        self.assertLessEqual(len(read_paths), 15)

    @patch.object(mod, "cv2")
    def test_only_png_files(self, mock_cv2):
        read_files = []

        def mock_imread(path, flag):
            read_files.append(os.path.basename(path))
            return np.full((8, 8), 128, dtype=np.uint8)

        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.imread.side_effect = mock_imread
        mock_cv2.calcHist.side_effect = _make_calcHist

        with patch("os.path.isdir", side_effect=lambda p: "train" in p), \
             patch("os.listdir", return_value=["img.png", "readme.txt", "data.csv", "photo.jpg", "drr.png"]):
            mod.build_drr_cdf("/fake", n_samples=100)

        for f in read_files:
            self.assertTrue(f.endswith(".png"), f"Non-png file read: {f}")


# ── preprocess_xray ──────────────────────────────────────────────────────────


class TestPreprocessXray(unittest.TestCase):
    """Tests for preprocess_xray()."""

    @patch.object(mod, "cv2")
    def test_resize_to_target_size(self, mock_cv2):
        img = np.zeros((512, 512), dtype=np.uint8)
        target = 256
        resized = np.zeros((target, target), dtype=np.uint8)

        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = resized
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = resized
        mock_cv2.createCLAHE.return_value = clahe_mock

        mod.preprocess_xray(img, drr_cdf=None, target_size=target)
        call_args = mock_cv2.resize.call_args
        self.assertEqual(call_args[0][1], (target, target))

    @patch.object(mod, "cv2")
    def test_clahe_applied(self, mock_cv2):
        img = np.zeros((128, 128), dtype=np.uint8)
        resized = np.zeros((256, 256), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = resized
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = resized
        mock_cv2.createCLAHE.return_value = clahe_mock

        mod.preprocess_xray(img, drr_cdf=None)
        mock_cv2.createCLAHE.assert_called_once_with(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_mock.apply.assert_called_once()

    @patch.object(mod, "cv2")
    def test_no_histogram_match_when_cdf_none(self, mock_cv2):
        img = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = img
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = img
        mock_cv2.createCLAHE.return_value = clahe_mock

        mod.preprocess_xray(img, drr_cdf=None)
        mock_cv2.calcHist.assert_not_called()

    @patch.object(mod, "cv2")
    def test_histogram_match_when_cdf_provided(self, mock_cv2):
        img = np.full((64, 64), 128, dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = img
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = img
        mock_cv2.createCLAHE.return_value = clahe_mock
        mock_cv2.calcHist.side_effect = _make_calcHist

        ref_cdf = np.linspace(0, 1, 256)
        mod.preprocess_xray(img, drr_cdf=ref_cdf)
        mock_cv2.calcHist.assert_called()

    @patch.object(mod, "cv2")
    def test_custom_target_size(self, mock_cv2):
        img = np.zeros((100, 100), dtype=np.uint8)
        target = 128
        resized = np.zeros((target, target), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = resized
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = resized
        mock_cv2.createCLAHE.return_value = clahe_mock

        mod.preprocess_xray(img, drr_cdf=None, target_size=target)
        call_args = mock_cv2.resize.call_args
        self.assertEqual(call_args[0][1], (target, target))

    @patch.object(mod, "cv2")
    def test_returns_ndarray(self, mock_cv2):
        img = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = img
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = img
        mock_cv2.createCLAHE.return_value = clahe_mock

        result = mod.preprocess_xray(img, drr_cdf=None)
        self.assertIsInstance(result, np.ndarray)


# ── draw_keypoints ───────────────────────────────────────────────────────────


class TestDrawKeypoints(unittest.TestCase):
    """Tests for draw_keypoints()."""

    def test_none_keypoints_returns_copy(self):
        """When keypoints is None, return a copy of the input."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = mod.draw_keypoints(img, None, None)
        np.testing.assert_array_equal(result, img)
        self.assertIsNot(result, img)

    def test_does_not_modify_input(self):
        """Input image should not be modified."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_copy = img.copy()
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        mod.draw_keypoints(img, kps, confs)
        np.testing.assert_array_equal(img, img_copy)

    @patch.object(mod, "cv2")
    def test_circle_called_per_keypoint(self, mock_cv2):
        """cv2.circle should be called twice per keypoint (filled + outline)."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)
        self.assertEqual(mock_cv2.circle.call_count, 12)

    @patch.object(mod, "cv2")
    def test_puttext_called_per_keypoint(self, mock_cv2):
        """cv2.putText should be called once per keypoint."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)
        self.assertEqual(mock_cv2.putText.call_count, 6)

    @patch.object(mod, "cv2")
    def test_label_format_contains_confidence(self, mock_cv2):
        """Labels should include truncated name and confidence value."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.array([0.95, 0.80, 0.70, 0.60, 0.50, 0.40])

        mod.draw_keypoints(img, kps, confs)

        first_call = mock_cv2.putText.call_args_list[0]
        label_text = first_call[0][1]
        self.assertIn("0.95", label_text)
        self.assertIn("humeru", label_text)

    @patch.object(mod, "cv2")
    def test_circle_positions_match_keypoints(self, mock_cv2):
        """Circle centers should match keypoint coordinates."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        kps = np.array([[15, 25], [35, 45], [55, 65], [75, 85], [95, 105], [115, 125]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)

        # cv2.circle(vis, (x, y), radius, color, thickness)
        first_circle = mock_cv2.circle.call_args_list[0]
        center = first_circle[0][1]
        self.assertEqual(center, (15, 25))

    @patch.object(mod, "cv2")
    def test_colors_match_kp_colors_bgr(self, mock_cv2):
        """Each keypoint should use its assigned BGR color."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)

        # cv2.circle(vis, (x, y), radius, color, thickness)
        for i in range(6):
            filled_call = mock_cv2.circle.call_args_list[i * 2]
            color = filled_call[0][3]
            self.assertEqual(color, mod.KP_COLORS_BGR[i])

    def test_output_shape_matches_input(self):
        """Output should have same shape as input."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        result = mod.draw_keypoints(img, kps, confs)
        self.assertEqual(result.shape, img.shape)

    @patch.object(mod, "cv2")
    def test_float_keypoints_converted_to_int(self, mock_cv2):
        """Float keypoint coordinates should be converted to int for drawing."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        kps = np.array([[10.7, 20.3], [30.5, 40.9], [50.1, 60.6],
                        [70.2, 80.4], [90.8, 100.1], [110.9, 120.5]], dtype=np.float64)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)

        first_circle = mock_cv2.circle.call_args_list[0]
        center = first_circle[0][1]
        self.assertIsInstance(center[0], int)
        self.assertIsInstance(center[1], int)

    @patch.object(mod, "cv2")
    def test_outline_circle_is_white(self, mock_cv2):
        """Outline circles should be white (255, 255, 255)."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)

        # cv2.circle(vis, (x, y), radius, color, thickness)
        for i in range(6):
            outline_call = mock_cv2.circle.call_args_list[i * 2 + 1]
            color = outline_call[0][3]
            self.assertEqual(color, (255, 255, 255))


# ── histogram_match edge cases ───────────────────────────────────────────────


class TestHistogramMatchEdgeCases(unittest.TestCase):
    """Edge case tests for histogram_match."""

    @patch.object(mod, "cv2")
    def test_all_black_image(self, mock_cv2):
        img = np.zeros((32, 32), dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.shape, (32, 32))
        self.assertEqual(len(np.unique(result)), 1)

    @patch.object(mod, "cv2")
    def test_all_white_image(self, mock_cv2):
        img = np.full((32, 32), 255, dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.shape, (32, 32))

    @patch.object(mod, "cv2")
    def test_single_pixel(self, mock_cv2):
        img = np.array([[128]], dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.shape, (1, 1))

    @patch.object(mod, "cv2")
    def test_step_function_cdf(self, mock_cv2):
        """Step CDF at midpoint should collapse values to few unique levels."""
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, (32, 32), dtype=np.uint8)
        ref_cdf = np.zeros(256, dtype=np.float64)
        ref_cdf[128:] = 1.0
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        # With step CDF, pixels map to either ~0 or ~128
        self.assertLessEqual(len(np.unique(result)), 3)

    @patch.object(mod, "cv2")
    def test_rectangular_image(self, mock_cv2):
        img = np.zeros((16, 64), dtype=np.uint8)
        ref_cdf = np.linspace(0, 1, 256)
        mock_cv2.calcHist.side_effect = _make_calcHist
        result = mod.histogram_match(img, ref_cdf)
        self.assertEqual(result.shape, (16, 64))


# ── Integration-style tests ──────────────────────────────────────────────────


class TestPreprocessXrayIntegration(unittest.TestCase):
    """Integration tests that chain preprocess_xray with real computations."""

    @patch.object(mod, "cv2")
    def test_full_pipeline_with_cdf(self, mock_cv2):
        """Full preprocess pipeline: resize + CLAHE + histogram match."""
        img = np.random.RandomState(42).randint(0, 256, (512, 512), dtype=np.uint8)
        target = 256

        resized = np.random.RandomState(43).randint(0, 256, (target, target), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = resized
        clahe_mock = MagicMock()
        clahe_result = np.random.RandomState(44).randint(0, 256, (target, target), dtype=np.uint8)
        clahe_mock.apply.return_value = clahe_result
        mock_cv2.createCLAHE.return_value = clahe_mock
        mock_cv2.calcHist.side_effect = _make_calcHist

        ref_cdf = np.linspace(0, 1, 256)
        result = mod.preprocess_xray(img, drr_cdf=ref_cdf, target_size=target)

        self.assertEqual(result.shape, (target, target))
        self.assertEqual(result.dtype, np.uint8)

    @patch.object(mod, "cv2")
    def test_pipeline_without_cdf(self, mock_cv2):
        """Pipeline without histogram matching should still return valid image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        target = 64
        resized = np.zeros((target, target), dtype=np.uint8)
        mock_cv2.INTER_AREA = 3
        mock_cv2.resize.return_value = resized
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = resized
        mock_cv2.createCLAHE.return_value = clahe_mock

        result = mod.preprocess_xray(img, drr_cdf=None, target_size=target)
        self.assertIsInstance(result, np.ndarray)


class TestDrawKeypointsEdgeCases(unittest.TestCase):
    """Edge cases for draw_keypoints."""

    @patch.object(mod, "cv2")
    def test_zero_confidence_still_draws(self, mock_cv2):
        """Zero confidence keypoints should still be drawn."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.zeros(6)

        mod.draw_keypoints(img, kps, confs)
        self.assertEqual(mock_cv2.circle.call_count, 12)

    def test_keypoints_at_image_boundary(self):
        """Keypoints at (0,0) should still work."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[0, 0], [63, 63], [0, 63], [63, 0], [32, 32], [1, 1]], dtype=np.float32)
        confs = np.ones(6)

        result = mod.draw_keypoints(img, kps, confs)
        self.assertEqual(result.shape, (64, 64, 3))

    @patch.object(mod, "cv2")
    def test_large_keypoint_values(self, mock_cv2):
        """Keypoints outside image bounds should still work."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.array([[999, 999], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], dtype=np.float32)
        confs = np.ones(6)

        mod.draw_keypoints(img, kps, confs)
        self.assertEqual(mock_cv2.circle.call_count, 12)


if __name__ == "__main__":
    unittest.main()
