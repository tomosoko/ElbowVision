"""Tests for scripts/flexion_2d_warp.py pure-function utilities.

Covers functions that operate solely on numpy arrays / geometric math and
do NOT require CT volumes or real X-ray files:
  - KP_ORDER (constant)
  - project_landmarks_2d
  - ssim_gray
  - draw_landmarks
  - tps_warp
"""
import sys
import os

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from flexion_2d_warp import (
    KP_ORDER,
    project_landmarks_2d,
    ssim_gray,
    draw_landmarks,
    tps_warp,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def make_bgr(h=64, w=64, fill=(128, 64, 32), dtype=np.uint8):
    img = np.zeros((h, w, 3), dtype=dtype)
    img[:, :] = fill
    return img


def make_random(h=64, w=64, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=dtype)


def make_landmarks_centered():
    """All normalized landmarks at volume center (0.5, 0.5, 0.5)."""
    return {name: (0.5, 0.5, 0.5) for name in KP_ORDER}


def make_landmarks_varied():
    """Unique but valid normalized landmarks for each keypoint."""
    rng = np.random.default_rng(42)
    return {name: tuple(rng.uniform(0.2, 0.8, 3)) for name in KP_ORDER}


# ── KP_ORDER constant ──────────────────────────────────────────────────────

class TestKpOrder:
    def test_has_six_elements(self):
        assert len(KP_ORDER) == 6

    def test_contains_expected_anatomical_names(self):
        expected = {
            "humerus_shaft",
            "lateral_epicondyle",
            "medial_epicondyle",
            "forearm_shaft",
            "radial_head",
            "olecranon",
        }
        assert set(KP_ORDER) == expected

    def test_is_list(self):
        assert isinstance(KP_ORDER, list)

    def test_all_elements_are_strings(self):
        assert all(isinstance(k, str) for k in KP_ORDER)


# ── project_landmarks_2d ───────────────────────────────────────────────────

class TestProjectLandmarks2d:
    """Test pure perspective-projection function."""

    # vol_shape = (NP, NA, NM)
    VOL = (100, 80, 60)   # (Proximal-Distal, Ant-Post, Med-Lat)
    SID = 1000.0
    VMM = 1.0

    def _pts_hw(self, axis="LAT", lm=None, vol=None, sid=None, vmm=None):
        if lm is None:
            lm = make_landmarks_centered()
        if vol is None:
            vol = self.VOL
        if sid is None:
            sid = self.SID
        if vmm is None:
            vmm = self.VMM
        return project_landmarks_2d(lm, vol, sid, vmm, axis=axis)

    # return structure
    def test_returns_tuple_of_two(self):
        result = self._pts_hw()
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_dict(self):
        pts, hw = self._pts_hw()
        assert isinstance(pts, dict)

    def test_second_element_is_tuple_of_two(self):
        pts, hw = self._pts_hw()
        assert isinstance(hw, tuple) and len(hw) == 2

    def test_dict_has_all_kp_keys(self):
        pts, _ = self._pts_hw()
        assert set(pts.keys()) == set(KP_ORDER)

    # LAT axis dimensions
    def test_lat_height_equals_np(self):
        NP, NA, NM = self.VOL
        _, (H, W) = self._pts_hw(axis="LAT")
        assert H == NP

    def test_lat_width_equals_na(self):
        NP, NA, NM = self.VOL
        _, (H, W) = self._pts_hw(axis="LAT")
        assert W == NA

    # AP axis dimensions
    def test_ap_height_equals_np(self):
        NP, NA, NM = self.VOL
        _, (H, W) = self._pts_hw(axis="AP")
        assert H == NP

    def test_ap_width_equals_nm(self):
        NP, NA, NM = self.VOL
        _, (H, W) = self._pts_hw(axis="AP")
        assert W == NM

    # Pixel coordinate validity
    def test_all_pixel_x_coords_are_finite(self):
        pts, _ = self._pts_hw(lm=make_landmarks_varied())
        for name, (x, y) in pts.items():
            assert np.isfinite(x), f"{name}: x={x} is not finite"

    def test_all_pixel_y_coords_are_finite(self):
        pts, _ = self._pts_hw(lm=make_landmarks_varied())
        for name, (x, y) in pts.items():
            assert np.isfinite(y), f"{name}: y={y} is not finite"

    # Centering property
    def test_centered_landmark_projects_near_image_center(self):
        pts, (H, W) = self._pts_hw(lm=make_landmarks_centered())
        for name, (x, y) in pts.items():
            assert abs(x - W / 2) < W * 0.5, f"{name}: x={x} far from center W/2={W/2}"
            assert abs(y - H / 2) < H * 0.5, f"{name}: y={y} far from center H/2={H/2}"

    # AP vs LAT give different results
    def test_ap_and_lat_give_different_image_sizes(self):
        _, (H_lat, W_lat) = self._pts_hw(axis="LAT")
        _, (H_ap, W_ap) = self._pts_hw(axis="AP")
        NP, NA, NM = self.VOL
        # LAT: W=NA=80, AP: W=NM=60
        assert W_lat != W_ap

    # SID effect
    def test_different_sid_changes_projection(self):
        lm = make_landmarks_varied()
        pts_near, _ = self._pts_hw(lm=lm, sid=500.0)
        pts_far, _ = self._pts_hw(lm=lm, sid=5000.0)
        # Coordinates should differ due to perspective difference
        diffs = [
            abs(pts_near[k][0] - pts_far[k][0]) + abs(pts_near[k][1] - pts_far[k][1])
            for k in KP_ORDER
        ]
        assert any(d > 0.01 for d in diffs)

    # Default axis is LAT
    def test_default_axis_is_lat(self):
        lm = make_landmarks_centered()
        vol = self.VOL
        pts_default, hw_default = project_landmarks_2d(lm, vol, self.SID, self.VMM)
        pts_lat, hw_lat = project_landmarks_2d(lm, vol, self.SID, self.VMM, axis="LAT")
        assert hw_default == hw_lat
        for k in KP_ORDER:
            assert abs(pts_default[k][0] - pts_lat[k][0]) < 1e-9
            assert abs(pts_default[k][1] - pts_lat[k][1]) < 1e-9


# ── ssim_gray ─────────────────────────────────────────────────────────────

class TestSsimGray:
    def test_returns_float(self):
        img = make_gray()
        result = ssim_gray(img, img)
        assert isinstance(result, float)

    def test_identical_images_return_one(self):
        img = make_random(64, 64, seed=1)
        result = ssim_gray(img, img)
        assert abs(result - 1.0) < 1e-6

    def test_different_images_return_less_than_one(self):
        a = make_random(64, 64, seed=1)
        b = make_random(64, 64, seed=2)
        result = ssim_gray(a, b)
        assert result < 1.0

    def test_inverted_image_has_low_score(self):
        a = make_random(64, 64, seed=3)
        b = (255 - a.astype(np.int32)).clip(0, 255).astype(np.uint8)
        result = ssim_gray(a, b)
        assert result < 0.5

    def test_slightly_noisy_image_has_high_score(self):
        base = make_random(64, 64, seed=4)
        noisy = base.copy()
        rng = np.random.default_rng(99)
        noise = rng.integers(-5, 5, base.shape).astype(np.int16)
        noisy = (base.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        result = ssim_gray(base, noisy)
        assert result > 0.9

    def test_uniform_vs_random_has_very_low_score(self):
        a = make_gray(fill=128)
        b = make_random(64, 64, seed=5)
        result = ssim_gray(a, b)
        assert result < 0.5

    def test_result_is_symmetric(self):
        a = make_random(64, 64, seed=6)
        b = make_random(64, 64, seed=7)
        assert abs(ssim_gray(a, b) - ssim_gray(b, a)) < 1e-9

    def test_result_bounded_between_minus_one_and_one(self):
        a = make_random(64, 64, seed=8)
        b = make_random(64, 64, seed=9)
        result = ssim_gray(a, b)
        assert -1.0 <= result <= 1.0


# ── draw_landmarks ─────────────────────────────────────────────────────────

class TestDrawLandmarks:
    """draw_landmarks(img, pts, color, radius=4, thickness=2) → BGR ndarray"""

    def make_pts(self, h=64, w=64):
        """Evenly spread 6 landmarks across the image."""
        xs = np.linspace(10, w - 10, len(KP_ORDER)).tolist()
        return {name: (xs[i], h / 2) for i, name in enumerate(KP_ORDER)}

    def test_output_has_three_channels(self):
        img = make_gray()
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 255, 0))
        assert out.ndim == 3 and out.shape[2] == 3

    def test_output_hw_matches_input_grayscale(self):
        img = make_gray(h=32, w=48)
        pts = self.make_pts(32, 48)
        out = draw_landmarks(img, pts, color=(255, 0, 0))
        assert out.shape[0] == 32 and out.shape[1] == 48

    def test_output_hw_matches_input_bgr(self):
        img = make_bgr(h=64, w=64)
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 0, 255))
        assert out.shape[:2] == (64, 64)

    def test_grayscale_input_converted_to_bgr(self):
        img = make_gray()
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 255, 0))
        assert out.ndim == 3 and out.shape[2] == 3

    def test_original_grayscale_not_modified(self):
        img = make_gray(fill=100)
        original_copy = img.copy()
        pts = self.make_pts()
        draw_landmarks(img, pts, color=(255, 0, 0))
        np.testing.assert_array_equal(img, original_copy)

    def test_original_bgr_not_modified(self):
        img = make_bgr(fill=(50, 100, 150))
        original_copy = img.copy()
        pts = self.make_pts()
        draw_landmarks(img, pts, color=(0, 0, 255))
        np.testing.assert_array_equal(img, original_copy)

    def test_drawing_changes_some_pixels(self):
        img = make_gray(fill=0)   # all black
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 255, 0))
        # At least some pixels should be non-zero (drawn circles/text)
        assert out.any()

    def test_different_colors_produce_different_results(self):
        img = make_gray(fill=0)
        pts = self.make_pts()
        out_green = draw_landmarks(img, pts, color=(0, 255, 0))
        out_red = draw_landmarks(img, pts, color=(0, 0, 255))
        assert not np.array_equal(out_green, out_red)

    def test_works_with_256x256_image(self):
        img = make_gray(h=256, w=256)
        pts = self.make_pts(256, 256)
        out = draw_landmarks(img, pts, color=(128, 128, 128))
        assert out.shape == (256, 256, 3)

    def test_output_dtype_is_uint8(self):
        img = make_gray()
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 128, 255))
        assert out.dtype == np.uint8

    def test_larger_radius_covers_more_pixels(self):
        img = make_gray(fill=0, h=128, w=128)
        pts = self.make_pts(128, 128)
        out_small = draw_landmarks(img, pts, color=(0, 255, 0), radius=2)
        out_large = draw_landmarks(img, pts, color=(0, 255, 0), radius=12)
        # Larger radius → more non-zero pixels
        assert out_large.sum() >= out_small.sum()

    def test_bgr_input_produces_bgr_output(self):
        img = make_bgr(fill=(200, 100, 50))
        pts = self.make_pts()
        out = draw_landmarks(img, pts, color=(0, 255, 0))
        assert out.ndim == 3 and out.shape[2] == 3


# ── tps_warp ───────────────────────────────────────────────────────────────

class TestTpsWarp:
    """tps_warp(src_img, src_pts, dst_pts) → warped ndarray"""

    def make_simple_pts(self, h=64, w=64, n=4):
        """n control points arranged in a grid."""
        xs = np.linspace(w * 0.2, w * 0.8, n).tolist()
        return [(x, h * 0.5) for x in xs]

    def test_output_shape_matches_input(self):
        img = make_gray(h=64, w=64)
        src = self.make_simple_pts()
        out = tps_warp(img, src, src)  # identity
        assert out.shape == img.shape

    def test_identity_warp_returns_similar_image(self):
        img = make_random(h=64, w=64, seed=10)
        pts = self.make_simple_pts()
        out = tps_warp(img, pts, pts)  # identity mapping
        score = ssim_gray(
            img.astype(np.uint8),
            out.clip(0, 255).astype(np.uint8),
        )
        assert score > 0.95, f"Identity warp SSIM={score:.4f}"

    def test_output_values_in_valid_pixel_range(self):
        img = make_random(h=64, w=64, seed=11)
        pts = self.make_simple_pts()
        out = tps_warp(img, pts, pts)
        # After clipping / remap, values should be representable as uint8
        clipped = out.clip(0, 255).astype(np.uint8)
        assert clipped.min() >= 0 and clipped.max() <= 255

    def test_works_without_crash_on_uniform_image(self):
        img = make_gray(fill=128)
        src = self.make_simple_pts()
        out = tps_warp(img, src, src)
        assert out is not None

    def test_large_displacement_changes_image_significantly(self):
        img = make_random(h=64, w=64, seed=12)
        # Move points far right
        src = [(16, 32), (32, 32), (48, 32), (60, 32)]
        dst = [(4, 32), (12, 32), (20, 32), (28, 32)]
        out = tps_warp(img, src, dst)
        score = ssim_gray(
            img.astype(np.uint8),
            out.clip(0, 255).astype(np.uint8),
        )
        assert score < 0.95, f"Large displacement should change image (SSIM={score:.4f})"

    def test_deterministic_same_inputs_give_same_output(self):
        img = make_random(h=64, w=64, seed=13)
        pts = self.make_simple_pts()
        out1 = tps_warp(img, pts, pts)
        out2 = tps_warp(img, pts, pts)
        np.testing.assert_array_equal(out1, out2)

    def test_works_with_256x256_images(self):
        img = make_gray(h=256, w=256, fill=100)
        pts = self.make_simple_pts(h=256, w=256)
        out = tps_warp(img, pts, pts)
        assert out.shape == (256, 256)

    def test_output_2d_for_grayscale_input(self):
        img = make_gray(h=64, w=64)
        pts = self.make_simple_pts()
        out = tps_warp(img, pts, pts)
        assert out.ndim == 2
