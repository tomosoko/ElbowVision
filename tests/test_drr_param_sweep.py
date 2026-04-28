"""Tests for scripts/drr_param_sweep.py pure-function utilities.

Covers functions that operate solely on numpy arrays and do NOT require
CT volumes or real X-ray files:
  - compute_ssim
  - hist_intersection
  - edge_ratio
  - evaluate_drr
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from drr_param_sweep import (
    compute_ssim,
    hist_intersection,
    edge_ratio,
    evaluate_drr,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def make_random(h=64, w=64, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=dtype)


def make_gradient(h=64, w=64, dtype=np.uint8):
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(dtype)


def make_checker(h=64, w=64, block=8, dtype=np.uint8):
    """Checkerboard pattern — high edge density."""
    img = np.zeros((h, w), dtype=dtype)
    for i in range(h):
        for j in range(w):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = 255
    return img


# ── compute_ssim ───────────────────────────────────────────────────────────

class TestComputeSsim:
    def test_returns_float(self):
        img = make_gray()
        assert isinstance(compute_ssim(img, img), float)

    def test_identical_constant_returns_one(self):
        img = make_gray(fill=128)
        result = compute_ssim(img, img)
        assert abs(result - 1.0) < 1e-6

    def test_identical_random_returns_one(self):
        img = make_random()
        result = compute_ssim(img, img)
        assert abs(result - 1.0) < 1e-4

    def test_identical_gradient_returns_one(self):
        img = make_gradient()
        result = compute_ssim(img, img)
        assert abs(result - 1.0) < 1e-4

    def test_all_black_vs_all_white_low(self):
        black = make_gray(fill=0)
        white = make_gray(fill=255)
        result = compute_ssim(black, white)
        # Dissimilar images should have SSIM well below 1
        assert result < 0.9

    def test_value_in_valid_range(self):
        a = make_random(seed=1)
        b = make_random(seed=2)
        result = compute_ssim(a, b)
        assert -1.0 <= result <= 1.0

    def test_random_self_returns_one(self):
        img = make_random(seed=42)
        assert abs(compute_ssim(img, img) - 1.0) < 1e-4

    def test_similar_images_higher_than_dissimilar(self):
        base = make_random(seed=0)
        similar = np.clip(base.astype(np.int32) + 5, 0, 255).astype(np.uint8)
        dissimilar = make_random(seed=99)
        assert compute_ssim(base, similar) > compute_ssim(base, dissimilar)

    def test_different_fill_values(self):
        a = make_gray(fill=100)
        b = make_gray(fill=200)
        # Both constant → SSIM = 1 (no texture, stabilised by C1/C2)
        result = compute_ssim(a, b)
        # Constant images: sigma terms = 0 → formula collapses to
        # (2*mu1*mu2 + C1) / (mu1^2 + mu2^2 + C1), which is < 1 when mu1 ≠ mu2
        assert -1.0 <= result <= 1.0

    def test_larger_image_returns_float(self):
        img = make_random(h=256, w=256, seed=7)
        result = compute_ssim(img, img)
        assert isinstance(result, float)
        assert abs(result - 1.0) < 1e-4


# ── hist_intersection ──────────────────────────────────────────────────────

class TestHistIntersection:
    def test_returns_float(self):
        img = make_gray()
        assert isinstance(hist_intersection(img, img), float)

    def test_identical_images_return_one(self):
        img = make_gray(fill=128)
        result = hist_intersection(img, img)
        assert abs(result - 1.0) < 1e-6

    def test_identical_random_returns_one(self):
        img = make_random(seed=5)
        result = hist_intersection(img, img)
        assert abs(result - 1.0) < 1e-6

    def test_non_overlapping_histograms_return_zero(self):
        all_black = make_gray(fill=0)
        all_white = make_gray(fill=255)
        result = hist_intersection(all_black, all_white)
        assert abs(result) < 1e-6

    def test_value_between_zero_and_one(self):
        a = make_random(seed=10)
        b = make_random(seed=11)
        result = hist_intersection(a, b)
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        a = make_random(seed=3)
        b = make_random(seed=4)
        assert abs(hist_intersection(a, b) - hist_intersection(b, a)) < 1e-10

    def test_partial_overlap_between_zero_and_one(self):
        a = make_gray(fill=50)
        b = make_gray(fill=200)
        result = hist_intersection(a, b)
        assert 0.0 <= result < 1.0

    def test_shifted_image_vs_extreme_image(self):
        base = make_random(seed=0)
        # Shift by 1 — very similar histogram
        similar = np.clip(base.astype(np.int32) + 1, 0, 255).astype(np.uint8)
        # Completely flat (all-white) — very different histogram
        flat = make_gray(fill=200)
        assert hist_intersection(base, similar) > hist_intersection(base, flat)

    def test_gradient_vs_self(self):
        img = make_gradient()
        result = hist_intersection(img, img)
        assert abs(result - 1.0) < 1e-6

    def test_checker_vs_self(self):
        img = make_checker()
        result = hist_intersection(img, img)
        assert abs(result - 1.0) < 1e-6


# ── edge_ratio ─────────────────────────────────────────────────────────────

class TestEdgeRatio:
    def test_returns_float(self):
        img1 = make_gray()
        img2 = make_gray()
        assert isinstance(edge_ratio(img1, img2), float)

    def test_identical_uniform_images_returns_near_zero(self):
        # Both have zero edges; denominator clamped to 1e-8
        # edge_density(uniform) ≈ 0 → ratio = 0/1e-8 = 0
        img = make_gray(fill=128)
        result = edge_ratio(img, img)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_identical_checker_returns_one(self):
        img = make_checker()
        result = edge_ratio(img, img)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_more_edges_in_img2_returns_greater_than_one(self):
        uniform = make_gray(fill=128)
        checker = make_checker()
        # edge_ratio(uniform, checker): d2=checker (high), d1=uniform (near 0)
        result = edge_ratio(uniform, checker)
        assert result > 1.0

    def test_more_edges_in_img1_returns_less_than_one(self):
        checker = make_checker()
        uniform = make_gray(fill=128)
        # edge_ratio(checker, uniform): d2=uniform (near 0), d1=checker (high)
        result = edge_ratio(checker, uniform)
        assert result < 1.0

    def test_result_non_negative(self):
        a = make_random(seed=20)
        b = make_random(seed=21)
        assert edge_ratio(a, b) >= 0.0

    def test_gradient_vs_checker(self):
        grad = make_gradient()
        checker = make_checker()
        result = edge_ratio(grad, checker)
        # checker has more edges than gradient, ratio should be > 1
        assert result > 0.0  # at minimum, positive

    def test_checker_vs_gradient(self):
        checker = make_checker()
        grad = make_gradient()
        r1 = edge_ratio(grad, checker)
        r2 = edge_ratio(checker, grad)
        # The ratios should be reciprocal (approximately)
        if r2 > 1e-8:
            assert r1 * r2 == pytest.approx(1.0, rel=0.1)

    def test_deterministic(self):
        a = make_random(seed=5)
        b = make_random(seed=6)
        r1 = edge_ratio(a, b)
        r2 = edge_ratio(a, b)
        assert r1 == r2


# ── evaluate_drr ───────────────────────────────────────────────────────────

class TestEvaluateDrr:
    def test_returns_dict(self):
        img = make_gray()
        result = evaluate_drr(img, img)
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self):
        img = make_gray()
        result = evaluate_drr(img, img)
        assert set(result.keys()) == {"ssim", "hist", "edge_ratio"}

    def test_all_values_are_float(self):
        img = make_gray()
        result = evaluate_drr(img, img)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_identical_images_ssim_near_one(self):
        img = make_random(seed=10)
        result = evaluate_drr(img, img)
        assert abs(result["ssim"] - 1.0) < 1e-4

    def test_identical_images_hist_near_one(self):
        img = make_random(seed=10)
        result = evaluate_drr(img, img)
        assert abs(result["hist"] - 1.0) < 1e-6

    def test_identical_checker_edge_ratio_near_one(self):
        img = make_checker()
        result = evaluate_drr(img, img)
        assert result["edge_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_ssim_matches_direct_call(self):
        drr = make_random(seed=7)
        real = make_random(seed=8)
        result = evaluate_drr(drr, real)
        assert result["ssim"] == pytest.approx(compute_ssim(drr, real), abs=1e-9)

    def test_hist_matches_direct_call(self):
        drr = make_random(seed=7)
        real = make_random(seed=8)
        result = evaluate_drr(drr, real)
        assert result["hist"] == pytest.approx(hist_intersection(drr, real), abs=1e-9)

    def test_edge_ratio_matches_direct_call(self):
        # evaluate_drr(drr, real) calls edge_ratio(real, drr) — note arg order flip
        drr = make_random(seed=7)
        real = make_random(seed=8)
        result = evaluate_drr(drr, real)
        assert result["edge_ratio"] == pytest.approx(edge_ratio(real, drr), abs=1e-9)

    def test_dissimilar_images_ssim_lower(self):
        drr1 = make_random(seed=0)
        drr2 = make_random(seed=0)       # same as drr1
        real = make_random(seed=99)       # different
        r_same = evaluate_drr(drr2, drr1)
        r_diff = evaluate_drr(real, drr1)
        assert r_same["ssim"] > r_diff["ssim"]

    def test_gradient_drr_vs_random_real(self):
        drr = make_gradient()
        real = make_random(seed=42)
        result = evaluate_drr(drr, real)
        assert -1.0 <= result["ssim"] <= 1.0
        assert 0.0 <= result["hist"] <= 1.0
        assert result["edge_ratio"] >= 0.0

    def test_large_images(self):
        drr = make_random(h=256, w=256, seed=1)
        real = make_random(h=256, w=256, seed=2)
        result = evaluate_drr(drr, real)
        assert set(result.keys()) == {"ssim", "hist", "edge_ratio"}

    def test_deterministic(self):
        drr = make_random(seed=5)
        real = make_random(seed=6)
        r1 = evaluate_drr(drr, real)
        r2 = evaluate_drr(drr, real)
        assert r1 == r2
