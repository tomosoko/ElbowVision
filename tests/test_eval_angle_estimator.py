"""Tests for scripts/eval_angle_estimator.py pure logic and model architecture.

Covers:
  - ANGLE_MIN / ANGLE_MAX module-level constants
  - Denormalization formula: pred = out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
  - AngleEstimator model instantiation and architecture
  - Forward pass output shapes for various batch sizes
  - Gradient flow and eval/train mode behaviour

Note: weights=None is used everywhere so no download is required.
"""
import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eval_angle_estimator import (
    ANGLE_MIN,
    ANGLE_MAX,
    AngleEstimator,
)


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_angle_min_is_90(self):
        assert ANGLE_MIN == 90.0

    def test_angle_max_is_180(self):
        assert ANGLE_MAX == 180.0

    def test_angle_min_is_float(self):
        assert isinstance(ANGLE_MIN, float)

    def test_angle_max_is_float(self):
        assert isinstance(ANGLE_MAX, float)

    def test_angle_max_greater_than_min(self):
        assert ANGLE_MAX > ANGLE_MIN

    def test_angle_range_is_90(self):
        assert ANGLE_MAX - ANGLE_MIN == 90.0


# ── Denormalization formula ────────────────────────────────────────────────────
#
# The script uses: pred_angle = out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
# This section validates that the formula maps the normalised range [0, 1]
# to the physical angle range [ANGLE_MIN, ANGLE_MAX].

class TestDenormalizationFormula:
    """Verify the inverse-normalisation formula used during inference."""

    @staticmethod
    def denorm(out: float) -> float:
        return out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN

    def test_zero_maps_to_angle_min(self):
        assert self.denorm(0.0) == pytest.approx(ANGLE_MIN)

    def test_one_maps_to_angle_max(self):
        assert self.denorm(1.0) == pytest.approx(ANGLE_MAX)

    def test_half_maps_to_midpoint(self):
        expected = (ANGLE_MIN + ANGLE_MAX) / 2.0
        assert self.denorm(0.5) == pytest.approx(expected)

    def test_midpoint_is_135(self):
        assert self.denorm(0.5) == pytest.approx(135.0)

    def test_negative_extrapolates_below_min(self):
        result = self.denorm(-1.0)
        assert result < ANGLE_MIN

    def test_greater_than_one_extrapolates_above_max(self):
        result = self.denorm(2.0)
        assert result > ANGLE_MAX

    def test_batch_denorm_via_numpy(self):
        out = np.array([0.0, 0.5, 1.0])
        preds = out * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
        assert preds[0] == pytest.approx(ANGLE_MIN)
        assert preds[1] == pytest.approx(135.0)
        assert preds[2] == pytest.approx(ANGLE_MAX)

    def test_denorm_is_linear(self):
        """Equal steps in normalised space → equal steps in degree space."""
        steps = np.linspace(0.0, 1.0, 11)
        preds = steps * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
        diffs = np.diff(preds)
        assert np.allclose(diffs, diffs[0])

    def test_quarter_maps_to_112_5(self):
        assert self.denorm(0.25) == pytest.approx(112.5)

    def test_three_quarters_maps_to_157_5(self):
        assert self.denorm(0.75) == pytest.approx(157.5)


# ── AngleEstimator instantiation ──────────────────────────────────────────────

class TestAngleEstimatorInit:
    def test_instantiation_succeeds(self):
        model = AngleEstimator()
        assert model is not None

    def test_is_nn_module(self):
        model = AngleEstimator()
        assert isinstance(model, nn.Module)

    def test_has_backbone_attribute(self):
        model = AngleEstimator()
        assert hasattr(model, "backbone")

    def test_backbone_has_classifier(self):
        model = AngleEstimator()
        assert hasattr(model.backbone, "classifier")

    def test_final_linear_output_features_is_1(self):
        model = AngleEstimator()
        final_layer = model.backbone.classifier[2]
        assert isinstance(final_layer, nn.Linear)
        assert final_layer.out_features == 1

    def test_final_linear_in_features_is_768(self):
        """ConvNeXt-Small classifier in_features == 768."""
        model = AngleEstimator()
        final_layer = model.backbone.classifier[2]
        assert final_layer.in_features == 768

    def test_multiple_instances_independent(self):
        m1 = AngleEstimator()
        m2 = AngleEstimator()
        assert m1 is not m2


# ── Forward pass ──────────────────────────────────────────────────────────────

class TestAngleEstimatorForward:
    @pytest.fixture
    def model(self):
        m = AngleEstimator()
        m.eval()
        return m

    @staticmethod
    def _random_batch(b: int) -> torch.Tensor:
        """Random ImageNet-normalised batch (B, 3, 224, 224)."""
        return torch.randn(b, 3, 224, 224)

    def test_output_shape_batch1(self, model):
        x = self._random_batch(1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_output_shape_batch4(self, model):
        x = self._random_batch(4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,)

    def test_output_shape_batch8(self, model):
        x = self._random_batch(8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8,)

    def test_output_is_1d(self, model):
        x = self._random_batch(3)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 1

    def test_output_dtype_is_float(self, model):
        x = self._random_batch(2)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, model):
        x = self._random_batch(4)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, model):
        x = self._random_batch(4)
        with torch.no_grad():
            out = model(x)
        assert not torch.isinf(out).any()

    def test_eval_mode_is_deterministic(self, model):
        x = self._random_batch(2)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_give_different_outputs(self, model):
        x1 = self._random_batch(1)
        x2 = self._random_batch(1)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        # Vanishingly unlikely to collide for random inputs
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_to_input(self):
        """Backprop must reach the input tensor."""
        model = AngleEstimator()
        model.train()
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_zero_input_no_crash(self, model):
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_ones_input_no_crash(self, model):
        x = torch.ones(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)


# ── Eval / train mode ─────────────────────────────────────────────────────────

class TestAngleEstimatorMode:
    def test_default_is_training_mode(self):
        model = AngleEstimator()
        assert model.training is True

    def test_eval_mode_switchable(self):
        model = AngleEstimator()
        model.eval()
        assert model.training is False

    def test_can_switch_back_to_train(self):
        model = AngleEstimator()
        model.eval()
        model.train()
        assert model.training is True

    def test_eval_propagates_to_backbone(self):
        model = AngleEstimator()
        model.eval()
        assert model.backbone.training is False

    def test_parameters_exist(self):
        model = AngleEstimator()
        params = list(model.parameters())
        assert len(params) > 0

    def test_requires_grad_by_default(self):
        model = AngleEstimator()
        params = list(model.parameters())
        assert all(p.requires_grad for p in params)
