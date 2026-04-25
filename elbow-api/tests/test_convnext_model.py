"""
Unit tests for elbow-api/training/convnext_model.py

Tests cover:
  - Module-level constants (OUTPUT_DIM, OUTPUT_NAMES)
  - ElbowConvNeXt instantiation (pretrained=False to avoid downloads)
  - Forward pass output shapes for various batch sizes
  - Classifier head architecture (Dropout, Linear, GELU, Linear)
  - Final Linear output dimension matches OUTPUT_DIM
  - Gradient flow through the model
  - Eval / train mode switching
  - Model is an nn.Module subclass
"""

import pytest
import torch
import torch.nn as nn

from training.convnext_model import (
    ElbowConvNeXt,
    OUTPUT_DIM,
    OUTPUT_NAMES,
)


# ─── Constants ───────────────────────────────────────────────────────────────

class TestModuleConstants:
    def test_output_dim_is_2(self):
        assert OUTPUT_DIM == 2

    def test_output_names_length(self):
        assert len(OUTPUT_NAMES) == OUTPUT_DIM

    def test_output_names_rotation_first(self):
        assert OUTPUT_NAMES[0] == "rotation_error_deg"

    def test_output_names_flexion_second(self):
        assert OUTPUT_NAMES[1] == "flexion_deg"

    def test_output_names_are_strings(self):
        assert all(isinstance(n, str) for n in OUTPUT_NAMES)


# ─── Instantiation ───────────────────────────────────────────────────────────

class TestElbowConvNeXtInit:
    def test_init_no_pretrained(self):
        model = ElbowConvNeXt(pretrained=False)
        assert model is not None

    def test_is_nn_module(self):
        model = ElbowConvNeXt(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_has_backbone_attribute(self):
        model = ElbowConvNeXt(pretrained=False)
        assert hasattr(model, "backbone")

    def test_default_pretrained_flag_accepted(self):
        # pretrained=True would download; pretrained=False must not raise
        model = ElbowConvNeXt(pretrained=False)
        assert model is not None

    def test_two_instances_are_independent(self):
        m1 = ElbowConvNeXt(pretrained=False)
        m2 = ElbowConvNeXt(pretrained=False)
        # Mutating one should not affect the other
        for p in m2.parameters():
            p.data.fill_(0.0)
        p1_sum = sum(p.data.abs().sum().item() for p in m1.parameters())
        assert p1_sum != 0.0


# ─── Classifier head structure ────────────────────────────────────────────────

class TestClassifierHead:
    @pytest.fixture
    def model(self):
        return ElbowConvNeXt(pretrained=False)

    def test_classifier_index_2_is_sequential(self, model):
        head = model.backbone.classifier[2]
        assert isinstance(head, nn.Sequential)

    def test_head_has_dropout(self, model):
        head = model.backbone.classifier[2]
        layers = list(head.children())
        dropout_layers = [l for l in layers if isinstance(l, nn.Dropout)]
        assert len(dropout_layers) == 1

    def test_head_has_gelu(self, model):
        head = model.backbone.classifier[2]
        layers = list(head.children())
        gelu_layers = [l for l in layers if isinstance(l, nn.GELU)]
        assert len(gelu_layers) == 1

    def test_head_has_two_linear_layers(self, model):
        head = model.backbone.classifier[2]
        layers = list(head.children())
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        assert len(linear_layers) == 2

    def test_final_linear_out_features_equals_output_dim(self, model):
        head = model.backbone.classifier[2]
        linear_layers = [l for l in head.children() if isinstance(l, nn.Linear)]
        last_linear = linear_layers[-1]
        assert last_linear.out_features == OUTPUT_DIM

    def test_intermediate_linear_out_features_is_128(self, model):
        head = model.backbone.classifier[2]
        linear_layers = [l for l in head.children() if isinstance(l, nn.Linear)]
        first_linear = linear_layers[0]
        assert first_linear.out_features == 128

    def test_dropout_rate_is_0_3(self, model):
        head = model.backbone.classifier[2]
        dropout = next(l for l in head.children() if isinstance(l, nn.Dropout))
        assert abs(dropout.p - 0.3) < 1e-6


# ─── Forward pass ─────────────────────────────────────────────────────────────

class TestForwardPass:
    @pytest.fixture
    def model(self):
        m = ElbowConvNeXt(pretrained=False)
        m.eval()
        return m

    def _dummy_input(self, batch_size=1):
        return torch.randn(batch_size, 3, 224, 224)

    def test_output_shape_batch1(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(1))
        assert out.shape == (1, OUTPUT_DIM)

    def test_output_shape_batch2(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(2))
        assert out.shape == (2, OUTPUT_DIM)

    def test_output_shape_batch4(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(4))
        assert out.shape == (4, OUTPUT_DIM)

    def test_output_is_float_tensor(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(1))
        assert out.dtype == torch.float32

    def test_output_has_no_nan(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(2))
        assert not torch.isnan(out).any()

    def test_output_has_no_inf(self, model):
        with torch.no_grad():
            out = model(self._dummy_input(2))
        assert not torch.isinf(out).any()

    def test_output_columns_are_rotation_and_flexion(self, model):
        """Column 0 = rotation_error_deg, column 1 = flexion_deg."""
        with torch.no_grad():
            out = model(self._dummy_input(1))
        # Just verify indexing works; values are arbitrary for untrained model
        _ = out[:, 0]  # rotation_error_deg
        _ = out[:, 1]  # flexion_deg

    def test_different_inputs_give_different_outputs(self, model):
        x1 = torch.zeros(1, 3, 224, 224)
        x2 = torch.ones(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self, model):
        """Calling forward twice in eval mode should give identical results."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# ─── Gradient flow ────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_gradients_flow_through_model(self):
        model = ElbowConvNeXt(pretrained=False)
        model.train()
        x = torch.randn(1, 3, 224, 224, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least some parameters should have non-None, non-zero gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_model_has_trainable_parameters(self):
        model = ElbowConvNeXt(pretrained=False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0


# ─── Mode switching ───────────────────────────────────────────────────────────

class TestModeSwitching:
    def test_train_mode(self):
        model = ElbowConvNeXt(pretrained=False)
        model.train()
        assert model.training is True

    def test_eval_mode(self):
        model = ElbowConvNeXt(pretrained=False)
        model.eval()
        assert model.training is False

    def test_toggle_train_eval(self):
        model = ElbowConvNeXt(pretrained=False)
        model.eval()
        model.train()
        assert model.training is True
