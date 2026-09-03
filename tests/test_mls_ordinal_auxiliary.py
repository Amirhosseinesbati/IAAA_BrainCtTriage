"""Contract tests for the training-only MLS ordinal auxiliary head."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from pydantic import ValidationError

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import ordinal_auxiliary_loss


class MLSOrdinalAuxiliaryTests(unittest.TestCase):
    def test_default_config_preserves_historical_architecture(self) -> None:
        config = MLSHeatmapConfig()
        self.assertFalse(config.use_ordinal_aux_head)
        self.assertEqual(config.ordinal_head_loss_weight, 0.0)

    def test_positive_weight_requires_explicit_head(self) -> None:
        with self.assertRaises(ValidationError):
            MLSHeatmapConfig(ordinal_head_loss_weight=0.2)

    def test_monotonic_penalty_is_zero_for_ordered_logits(self) -> None:
        logits = torch.tensor([[4.0, 1.0, -2.0]], requires_grad=True)
        total, bce, monotonic = ordinal_auxiliary_loss(
            logits, torch.tensor([3.5]), monotonic_penalty_weight=0.2,
        )
        self.assertEqual(float(monotonic.detach()), 0.0)
        self.assertEqual(float(total.detach()), float(bce.detach()))

    def test_loss_penalizes_order_violation_and_backpropagates(self) -> None:
        ordered = torch.tensor([[3.0, 1.0, -1.0]])
        violated = torch.tensor([[-1.0, 1.0, 3.0]], requires_grad=True)
        ordered_total, _, _ = ordinal_auxiliary_loss(
            ordered, torch.tensor([2.0]), monotonic_penalty_weight=0.5,
        )
        violated_total, _, violated_penalty = ordinal_auxiliary_loss(
            violated, torch.tensor([2.0]), monotonic_penalty_weight=0.5,
        )
        self.assertGreater(float(violated_penalty.detach()), 0.0)
        self.assertGreater(float(violated_total.detach()), float(ordered_total.detach()))
        violated_total.backward()
        self.assertGreater(float(violated.grad.abs().sum()), 0.0)

    def test_extended_forward_propagates_ordinal_gradient_to_shared_backbone(self) -> None:
        class TinyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)

            def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
                return [self.conv(inputs)]

        torch.manual_seed(7)
        model = HRNetHeatmapModel.__new__(HRNetHeatmapModel)
        nn.Module.__init__(model)
        model.backbone = TinyBackbone()
        model.head = nn.Conv2d(4, 3, kernel_size=1)
        model.selector_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1),
        )
        model.ordinal_aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3),
        )
        model.selector_head_mode = "single"

        inputs = torch.randn(2, 1, 3, 3)
        _, _, ordinal_logits = model.forward_multitask_extended(inputs)
        self.assertIsNotNone(ordinal_logits)
        loss, _, _ = ordinal_auxiliary_loss(
            ordinal_logits,
            torch.tensor([0.5, 6.0]),
            monotonic_penalty_weight=0.1,
        )
        loss.backward()

        shared_gradient = model.backbone.conv.weight.grad
        self.assertIsNotNone(shared_gradient)
        self.assertGreater(float(shared_gradient.abs().sum()), 0.0)

    def test_shape_contract_rejects_non_three_logit_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected \\[batch, 3\\]"):
            ordinal_auxiliary_loss(
                torch.zeros(2, 2), torch.zeros(2), monotonic_penalty_weight=0.1,
            )

    def test_boundary_weights_emphasize_five_mm_error(self) -> None:
        truth = torch.tensor([6.0])
        wrong_at_one, _, _ = ordinal_auxiliary_loss(
            torch.tensor([[-4.0, 4.0, 4.0]]), truth,
            monotonic_penalty_weight=0.0,
            boundary_weights=(0.75, 1.0, 1.25),
        )
        wrong_at_five, _, _ = ordinal_auxiliary_loss(
            torch.tensor([[4.0, 4.0, -4.0]]), truth,
            monotonic_penalty_weight=0.0,
            boundary_weights=(0.75, 1.0, 1.25),
        )
        self.assertGreater(float(wrong_at_five), float(wrong_at_one))


if __name__ == "__main__":
    unittest.main()
