"""Contract tests for the training-only MLS ordinal auxiliary head."""

from __future__ import annotations

import unittest

import torch
from pydantic import ValidationError

from src.strategies.config_models import MLSHeatmapConfig
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
