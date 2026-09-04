"""Contracts for optional signed geometry supervision in MLS training."""

from __future__ import annotations

import unittest

import torch

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.train_multitask import signed_falx_offset_mm


class MLSSignedGeometryLossTests(unittest.TestCase):
    def test_default_preserves_historical_absolute_only_loss(self) -> None:
        self.assertEqual(MLSHeatmapConfig().signed_offset_loss_weight, 0.0)

    def test_signed_offset_distinguishes_mirrored_outer_points(self) -> None:
        # The attachment order is fixed by the annotation schema. Mirroring the
        # outermost point keeps absolute MLS unchanged but flips laterality.
        positive = torch.tensor([[
            [256.0, 100.0], [256.0, 400.0], [266.0, 250.0],
        ]], requires_grad=True)
        mirrored = torch.tensor([[
            [256.0, 100.0], [256.0, 400.0], [246.0, 250.0],
        ]])
        spacing = torch.tensor([0.5])
        positive_offset = signed_falx_offset_mm(positive, spacing)
        mirrored_offset = signed_falx_offset_mm(mirrored, spacing)
        self.assertAlmostEqual(float(positive_offset.detach()), 5.0)
        self.assertAlmostEqual(float(mirrored_offset.detach()), -5.0)
        self.assertAlmostEqual(
            float(positive_offset.detach().abs()), float(mirrored_offset.abs()),
        )
        loss = torch.nn.functional.smooth_l1_loss(positive_offset, mirrored_offset)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)

    def test_shape_contract_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape \\[batch, 3, 2\\]"):
            signed_falx_offset_mm(torch.zeros(2, 2), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
