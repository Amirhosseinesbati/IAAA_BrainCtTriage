"""Unit contract for opt-in MLS left-right flip TTA fusion.

The test is deliberately device-agnostic; execution is deferred to the remote
GPU validation session under the project's CUDA-only model-operation policy.
"""

from __future__ import annotations

import unittest

import torch

from src.strategies.mls_heatmap.predict_multitask import (
    fuse_horizontal_flip_tta_probabilities,
)


class TestMLSFlipTTA(unittest.TestCase):
    def test_restores_reflected_heatmap_to_original_x_axis(self) -> None:
        original = torch.tensor([[[[0.0, 8.0, 0.0, 0.0]]]])
        reflected = torch.tensor([[[[0.0, 0.0, 8.0, 0.0]]]])
        heatmap, target, peak = fuse_horizontal_flip_tta_probabilities(
            original,
            torch.tensor([0.0]),
            reflected,
            torch.tensor([0.0]),
            "single",
        )
        self.assertEqual(tuple(heatmap.shape), (1, 1, 1, 4))
        self.assertEqual(int(heatmap.flatten().argmax()), 1)
        self.assertAlmostEqual(float(target.item()), 0.5, places=7)
        self.assertAlmostEqual(float(peak.item()), 0.5, places=7)

    def test_dual_selector_averages_probabilities_not_logits(self) -> None:
        original = torch.zeros((1, 3, 2, 2))
        reflected = torch.zeros((1, 3, 2, 2))
        _, target, peak = fuse_horizontal_flip_tta_probabilities(
            original,
            torch.tensor([[2.0, -2.0]]),
            reflected,
            torch.tensor([[0.0, 0.0]]),
            "dual",
        )
        self.assertAlmostEqual(float(target.item()), (float(torch.sigmoid(torch.tensor(2.0))) + 0.5) / 2.0, places=7)
        self.assertAlmostEqual(float(peak.item()), (float(torch.sigmoid(torch.tensor(-2.0))) + 0.5) / 2.0, places=7)


if __name__ == "__main__":
    unittest.main()
