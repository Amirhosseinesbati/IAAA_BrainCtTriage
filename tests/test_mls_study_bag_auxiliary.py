from __future__ import annotations

import unittest

import torch

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import collate_mls_study_bag
from src.strategies.mls_heatmap.train_multitask import study_bag_selection_loss


def _config() -> MLSHeatmapConfig:
    return MLSHeatmapConfig(
        use_selector=True,
        selector_head_mode="dual",
        dataset_variant="multitask_v2",
        image_size=256,
        study_bag_loss_weight=0.2,
    )


class MLSStudyBagAuxiliaryTests(unittest.TestCase):
    def test_collate_rejects_mixed_studies_and_stacks_one_bag(self) -> None:
        sample = (
            torch.zeros(3, 16, 16),
            torch.zeros(3, 4, 4),
            torch.ones(3),
            torch.zeros(3, 2),
            torch.tensor(0.5),
            torch.tensor(1.0),
            torch.tensor(3.0),
            "study-a",
        )
        stacked = collate_mls_study_bag([[sample, sample]])
        self.assertEqual(tuple(stacked[0].shape), (2, 3, 16, 16))
        self.assertEqual(stacked[-1], ("study-a", "study-a"))
        mixed = (*sample[:-1], "study-b")
        with self.assertRaisesRegex(ValueError, "more than one study"):
            collate_mls_study_bag([[sample, mixed]])

    def test_loss_is_finite_differentiable_and_requires_one_study_target(self) -> None:
        config = _config()
        heatmaps = torch.zeros(2, 3, 4, 4, requires_grad=True)
        selector = torch.zeros(2, 2, requires_grad=True)
        masks = torch.ones(2, 3)
        keypoints = torch.tensor([
            [[4.0, 1.0], [4.0, 12.0], [6.0, 8.0]],
            [[4.0, 1.0], [4.0, 12.0], [6.0, 8.0]],
        ])
        spacing = torch.tensor([0.5, 0.5])
        is_target = torch.ones(2)
        study_mls = torch.tensor([1.0, 1.0])
        loss, parts = study_bag_selection_loss(
            heatmaps, selector, masks, keypoints, spacing, is_target, study_mls, config,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["study_bag_regression"]))
        loss.backward()
        self.assertTrue(torch.isfinite(heatmaps.grad).all())
        self.assertTrue(torch.isfinite(selector.grad).all())
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            study_bag_selection_loss(
                heatmaps.detach(), selector.detach(), masks, keypoints, spacing,
                is_target, torch.tensor([1.0, 3.0]), config,
            )

    def test_config_refuses_bag_loss_without_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires use_selector"):
            MLSHeatmapConfig(
                use_selector=False,
                dataset_variant="multitask_v2",
                study_bag_loss_weight=0.2,
            )


if __name__ == "__main__":
    unittest.main()
