from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import (
    MLSPositiveStudyPairDataset,
    collate_mls_study_pair,
    rotate_image_and_keypoints,
    translate_image_and_keypoints,
)
from src.strategies.mls_heatmap.train import differentiable_mls_mm
from src.strategies.mls_heatmap.train_multitask import within_study_pair_rank_loss


def _config() -> MLSHeatmapConfig:
    return MLSHeatmapConfig(
        use_selector=True,
        dataset_variant="multitask_v2",
        within_study_rank_loss_weight=0.1,
        within_study_rank_min_gap_mm=1.0,
    )


def _sample(study_id: str) -> tuple:
    return (
        torch.zeros(3, 16, 16),
        torch.zeros(3, 4, 4),
        torch.ones(3),
        torch.tensor([[4.0, 1.0], [4.0, 12.0], [6.0, 8.0]]),
        torch.tensor(0.5),
        torch.tensor(1.0),
        torch.tensor(4.0),
        study_id,
    )


class MLSWithinStudyRankAuxiliaryTests(unittest.TestCase):
    def test_pair_dataset_never_crosses_study_and_enumerates_pairs(self) -> None:
        class TinyDataset:
            return_selector = True
            data = pd.DataFrame({
                "patient_id": ["a", "a", "a", "b", "b"],
                "is_target": [1, 1, 1, 1, 0],
            })

            def __getitem__(self, index: int) -> tuple:
                return _sample(str(self.data.iloc[index]["patient_id"]))

        pairs = MLSPositiveStudyPairDataset(TinyDataset())
        self.assertEqual(len(pairs), 3)
        first, second = pairs[0]
        self.assertEqual(first[-1], second[-1])

    def test_pair_collate_stacks_one_same_study_pair(self) -> None:
        pair = collate_mls_study_pair([(_sample("a"), _sample("a"))])
        self.assertEqual(tuple(pair[0].shape), (2, 3, 16, 16))
        self.assertEqual(pair[-1], ("a", "a"))
        with self.assertRaisesRegex(ValueError, "more than one study"):
            collate_mls_study_pair([(_sample("a"), _sample("b"))])

    def test_rank_loss_prefers_higher_local_mls_and_backpropagates(self) -> None:
        config = _config()
        selector = torch.tensor([2.0, -2.0], requires_grad=True)
        masks = torch.ones(2, 3)
        keypoints = torch.tensor([
            [[4.0, 1.0], [4.0, 12.0], [10.0, 8.0]],
            [[4.0, 1.0], [4.0, 12.0], [5.0, 8.0]],
        ])
        loss, parts = within_study_pair_rank_loss(
            selector,
            masks,
            keypoints,
            torch.tensor([0.5, 0.5]),
            torch.ones(2),
            config,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(parts["within_study_rank_qualified_pairs"]), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(selector.grad).all())
        self.assertGreater(float(selector.grad[0]), -0.1)

    def test_small_gap_is_fail_closed_no_gradient_auxiliary(self) -> None:
        config = _config()
        selector = torch.zeros(2, requires_grad=True)
        keypoints = torch.tensor([
            [[4.0, 1.0], [4.0, 12.0], [6.0, 8.0]],
            [[4.0, 1.0], [4.0, 12.0], [6.5, 8.0]],
        ])
        loss, parts = within_study_pair_rank_loss(
            selector,
            torch.ones(2, 3),
            keypoints,
            torch.tensor([0.5, 0.5]),
            torch.ones(2),
            config,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["within_study_rank_qualified_pairs"]), 0.0)

    def test_rank_target_is_invariant_to_independent_rigid_augmentations(self) -> None:
        """Pair supervision must use the same local-MLS order after augmentation."""
        image = np.zeros((32, 32, 3), dtype=np.float32)
        first = np.array([[10.0, 5.0], [10.0, 25.0], [16.0, 15.0]], dtype=np.float32)
        second = np.array([[10.0, 5.0], [10.0, 25.0], [13.0, 15.0]], dtype=np.float32)
        _image, first = rotate_image_and_keypoints(image, first, 7.0, 32)
        _image, first = translate_image_and_keypoints(image, first, 2.0, -1.0)
        _image, second = rotate_image_and_keypoints(image, second, -6.0, 32)
        _image, second = translate_image_and_keypoints(image, second, -3.0, 2.0)
        transformed = differentiable_mls_mm(
            torch.tensor(np.stack([first, second]), dtype=torch.float32),
            torch.tensor([0.5, 0.5]),
        )
        self.assertGreater(float(transformed[0]), float(transformed[1]))
        self.assertTrue(torch.allclose(transformed, torch.tensor([3.0, 1.5]), atol=1e-5))

    def test_config_refuses_ranking_without_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires use_selector"):
            MLSHeatmapConfig(
                use_selector=False,
                dataset_variant="multitask_v2",
                within_study_rank_loss_weight=0.1,
            )


if __name__ == "__main__":
    unittest.main()
