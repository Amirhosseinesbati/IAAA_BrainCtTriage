import unittest

import torch

from scripts.diagnose_ich_multitask_gradient_conflict import (
    _classification_loss_for_label,
    _gradient_geometry,
    _summary,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


class ICHGradientConflictDiagnosticTests(unittest.TestCase):
    def test_gradient_geometry_detects_conflict_and_ignores_unused_parameters(self):
        cosine, left_norm, right_norm = _gradient_geometry(
            (torch.tensor([1.0, 0.0]), None),
            (torch.tensor([-1.0, 0.0]), torch.tensor([4.0])),
        )
        self.assertEqual(cosine, -1.0 / (17.0 ** 0.5))
        self.assertEqual(left_norm, 1.0)
        self.assertEqual(right_norm, 17.0 ** 0.5)

    def test_gradient_geometry_reports_zero_norm_without_false_cosine(self):
        cosine, left_norm, right_norm = _gradient_geometry(
            (torch.zeros(2),),
            (torch.ones(2),),
        )
        self.assertIsNone(cosine)
        self.assertEqual(left_norm, 0.0)
        self.assertAlmostEqual(right_norm, 2.0 ** 0.5)

    def test_per_label_classification_losses_reconstruct_multilabel_mean(self):
        logits = torch.tensor(
            [[0.2, -0.1, 0.7, -0.4, 0.3, -0.8], [0.4, 0.5, -0.2, 0.1, -0.6, 0.9]],
            requires_grad=True,
        )
        targets = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        )
        known = torch.ones(2)
        pos_weight = torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5, 4.0])
        losses = [
            _classification_loss_for_label(
                logits,
                targets,
                known,
                label_index=index,
                pos_weight=pos_weight,
                focal_gamma=1.0,
            )
            for index in range(len(OUTPUT_LABELS))
        ]

        selected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        correct = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        expected = (selected_bce * (1.0 - correct)).mean()
        self.assertTrue(torch.allclose(torch.stack(losses).mean(), expected))

    def test_summary_reports_conflict_fraction(self):
        result = _summary([-0.5, 0.0, 0.5, -0.25])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["negative_fraction"], 0.5)
        self.assertEqual(result["median"], -0.125)


if __name__ == "__main__":
    unittest.main()
