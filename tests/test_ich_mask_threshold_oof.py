import unittest

import numpy as np

from scripts.evaluate_ich_mask_threshold_oof import (
    PROMOTION_POLICY,
    promotion_decision,
    two_sided_presence_gate,
)


class TestICHMaskThresholdOOF(unittest.TestCase):
    def test_two_sided_gate_preserves_hard_presence_decision(self) -> None:
        hard = np.asarray([0.0, 0.05, 0.10, 2.0])
        threshold = np.asarray([0.0, 1.0, 0.05, 3.0])

        use_threshold = two_sided_presence_gate(hard, threshold)

        np.testing.assert_array_equal(use_threshold, [False, False, False, True])

    def test_promotion_requires_uncertainty_and_fold_robustness_gates(self) -> None:
        metrics = {}
        for metric in (
            "mean_foreground_dice",
            "selection_score",
            "total_volume_mae_ml",
        ):
            metrics[metric] = {
                "bootstrap_probability_candidate_better": 0.99,
                "delta_ci95": [0.001, 0.01],
            }
        metrics["total_volume_mae_ml"]["delta_ci95"] = [-0.5, -0.01]
        bootstrap = {"metrics": metrics}
        hard = {
            "normal_false_positive_rate_at_0_1ml": 0.2,
            "presence_f1_at_0_1ml": 0.8,
            "any_ich_study_auc": 0.9,
            "macro_subtype_study_auc": 0.8,
            "total_volume_bias_ml": -3.0,
        }
        gated = {**hard, "total_volume_bias_ml": -2.0}
        folds = [
            {"deltas": {"selection_score": value}}
            for value in (0.01, 0.02, 0.01, -0.001, -0.002)
        ]

        accepted = promotion_decision(hard, gated, bootstrap, folds)
        self.assertTrue(accepted["promotion_allowed"])

        bootstrap["metrics"]["mean_foreground_dice"][
            "bootstrap_probability_candidate_better"
        ] = PROMOTION_POLICY["minimum_dice_probability"] - 0.01
        rejected = promotion_decision(hard, gated, bootstrap, folds)
        self.assertFalse(rejected["promotion_allowed"])
        self.assertFalse(rejected["gates"]["dice_probability"])


if __name__ == "__main__":
    unittest.main()
