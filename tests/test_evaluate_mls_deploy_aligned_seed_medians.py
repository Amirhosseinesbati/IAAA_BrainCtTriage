"""Pure metric tests for canonical deploy-aligned MLS comparison."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_mls_deploy_aligned_seed_medians import (
    _classification_metrics,
    _threshold_metrics,
)


class DeployAlignedSeedMedianMetricTests(unittest.TestCase):
    def test_perfect_classification_has_unit_macro_and_all_classes(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2])
        metrics = _classification_metrics(truth, truth.copy())
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(set(metrics["per_class"]), {"Normal", "Urgent", "Critical"})
        self.assertEqual(metrics["catastrophic_errors"]["normal_to_critical"], 0)

    def test_threshold_metrics_expose_one_three_five(self) -> None:
        truth = np.asarray([0.5, 2.0, 4.0, 6.0])
        metrics = _threshold_metrics(truth, truth.copy())
        self.assertEqual(metrics, {"f1_1mm": 1.0, "f1_3mm": 1.0, "f1_5mm": 1.0})


if __name__ == "__main__":
    unittest.main()
