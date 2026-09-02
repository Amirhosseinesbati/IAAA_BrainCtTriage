"""Pure metric tests for canonical deploy-aligned MLS comparison."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_mls_deploy_aligned_seed_medians import (
    _classification_metrics,
    _load_oof,
    _parse_fold_path,
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

    def test_fold_four_is_supported_for_full_oof(self) -> None:
        fold, path = _parse_fold_path("4=fold4.csv")
        self.assertEqual(fold, 4)
        self.assertEqual(path, Path("fold4.csv"))

    def test_oof_loader_enforces_exact_immutable_fold_membership(self) -> None:
        manifest = pd.DataFrame([
            {"study_id": "a", "patient_id": "p1", "triage_class": 0, "fold": 3},
            {"study_id": "b", "patient_id": "p2", "triage_class": 1, "fold": 4},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold4.csv"
            pd.DataFrame([{
                "study_id": "b",
                "patient_id": "p2",
                "triage_class": 1,
                "gt_MLS_mm": 2.0,
                "median_MLS_mm": 2.1,
            }]).to_csv(path, index=False)
            loaded, sources = _load_oof([(4, path)], "candidate", manifest)
            self.assertEqual(loaded["study_id"].tolist(), ["b"])
            self.assertEqual(sources[0]["fold"], 4)

            wrong = pd.DataFrame([{
                "study_id": "a",
                "patient_id": "p1",
                "triage_class": 0,
                "gt_MLS_mm": 0.0,
                "median_MLS_mm": 0.0,
            }])
            wrong.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "membership"):
                _load_oof([(4, path)], "candidate", manifest)


if __name__ == "__main__":
    unittest.main()
