"""Pure metric tests for canonical deploy-aligned MLS comparison."""

from __future__ import annotations

import hashlib
import json
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
    @staticmethod
    def _write_audit_summary(csv_path: Path, summary_path: Path, *, fold: int) -> None:
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        checkpoints = {
            "seed42": {"epoch": 15, "seed": 42, "sha256": "a" * 64},
            "seed2026": {"epoch": 15, "seed": 2026, "sha256": "b" * 64},
            "seed3407": {"epoch": 15, "seed": 3407, "sha256": "c" * 64},
        }
        summary_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "completed",
            "protocol": "heldout_fold_fixed_epoch15_three_distinct_seed_median",
            "compute_policy": "cuda_only_no_cpu_model_fallback",
            "fold": fold,
            "studies": 1,
            "fixed_epoch": 15,
            "seeds": [42, 2026, 3407],
            "config_differences": ["seed", "snapshot_start_epoch"],
            "checkpoint_manifest": checkpoints,
            "private_predictions_sha256": digest,
            "raw_predictions_uploaded_to_mlflow": False,
        }), encoding="utf-8")

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
            summary_path = Path(directory) / "fold4_summary.json"
            pd.DataFrame([{
                "study_id": "b",
                "patient_id": "p2",
                "triage_class": 1,
                "gt_MLS_mm": 2.0,
                "median_MLS_mm": 2.1,
                "seed42_MLS_mm": 2.0,
                "seed2026_MLS_mm": 2.1,
                "seed3407_MLS_mm": 2.2,
                "error": "",
            }]).to_csv(path, index=False)
            self._write_audit_summary(path, summary_path, fold=4)
            loaded, sources = _load_oof(
                [(4, path)], [(4, summary_path)], "candidate", manifest,
            )
            self.assertEqual(loaded["study_id"].tolist(), ["b"])
            self.assertEqual(sources[0]["fold"], 4)
            self.assertEqual(set(sources[0]["checkpoint_sha256"]), {
                "seed42", "seed2026", "seed3407",
            })

            wrong = pd.DataFrame([{
                "study_id": "a",
                "patient_id": "p1",
                "triage_class": 0,
                "gt_MLS_mm": 0.0,
                "median_MLS_mm": 0.0,
                "seed42_MLS_mm": 0.0,
                "seed2026_MLS_mm": 0.0,
                "seed3407_MLS_mm": 0.0,
                "error": "",
            }])
            wrong.to_csv(path, index=False)
            self._write_audit_summary(path, summary_path, fold=4)
            with self.assertRaisesRegex(ValueError, "membership"):
                _load_oof([(4, path)], [(4, summary_path)], "candidate", manifest)

    def test_oof_loader_rejects_csv_not_bound_to_audit_summary(self) -> None:
        manifest = pd.DataFrame([{
            "study_id": "a", "patient_id": "p1", "triage_class": 0, "fold": 0,
        }])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold0.csv"
            summary_path = Path(directory) / "fold0_summary.json"
            frame = pd.DataFrame([{
                "study_id": "a", "patient_id": "p1", "triage_class": 0,
                "gt_MLS_mm": 0.1, "median_MLS_mm": 0.2,
                "seed42_MLS_mm": 0.1, "seed2026_MLS_mm": 0.2,
                "seed3407_MLS_mm": 0.3, "error": "",
            }])
            frame.to_csv(path, index=False)
            self._write_audit_summary(path, summary_path, fold=0)
            frame.loc[0, "seed3407_MLS_mm"] = 9.0
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "private_predictions_sha256"):
                _load_oof([(0, path)], [(0, summary_path)], "candidate", manifest)

    def test_oof_loader_rejects_incorrect_stored_median(self) -> None:
        manifest = pd.DataFrame([{
            "study_id": "a", "patient_id": "p1", "triage_class": 0, "fold": 0,
        }])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold0.csv"
            summary_path = Path(directory) / "fold0_summary.json"
            pd.DataFrame([{
                "study_id": "a", "patient_id": "p1", "triage_class": 0,
                "gt_MLS_mm": 0.1, "median_MLS_mm": 7.0,
                "seed42_MLS_mm": 0.1, "seed2026_MLS_mm": 0.2,
                "seed3407_MLS_mm": 0.3, "error": "",
            }]).to_csv(path, index=False)
            self._write_audit_summary(path, summary_path, fold=0)
            with self.assertRaisesRegex(ValueError, "Stored median"):
                _load_oof([(0, path)], [(0, summary_path)], "candidate", manifest)


if __name__ == "__main__":
    unittest.main()
