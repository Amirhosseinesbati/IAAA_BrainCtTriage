from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ich_segmentation_promotion import compare_runs


METRICS = {
    "selection_score": 0.63,
    "mean_foreground_dice": 0.40,
    "any_ich_study_auc": 0.95,
    "macro_subtype_study_auc": 0.85,
    "presence_f1_at_0_1ml": 0.78,
    "normal_false_positive_rate_at_0_1ml": 0.43,
    "total_volume_mae_ml": 9.0,
}


def _run(root: Path, name: str, metrics: dict[str, float], **config) -> Path:
    run = root / name
    run.mkdir()
    resolved = {
        "run_name": name,
        "output_dir": str(run),
        "outer_fold": 0,
        "calibration_fold": 1,
        "empty_foreground_weight": 0.0,
        "empty_foreground_top_fraction": 1.0,
        "checkpoint_selection_strategy": "legacy",
        **config,
    }
    (run / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    (run / "run_summary.json").write_text(
        json.dumps({"manifest_sha256": "manifest"}), encoding="utf-8"
    )
    (run / "outer_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


class ICHSegmentationPromotionTests(unittest.TestCase):
    def test_candidate_advances_only_when_every_primary_gate_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", METRICS)
            candidate_metrics = {
                **METRICS,
                "selection_score": 0.628,
                "mean_foreground_dice": 0.395,
                "presence_f1_at_0_1ml": 0.80,
                "normal_false_positive_rate_at_0_1ml": 0.35,
                "total_volume_mae_ml": 8.8,
            }
            candidate = _run(
                root,
                "candidate",
                candidate_metrics,
                empty_foreground_weight=0.05,
                checkpoint_selection_strategy="fpr_penalized",
            )
            result = compare_runs(baseline, candidate)
            self.assertTrue(result["all_primary_gates_passed"])
            self.assertEqual(result["decision"], "advance_to_independent_fold")
            self.assertEqual(result["mae_warning"], "candidate_mae_not_worse")

    def test_candidate_is_rejected_when_fpr_does_not_improve_enough(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", METRICS)
            candidate = _run(
                root,
                "candidate",
                {**METRICS, "normal_false_positive_rate_at_0_1ml": 0.40},
                empty_foreground_weight=0.05,
            )
            result = compare_runs(baseline, candidate)
            self.assertFalse(result["all_primary_gates_passed"])
            self.assertFalse(result["gates"]["fpr_reduction_at_least_0_05"]["passed"])

    def test_unexpected_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", METRICS, batch_size=16)
            candidate = _run(
                root,
                "candidate",
                METRICS,
                empty_foreground_weight=0.05,
                batch_size=8,
            )
            with self.assertRaisesRegex(ValueError, "approved-method"):
                compare_runs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
