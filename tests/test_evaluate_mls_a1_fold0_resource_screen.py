from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_mls_a1_fold0_resource_screen import EXPECTED_GATES, evaluate


def _classification(macro: float, accuracy: float, urgent: float) -> dict:
    return {
        "macro_f1": macro,
        "accuracy": accuracy,
        "per_class": {
            "Normal": {"f1": 0.90},
            "Urgent": {"f1": urgent},
            "Critical": {"f1": 0.88},
        },
    }


def _preregistration() -> dict:
    return {
        "status": "locked_before_any_a1_training_outcome",
        "fixed_screen_gates": list(EXPECTED_GATES),
        "control": {
            "audit_summary_sha256": "a" * 64,
            "private_predictions_sha256": "b" * 64,
        },
    }


def _summary() -> dict:
    baseline = _classification(0.80, 0.84, 0.70)
    candidate = _classification(0.82, 0.85, 0.74)
    return {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "selected_folds": [0],
        "studies": 70,
        "full_fold_coverage": False,
        "promotion_eligible": False,
        "evaluation_scope": "development_oof_subset",
        "sources": {
            "baseline_folds": [{
                "fold": 0,
                "studies": 70,
                "sha256": "b" * 64,
                "audit_summary_sha256": "a" * 64,
            }],
            "candidate_folds": [{
                "fold": 0,
                "studies": 70,
                "sha256": "c" * 64,
                "audit_summary_sha256": "d" * 64,
            }],
        },
        "contexts": {
            "frozen_champion": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": {"macro_f1": 0.02, "urgent_f1": 0.04},
            },
            "oracle": {
                "delta": {"macro_f1": 0.01, "urgent_f1": 0.02},
            },
        },
        "threshold_metrics": {
            "baseline": {"f1_3mm": 0.78, "f1_5mm": 0.81},
            "candidate": {"f1_3mm": 0.79, "f1_5mm": 0.82},
        },
    }


class A1Fold0ResourceScreenTests(unittest.TestCase):
    def _run(self, summary: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate_summary.json"
            prereg = root / "prereg.json"
            output = root / "decision.json"
            aggregate.write_text(json.dumps(summary), encoding="utf-8")
            prereg.write_text(json.dumps(_preregistration()), encoding="utf-8")
            result = evaluate(aggregate, prereg, output)
            self.assertTrue(output.is_file())
            self.assertFalse(result["promotion_eligible"])
            self.assertFalse(result["submission_zip_allowed"])
            return result

    def test_passes_only_for_threefold_expansion(self) -> None:
        result = self._run(_summary())
        self.assertEqual(result["status"], "passed_for_threefold_expansion")
        self.assertTrue(result["can_expand_to_folds_1_2"])
        self.assertEqual(result["failed_gates"], [])

    def test_strict_urgent_improvement_rejects_tie(self) -> None:
        summary = _summary()
        summary["contexts"]["frozen_champion"]["delta"]["urgent_f1"] = 0.0
        result = self._run(summary)
        self.assertEqual(result["status"], "rejected_stop_a1_expansion")
        self.assertIn("frozen_context_urgent_f1_strictly_improved", result["failed_gates"])

    def test_refuses_wrong_control_provenance(self) -> None:
        summary = _summary()
        summary["sources"]["baseline_folds"][0]["sha256"] = "e" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate_summary.json"
            prereg = root / "prereg.json"
            aggregate.write_text(json.dumps(summary), encoding="utf-8")
            prereg.write_text(json.dumps(_preregistration()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction hash"):
                evaluate(aggregate, prereg, root / "decision.json")

    def test_refuses_promotion_claim_from_single_fold(self) -> None:
        summary = _summary()
        summary["promotion_eligible"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate_summary.json"
            prereg = root / "prereg.json"
            aggregate.write_text(json.dumps(summary), encoding="utf-8")
            prereg.write_text(json.dumps(_preregistration()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot claim"):
                evaluate(aggregate, prereg, root / "decision.json")


if __name__ == "__main__":
    unittest.main()
