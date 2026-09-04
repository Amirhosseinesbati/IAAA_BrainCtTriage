from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_mls_a3_fold0_triage_screen import EXPECTED_GATES, evaluate


BASE_SHA = "a" * 64
CANDIDATE_SHA = "b" * 64
AUDIT_SHA = "c" * 64


def _source(*, prediction_sha: str, checkpoint_prefix: str) -> dict:
    return {
        "fold": 0,
        "studies": 70,
        "sha256": prediction_sha,
        "audit_summary_sha256": AUDIT_SHA,
        "checkpoint_sha256": {
            "seed42": checkpoint_prefix + "1" * 63,
            "seed2026": checkpoint_prefix + "2" * 63,
            "seed3407": checkpoint_prefix + "3" * 63,
        },
    }


def _metrics(*, macro_delta: float = 0.02, urgent_delta: float = 0.03) -> dict:
    baseline = {
        "accuracy": 0.8,
        "per_class": {"Normal": {"f1": 0.8}, "Critical": {"f1": 0.8}},
    }
    candidate = {
        "accuracy": 0.81,
        "per_class": {"Normal": {"f1": 0.8}, "Critical": {"f1": 0.8}},
    }
    return {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "selected_folds": [0],
        "studies": 70,
        "full_fold_coverage": False,
        "promotion_eligible": False,
        "evaluation_scope": "development_oof_subset",
        "sources": {
            "baseline_folds": [_source(prediction_sha=BASE_SHA, checkpoint_prefix="d")],
            "candidate_folds": [_source(prediction_sha=CANDIDATE_SHA, checkpoint_prefix="e")],
        },
        "contexts": {
            "frozen_champion": {
                "baseline": baseline,
                "candidate": candidate,
                "delta": {"macro_f1": macro_delta, "urgent_f1": urgent_delta},
            },
            "oracle": {"delta": {"macro_f1": 0.01, "urgent_f1": 0.01}},
        },
        "threshold_metrics": {
            "baseline": {"f1_3mm": 0.8, "f1_5mm": 0.8},
            "candidate": {"f1_3mm": 0.8, "f1_5mm": 0.8},
        },
    }


def _prereg() -> dict:
    return {
        "status": "locked_before_any_a3_fold0_three_seed_outcome",
        "candidate": {"stage": "a3_study_bag"},
        "fixed_screen_gates": list(EXPECTED_GATES),
        "control": {
            "fold": 0, "studies": 70, "fixed_epoch": 15,
            "seeds": [42, 2026, 3407], "aggregation": "median",
            "audit_summary_sha256": AUDIT_SHA,
            "private_predictions_sha256": BASE_SHA,
        },
    }


class A3Fold0TriageScreenTests(unittest.TestCase):
    def _run(self, summary: dict, prereg: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            prereg_path = root / "prereg.json"
            output = root / "decision.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            prereg_path.write_text(json.dumps(prereg or _prereg()), encoding="utf-8")
            return evaluate(summary_path, prereg_path, output)

    def test_pass_only_authorizes_two_fold_development_expansion(self) -> None:
        result = self._run(_metrics())
        self.assertEqual(result["status"], "passed_for_a3_folds_1_2_development_expansion")
        self.assertTrue(result["can_expand_to_folds_1_2"])
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["submission_zip_allowed"])

    def test_macro_or_urgent_decline_rejects_cross_fold_expansion(self) -> None:
        result = self._run(_metrics(macro_delta=-0.001))
        self.assertEqual(result["status"], "rejected_stop_a3_cross_fold_expansion")
        self.assertIn("frozen_context_macro_f1_strictly_improved", result["failed_gates"])
        self.assertFalse(result["can_expand_to_folds_1_2"])

    def test_control_hash_substitution_is_refused(self) -> None:
        prereg = _prereg()
        prereg["control"]["private_predictions_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source contract"):
            self._run(_metrics(), prereg)


if __name__ == "__main__":
    unittest.main()
