from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_mls_a4_fold0_triage_screen import evaluate


BASE_SHA = "a" * 64
CANDIDATE_SHA = "b" * 64


def _source(prediction_sha: str, checkpoint_prefix: str) -> dict:
    return {
        "fold": 0,
        "studies": 70,
        "sha256": prediction_sha,
        "audit_summary_sha256": "b7eee359a1283000da9228ecf565da2b3168219d39ba48607bf933e0fcdb1f27",
        "checkpoint_sha256": {
            "seed42": checkpoint_prefix * 64,
            "seed2026": ("c" if checkpoint_prefix == "d" else "f") * 64,
            "seed3407": ("e" if checkpoint_prefix == "d" else "0") * 64,
        },
    }


def _prereg() -> dict:
    return {
        "status": "locked_before_any_a4_audit_or_triage_outcome",
        "candidate": {"stage": "a4_pair_rank"},
        "fixed_screen_gates": [
            "frozen_context_macro_f1_strictly_improved",
            "frozen_context_urgent_f1_strictly_improved",
            "frozen_context_accuracy_noninferior",
            "normal_f1_not_below_control_minus_0p01",
            "critical_f1_not_below_control_minus_0p01",
            "f1_3mm_noninferior",
            "f1_5mm_noninferior",
            "oracle_macro_and_urgent_directions_nonnegative",
        ],
        "control": {
            "fold": 0,
            "studies": 70,
            "fixed_epoch": 15,
            "seeds": [42, 2026, 3407],
            "aggregation": "median",
            "audit_summary_sha256": "b7eee359a1283000da9228ecf565da2b3168219d39ba48607bf933e0fcdb1f27",
            "private_predictions_sha256": BASE_SHA,
        },
    }


def _summary(macro_delta: float = 0.01, urgent_delta: float = 0.01) -> dict:
    baseline = {"accuracy": 0.9, "per_class": {"Normal": {"f1": 0.9}, "Critical": {"f1": 0.8}}}
    candidate = {"accuracy": 0.9, "per_class": {"Normal": {"f1": 0.9}, "Critical": {"f1": 0.8}}}
    return {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "selected_folds": [0],
        "studies": 70,
        "full_fold_coverage": False,
        "promotion_eligible": False,
        "evaluation_scope": "development_oof_subset",
        "sources": {"baseline_folds": [_source(BASE_SHA, "d")], "candidate_folds": [_source(CANDIDATE_SHA, "1")]},
        "contexts": {
            "frozen_champion": {"baseline": baseline, "candidate": candidate, "delta": {"macro_f1": macro_delta, "urgent_f1": urgent_delta}},
            "oracle": {"delta": {"macro_f1": 0.01, "urgent_f1": 0.01}},
        },
        "threshold_metrics": {"baseline": {"f1_3mm": 0.8, "f1_5mm": 0.8}, "candidate": {"f1_3mm": 0.8, "f1_5mm": 0.8}},
    }


class A4TriageScreenTests(unittest.TestCase):
    def _run(self, summary: dict, prereg: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            prereg_path = root / "prereg.json"
            output = root / "result.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            prereg_path.write_text(json.dumps(prereg or _prereg()), encoding="utf-8")
            return evaluate(summary_path, prereg_path, output)

    def test_pass_only_authorizes_a4_development_expansion(self) -> None:
        result = self._run(_summary())
        self.assertEqual(result["status"], "passed_for_a4_folds_1_2_development_expansion")
        self.assertTrue(result["can_expand_to_folds_1_2"])
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["submission_zip_allowed"])

    def test_macro_or_urgent_decline_rejects_expansion(self) -> None:
        result = self._run(_summary(macro_delta=-0.001))
        self.assertEqual(result["status"], "rejected_stop_a4_cross_fold_expansion")
        self.assertIn("frozen_context_macro_f1_strictly_improved", result["failed_gates"])

    def test_a3_or_unlocked_preregistration_is_refused(self) -> None:
        prereg = _prereg()
        prereg["candidate"]["stage"] = "a3_study_bag"
        with self.assertRaisesRegex(ValueError, "unexpected candidate"):
            self._run(_summary(), prereg)

    def test_private_prediction_hash_substitution_is_refused(self) -> None:
        prereg = _prereg()
        prereg["control"]["private_predictions_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source contract"):
            self._run(_summary(), prereg)


if __name__ == "__main__":
    unittest.main()
