from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.log_mls_deploy_aligned_meta_evaluation import (
    _load_and_validate,
    _metric_payload,
    _source_hashes,
)


def _classification(macro: float, urgent: float) -> dict:
    return {
        "macro_f1": macro,
        "accuracy": 0.8,
        "per_class": {
            "Normal": {"f1": 0.9},
            "Urgent": {"f1": urgent},
            "Critical": {"f1": 0.7},
        },
    }


def _context() -> dict:
    return {
        "baseline": _classification(0.70, 0.60),
        "candidate": _classification(0.74, 0.66),
        "delta": {"macro_f1": 0.04, "accuracy": 0.0, "urgent_f1": 0.06},
        "paired_patient_bootstrap": {
            "probability_of_improvement": 0.97,
            "ci95_low": 0.01,
            "ci95_high": 0.08,
        },
        "changed_triage_decisions": 4,
    }


def _payload() -> dict:
    source = {"fold": 0, "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "evaluation_scope": "development_oof_subset",
        "development_gate_passed": True,
        "promotion_eligible": False,
        "selected_folds": [0],
        "available_folds": [0, 1, 2, 3, 4],
        "full_fold_coverage": False,
        "sources": {
            "baseline_folds": [source],
            "candidate_folds": [source],
            "frozen_champion_predictions": {"sha256": "b" * 64},
            "truth_table": {"sha256": "c" * 64},
            "fold_manifest": {"sha256": "d" * 64},
        },
        "studies": 70,
        "threshold_metrics": {
            branch: {f"f1_{threshold}mm": 0.5 for threshold in (1, 3, 5)}
            for branch in ("baseline", "candidate", "delta")
        },
        "contexts": {"frozen_champion": _context(), "oracle": _context()},
        "promotion_gates": {"macro_f1_improved": True, "full_immutable_fold_coverage": False},
        "failed_hard_gates": ["full_immutable_fold_coverage"],
    }


class LogMlsDeployAlignedMetaEvaluationTest(unittest.TestCase):
    def test_extracts_primary_metrics_and_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate_summary.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            payload = _load_and_validate(path)
        metrics = _metric_payload(payload)
        hashes = _source_hashes(payload)
        self.assertEqual(metrics["frozen_candidate_macro_f1"], 0.74)
        self.assertEqual(metrics["frozen_delta_urgent_f1"], 0.06)
        self.assertEqual(metrics["frozen_bootstrap_probability_improvement"], 0.97)
        self.assertEqual(hashes["baseline_folds_fold0_sha256"], "a" * 64)

    def test_rejects_nonaggregate_and_private_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            for name in (
                "per_study_private.csv",
                "study_member_predictions_private.csv",
                "metrics.json",
            ):
                path = Path(directory) / name
                path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Only 'aggregate_summary.json'"):
                    _load_and_validate(path)

    def test_promotion_requires_complete_338_study_oof(self) -> None:
        payload = _payload()
        payload["promotion_eligible"] = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate_summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full fold coverage"):
                _load_and_validate(path)


if __name__ == "__main__":
    unittest.main()
