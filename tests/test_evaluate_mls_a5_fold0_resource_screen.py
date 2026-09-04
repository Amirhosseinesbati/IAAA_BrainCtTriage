from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluate_mls_a5_fold0_resource_screen import (
    _atomic_json,
    _log_aggregate_to_mlflow,
    evaluate,
)


def _audit(checkpoint: Path) -> dict:
    return {
        "state": "completed",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "fold": 0,
        "expected_studies": 70,
        "candidates": {"epoch015": {
            "state": "completed", "exit_code": 0, "checkpoint": str(checkpoint.resolve()),
        }},
    }


def _metrics(checkpoint: Path, *, mae: float = 1.4) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()), "fold": 0, "n_studies": 70, "failures": 0,
        "fixed_profile_pre_registered": {
            "selector_threshold": 0.5, "top_k": 3, "aggregation": "p90",
            "mae_mm": mae, "f1_3mm": 0.83, "f1_5mm": 0.79,
        },
    }


class A5Fold0ResourceScreenTests(unittest.TestCase):
    def test_launchers_are_a5_specific_and_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        train = (root / "scripts" / "run_vast_mls_a5_fold0_seed42_train.sh").read_text(
            encoding="utf-8",
        )
        resource = (
            root / "scripts" / "run_vast_mls_a5_fold0_seed42_resource_screen.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('run_name="mls-vast-da-a5-detached-rank-fold0-seed42"', train)
        self.assertIn('Refusing to overwrite existing A5 terminal or running status', train)
        self.assertIn('scripts/evaluate_mls_a5_fold0_resource_screen.py', resource)
        self.assertIn('.state == "completed" and .exit_code == 0', resource)
        self.assertIn('Refusing to overwrite an existing A5 audit or decision', resource)

    def _run(self, *, mae: float = 1.4) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch015.pth"
            checkpoint.write_bytes(b"checkpoint")
            audit = root / "audit.json"
            metrics = root / "metrics.json"
            decision = root / "decision.json"
            audit.write_text(json.dumps(_audit(checkpoint)), encoding="utf-8")
            metrics.write_text(json.dumps(_metrics(checkpoint, mae=mae)), encoding="utf-8")
            with mock.patch(
                "scripts.evaluate_mls_a2_fold0_resource_screen._atomic_json",
                side_effect=AssertionError("An intermediate A2 decision must never be published"),
            ), mock.patch(
                "scripts.evaluate_mls_a5_fold0_resource_screen._atomic_json",
                wraps=_atomic_json,
            ) as publish:
                result = evaluate(audit, metrics, checkpoint, decision)
            publish.assert_called_once()
            self.assertEqual(json.loads(decision.read_text(encoding="utf-8")), result)
            self.assertFalse(result["can_start_only_seeds_2026_and_3407_on_fold0"])
            return result

    def test_pass_is_not_promotion_or_submission(self) -> None:
        result = self._run()
        self.assertEqual(result["candidate"], "mls-vast-deploy-aligned-a5-detached-rank")
        self.assertEqual(result["screen_scope"], "a5_fold0_seed42_resource_screen_only")
        self.assertEqual(result["status"], "passed_for_manual_a5_replication_preregistration")
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["submission_zip_allowed"])

    def test_failure_stops_a5_expansion(self) -> None:
        result = self._run(mae=1.5)
        self.assertEqual(result["status"], "rejected_stop_a5_expansion")

    def test_mlflow_is_aggregate_only_and_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = Path(directory) / "decision.json"
            decision.write_text("{}", encoding="utf-8")
            result = {
                "observed": {"mae_mm": 1.4, "f1_3mm": 0.83, "f1_5mm": 0.79, "boundary_f1": 0.81, "selection_objective": 1.78},
                "gate_results": {"mae_mm_lte": True, "f1_3mm_gte": True, "f1_5mm_gte": True, "boundary_f1_gte": True, "selection_objective_lte": True},
                "failed_gates": [], "status": "passed_for_manual_a5_replication_preregistration", "checkpoint_sha256": "a" * 64,
            }
            with mock.patch("src.mlops.tracking.configure_tracking_environment"), mock.patch("mlflow.tracking.MlflowClient") as client_class:
                logged = _log_aggregate_to_mlflow("b" * 32, result, decision)
            self.assertEqual(logged, {"status": "logged", "run_id": "b" * 32})
            names = [call.args[1] for call in client_class.return_value.log_metric.call_args_list]
            self.assertEqual(len(names), 11)
            self.assertTrue(all(name.startswith("a5_resource_") for name in names))
            self.assertTrue(all("prediction" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
