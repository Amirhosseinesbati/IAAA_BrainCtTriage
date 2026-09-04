from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluate_mls_a4_fold0_resource_screen import (
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


class A4Fold0ResourceScreenTests(unittest.TestCase):
    def test_runner_is_fail_closed_and_uses_only_a4_evaluator(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "run_vast_mls_a4_fold0_seed42_resource_screen.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('run_name="mls-vast-da-a4-pair-rank-fold0-seed42"', runner)
        self.assertIn('scripts/evaluate_mls_a4_fold0_resource_screen.py', runner)
        self.assertIn('write_status "failed_resource_gate_evaluator" "$gate_exit"', runner)
        self.assertIn('write_status "refused_gpu_compute_process_exists" 7', runner)

    def test_training_launcher_writes_a3_compatible_terminal_status(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "run_vast_mls_a4_fold0_seed42_train.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"state":"%s"', launcher)
        self.assertIn('write_status "running" "null"', launcher)
        self.assertIn('write_status "completed" 0', launcher)
        self.assertIn('write_status "failed" "$exit_code"', launcher)
        self.assertIn('source_commit_local="6ddd738244cc8b5d702235e64c88b1c8608a93f3"', launcher)

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
            return evaluate(audit, metrics, checkpoint, decision)

    def test_pass_is_only_seed_replication_authorization(self) -> None:
        result = self._run()
        self.assertEqual(result["candidate"], "mls-vast-deploy-aligned-a4-pair-rank")
        self.assertEqual(result["screen_scope"], "a4_fold0_seed42_resource_screen_only")
        self.assertEqual(result["status"], "passed_for_two_remaining_fold0_seed_replications")
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["submission_zip_allowed"])

    def test_failure_stops_a4_expansion(self) -> None:
        result = self._run(mae=1.5)
        self.assertEqual(result["status"], "rejected_stop_a4_expansion")
        self.assertFalse(result["can_start_only_seeds_2026_and_3407_on_fold0"])

    def test_mlflow_is_aggregate_only_and_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = Path(directory) / "decision.json"
            decision.write_text("{}", encoding="utf-8")
            result = {
                "observed": {"mae_mm": 1.4, "f1_3mm": 0.83, "f1_5mm": 0.79, "boundary_f1": 0.81, "selection_objective": 1.78},
                "gate_results": {"mae_mm_lte": True, "f1_3mm_gte": True, "f1_5mm_gte": True, "boundary_f1_gte": True, "selection_objective_lte": True},
                "failed_gates": [], "status": "passed_for_two_remaining_fold0_seed_replications", "checkpoint_sha256": "a" * 64,
            }
            with mock.patch("src.mlops.tracking.configure_tracking_environment"), mock.patch("mlflow.tracking.MlflowClient") as client_class:
                logged = _log_aggregate_to_mlflow("b" * 32, result, decision)
            self.assertEqual(logged, {"status": "logged", "run_id": "b" * 32})
            names = [call.args[1] for call in client_class.return_value.log_metric.call_args_list]
            self.assertEqual(len(names), 11)
            self.assertTrue(all(name.startswith("a4_resource_") for name in names))
            self.assertTrue(all("prediction" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
