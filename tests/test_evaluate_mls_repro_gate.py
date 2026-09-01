from __future__ import annotations

import unittest

from scripts.evaluate_mls_repro_gate import evaluate_repro_gate


def _row(epoch: int, mae: float, boundary_f1: float, auc: float) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "study_mls_mae_mm": mae,
        "study_boundary_f1": boundary_f1,
        "selector_auc": auc,
    }


class MLSReproGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"fold": 2, "heatmap_sigma": 3.0}
        self.reference = [_row(15, 1.4756, 0.9218, 0.9112)]

    def test_passes_completed_exact_config_within_guards(self) -> None:
        result = evaluate_repro_gate(
            self.reference,
            [_row(15, 1.60, 0.90, 0.90)],
            {"status": "completed", "config": self.config, "mlflow_run_id": "run-1"},
            self.config,
        )
        self.assertTrue(result["passed"])

    def test_rejects_metric_drift(self) -> None:
        result = evaluate_repro_gate(
            self.reference,
            [_row(15, 1.90, 0.90, 0.90)],
            {"status": "completed", "config": self.config, "mlflow_run_id": "run-1"},
            self.config,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["study_mls_mae_mm"]["passed"])

    def test_rejects_nonterminal_or_config_drift(self) -> None:
        result = evaluate_repro_gate(
            self.reference,
            [_row(15, 1.50, 0.92, 0.91)],
            {"status": "running", "config": {"fold": 1}, "mlflow_run_id": None},
            self.config,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["terminal_completed"]["passed"])
        self.assertFalse(result["checks"]["manifest_config_exact"]["passed"])
        self.assertFalse(result["checks"]["mlflow_run_identified"]["passed"])

    def test_requires_exactly_one_fixed_epoch(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_repro_gate(
                self.reference,
                [_row(14, 1.50, 0.92, 0.91)],
                {"status": "completed", "config": self.config, "mlflow_run_id": "run-1"},
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
