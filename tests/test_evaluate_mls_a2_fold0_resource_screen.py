from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_mls_a2_fold0_resource_screen import evaluate


def _audit(checkpoint: Path) -> dict:
    return {
        "state": "completed",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "fold": 0,
        "expected_studies": 70,
        "candidates": {
            "epoch015": {
                "state": "completed",
                "exit_code": 0,
                "checkpoint": str(checkpoint.resolve()),
            }
        },
    }


def _metrics(checkpoint: Path, *, mae: float = 1.4, f1_3: float = 0.83, f1_5: float = 0.79) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "fold": 0,
        "n_studies": 70,
        "failures": 0,
        "fixed_profile_pre_registered": {
            "selector_threshold": 0.5,
            "top_k": 3,
            "aggregation": "p90",
            "mae_mm": mae,
            "f1_3mm": f1_3,
            "f1_5mm": f1_5,
        },
    }


class A2Fold0ResourceScreenTests(unittest.TestCase):
    def _run(self, *, mae: float = 1.4, f1_3: float = 0.83, f1_5: float = 0.79) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch015.pth"
            checkpoint.write_bytes(b"checkpoint")
            audit_path = root / "audit.json"
            metrics_path = root / "metrics.json"
            output = root / "decision.json"
            audit_path.write_text(json.dumps(_audit(checkpoint)), encoding="utf-8")
            metrics_path.write_text(
                json.dumps(_metrics(checkpoint, mae=mae, f1_3=f1_3, f1_5=f1_5)),
                encoding="utf-8",
            )
            result = evaluate(audit_path, metrics_path, checkpoint, output)
            self.assertTrue(output.is_file())
            self.assertFalse(result["promotion_eligible"])
            self.assertFalse(result["submission_zip_allowed"])
            return result

    def test_passing_screen_authorizes_only_the_two_fixed_replications(self) -> None:
        result = self._run()
        self.assertEqual(result["status"], "passed_for_two_remaining_fold0_seed_replications")
        self.assertTrue(result["can_start_only_seeds_2026_and_3407_on_fold0"])

    def test_noninferior_but_not_improved_mae_rejects(self) -> None:
        result = self._run(mae=1.5)
        self.assertEqual(result["status"], "rejected_stop_a2_expansion")
        self.assertIn("mae_mm_lte", result["failed_gates"])

    def test_refuses_multiple_or_wrong_checkpoint_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch015.pth"
            checkpoint.write_bytes(b"checkpoint")
            audit = _audit(checkpoint)
            audit["candidates"]["best"] = dict(audit["candidates"]["epoch015"])
            audit_path = root / "audit.json"
            metrics_path = root / "metrics.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            metrics_path.write_text(json.dumps(_metrics(checkpoint)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly epoch015"):
                evaluate(audit_path, metrics_path, checkpoint, root / "decision.json")


if __name__ == "__main__":
    unittest.main()
