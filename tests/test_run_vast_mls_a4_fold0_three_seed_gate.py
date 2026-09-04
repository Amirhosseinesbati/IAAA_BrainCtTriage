from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_vast_mls_a4_fold0_three_seed_gate.sh"
PREREGISTRATION = (
    PROJECT_ROOT
    / "reports"
    / "mls_experiments"
    / "mls-deploy-aligned-upgrade-20260902"
    / "FOLD0_A4_THREE_SEED_AUDIT_PREREGISTRATION.json"
)


class A4ThreeSeedGateTests(unittest.TestCase):
    def test_preregistration_locks_the_exact_audit_inputs(self) -> None:
        payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "locked_before_any_a4_fold0_three_seed_outcome")
        self.assertEqual(payload["fold"], 0)
        self.assertEqual(payload["expected_studies"], 70)
        self.assertEqual(payload["fixed_epoch"], 15)
        self.assertEqual(payload["seeds"], [42, 2026, 3407])
        self.assertFalse(payload["adaptive_checkpoint_selection_allowed"])
        self.assertFalse(payload["adaptive_pooling_or_threshold_selection_allowed"])
        self.assertFalse(payload["submission_zip_allowed"])

    def test_runner_requires_passed_resource_and_three_terminal_replicas(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("passed_for_two_remaining_fold0_seed_replications", source)
        self.assertIn("can_start_only_seeds_2026_and_3407_on_fold0", source)
        self.assertIn('resource.get("failed_gates") == []', source)
        self.assertIn('item.get("state") == "completed" and item.get("exit_code") == 0', source)
        self.assertIn('item.get("run_name") for item in statuses', source)
        self.assertIn('[item.get("seed") for item in statuses[1:]] == [2026, 3407]', source)

    def test_runner_uses_only_fixed_epoch_checkpoints_and_deploy_comparison(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("mls_multitask_epoch_015.pth", source)
        self.assertIn("--fold 0 --fixed-epoch 15 --expected-studies 70 --batch-size 6", source)
        self.assertIn("evaluate_mls_deploy_aligned_seed_medians.py", source)
        self.assertIn("evaluate_mls_a4_fold0_triage_screen.py", source)
        self.assertIn("--output-dir \"$triage_root\"", source)
        self.assertIn('"truth": hashlib.sha256(truth_path.read_bytes()).hexdigest()', source)
        self.assertIn('"fold_manifest": hashlib.sha256(fold_manifest.read_bytes()).hexdigest()', source)

    def test_runner_preserves_private_predictions_and_refuses_overwrite_or_gpu_overlap(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("candidate_private_output_upload_allowed", PREREGISTRATION.read_text(encoding="utf-8"))
        self.assertIn('[[ -e "$candidate_audit_root" || -e "$triage_root" || -e "$triage_decision" ]]', source)
        self.assertIn('[[ -e "$global_gpu_lock" ]]', source)
        self.assertIn("nvidia-smi --query-compute-apps=pid", source)
        self.assertNotIn("scp", source)
        self.assertNotIn("mlflow", source.lower())

    def test_runner_writes_terminal_status_without_reading_logs(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('gate_status="$campaign_root/a4_fold0_three_seed_gate_status.json"', source)
        self.assertIn('write_status "preflight" "null"', source)
        self.assertIn('write_status "running" "null"', source)
        self.assertIn('write_status "completed" 0', source)
        self.assertIn('write_status "failed" "$exit_code"', source)
        self.assertNotIn("train.log", source)


if __name__ == "__main__":
    unittest.main()
