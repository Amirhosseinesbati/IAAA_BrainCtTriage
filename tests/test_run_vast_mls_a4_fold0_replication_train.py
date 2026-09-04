from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_vast_mls_a4_fold0_replication_train.sh"
PREREGISTRATION = (
    PROJECT_ROOT
    / "reports"
    / "mls_experiments"
    / "mls-deploy-aligned-upgrade-20260902"
    / "FOLD0_A4_SEED_REPLICATION_PREREGISTRATION.json"
)


class A4ReplicationLauncherTests(unittest.TestCase):
    def test_preregistration_locks_only_the_remaining_fold0_seeds(self) -> None:
        payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "locked_before_a4_seed42_resource_outcome")
        self.assertEqual(payload["fold"], 0)
        self.assertEqual(payload["studies"], 70)
        self.assertEqual(payload["fixed_epoch"], 15)
        self.assertEqual(payload["allowed_replication_seeds"], [2026, 3407])
        self.assertEqual(
            payload["only_allowed_manifest_changes"], ["run_name", "training_config.seed"]
        )
        self.assertFalse(payload["submission_zip_allowed"])

    def test_launcher_is_seed_limited_and_resource_gated(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"$1" != "2026" && "$1" != "3407"', source)
        self.assertIn("passed_for_two_remaining_fold0_seed_replications", source)
        self.assertIn("can_start_only_seeds_2026_and_3407_on_fold0", source)
        self.assertIn("A4 replication contract refused", source)
        self.assertIn('decision.get("failed_gates") == []', source)

    def test_launcher_allows_only_seed_and_run_name_manifest_edits(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"only_allowed_manifest_changes") == ["run_name", "training_config.seed"]', source)
        self.assertIn('template.count(f"run_name: {expected_name}") != 1', source)
        self.assertIn('template.count(expected_seed) != 1', source)
        self.assertIn('f"run_name: {run_name}"', source)
        self.assertIn('f"  seed: {seed}\\n"', source)

    def test_launcher_refuses_overwrite_and_concurrent_gpu_work(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('[[ -e "$artifact_root" || -e "$checkpoint_dir" ]]', source)
        self.assertIn('mkdir "$global_gpu_lock"', source)
        self.assertIn("nvidia-smi --query-compute-apps=pid", source)
        self.assertIn('"compute_policy":"cuda_only_no_cpu_fallback"', source)


if __name__ == "__main__":
    unittest.main()
