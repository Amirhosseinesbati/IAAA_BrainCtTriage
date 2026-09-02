from __future__ import annotations

import unittest
from pathlib import Path


class DeployAlignedLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).resolve().parents[1]
            / "scripts/run_vast_mls_deploy_aligned_job.sh"
        ).read_text(encoding="utf-8")

    def test_uses_campaign_wide_gpu_lock(self) -> None:
        self.assertIn('global_gpu_lock="$campaign_root/gpu_training.lock"', self.source)
        self.assertIn('mkdir "$global_gpu_lock"', self.source)
        self.assertIn('rmdir "$global_gpu_lock"', self.source)

    def test_allows_clean_worktree_without_mutating_dirty_checkout(self) -> None:
        self.assertIn(
            'project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage}"',
            self.source,
        )

    def test_refuses_other_gpu_compute_process(self) -> None:
        self.assertIn("nvidia-smi --query-compute-apps=pid", self.source)
        self.assertIn('reason":"gpu_compute_process_exists', self.source)

    def test_requires_minimum_free_disk(self) -> None:
        self.assertIn('minimum_free_gib="${IAAA_MLS_MIN_FREE_GIB:-4}"', self.source)
        self.assertIn('reason":"insufficient_disk', self.source)

    def test_preserves_cuda_runtime_guard_and_no_cpu_fallback(self) -> None:
        self.assertIn("export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu", self.source)
        self.assertIn('"compute_policy":"cuda_only_no_cpu_fallback"', self.source)

    def test_trains_from_immutable_manifest_snapshot(self) -> None:
        self.assertIn(
            'manifest_snapshot="$artifact_root/training_manifest_used.yaml"',
            self.source,
        )
        self.assertIn('cp -- "$manifest" "$manifest_snapshot"', self.source)
        self.assertIn('--manifest "$manifest_snapshot"', self.source)
        self.assertIn('sha256sum "$manifest_snapshot"', self.source)


if __name__ == "__main__":
    unittest.main()
