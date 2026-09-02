from __future__ import annotations

import unittest
from pathlib import Path


class ThreeSeedAuditLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).resolve().parents[1]
            / "scripts/run_vast_mls_three_seed_audit.sh"
        ).read_text(encoding="utf-8")

    def test_uses_same_campaign_wide_gpu_lock_as_training(self) -> None:
        self.assertIn('global_gpu_lock="$campaign_root/gpu_training.lock"', self.source)
        self.assertIn('mkdir "$global_gpu_lock"', self.source)
        self.assertIn('rmdir "$global_gpu_lock"', self.source)

    def test_refuses_concurrency_instead_of_waiting(self) -> None:
        self.assertIn("exit 73", self.source)
        self.assertNotIn("sleep ", self.source)

    def test_forwards_arguments_to_cuda_evaluator(self) -> None:
        self.assertIn("evaluate_mls_three_seed_fold_cuda.py", self.source)
        self.assertIn('"$@"', self.source)

    def test_uses_clean_worktree_override_and_cuda_library_path(self) -> None:
        self.assertIn("IAAA_PROJECT_ROOT", self.source)
        self.assertIn("LD_LIBRARY_PATH", self.source)


if __name__ == "__main__":
    unittest.main()
