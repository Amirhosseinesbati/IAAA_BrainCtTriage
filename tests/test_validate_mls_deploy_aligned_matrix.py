from __future__ import annotations

import unittest
from pathlib import Path

from scripts.validate_mls_deploy_aligned_matrix import validate_matrix


ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROOT = ROOT / "config/experiments/generated/mls-deploy-aligned-20260902"


class DeployAlignedMatrixTests(unittest.TestCase):
    def test_baseline_is_exact_locked_cartesian_matrix(self) -> None:
        result = validate_matrix(MATRIX_ROOT / "baseline_matrix_manifest.json")
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["runs"], 9)
        self.assertEqual(result["execution_required"], 6)
        self.assertEqual(result["reused_fixed_epoch15"], 3)
        self.assertEqual(result["within_stage_config_differences"], ["fold", "seed"])

    def test_a1_is_exact_locked_cartesian_matrix(self) -> None:
        result = validate_matrix(MATRIX_ROOT / "a1_ordinal_matrix_manifest.json")
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["runs"], 9)
        self.assertEqual(result["execution_required"], 9)
        self.assertEqual(result["reused_fixed_epoch15"], 0)


if __name__ == "__main__":
    unittest.main()
