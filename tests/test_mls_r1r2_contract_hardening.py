"""CPU-light static checks for the R1R2 contract hardening surface.

These tests intentionally never instantiate a model or decode DICOM.  They
guard provenance/runner properties that must fail before CUDA work begins.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.materialize_mls_r1_replication_matrix import (
    AUDIT_SOURCE_RELATIVE_PATHS,
    _raw_dicom_binding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_vast_mls_r1r_three_seed_development_gate.sh"
TRIAGE = PROJECT_ROOT / "scripts" / "evaluate_mls_r1r_fold1_development_triage.py"


class R1R2ContractHardeningTests(unittest.TestCase):
    def test_audit_surface_binds_config_fold_reader_and_evaluators(self) -> None:
        expected = {
            "three_seed_cuda_evaluator",
            "development_triage_evaluator",
            "development_gate_runner",
            "canonical_triage_evaluator",
            "config_loader",
            "project_config",
            "fold_split_loader",
            "fold_validator",
            "triage_rules",
            "dicom_reader",
        }
        self.assertTrue(expected.issubset(AUDIT_SOURCE_RELATIVE_PATHS))

    def test_raw_binding_requires_prior_fingerprint_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binding = _raw_dicom_binding(root, {
                "raw_fingerprints_verified": True,
                "raw_dicom_bytes": 123,
            })
            self.assertEqual(binding["resolved_root"], str(root.resolve()))
            with self.assertRaisesRegex(ValueError, "does not verify"):
                _raw_dicom_binding(root, {"raw_fingerprints_verified": False, "raw_dicom_bytes": 123})

    def test_runner_preserves_failure_exit_and_uses_shared_lock(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('gpu_training.lock', source)
        self.assertNotIn('gpu_audit.lock', source)
        self.assertIn("trap 'exit 130' INT", source)
        self.assertIn("trap 'exit 143' TERM", source)
        self.assertLess(
            source.index('write_status "completed" 0'),
            source.index('terminal_status_written=1', source.index('write_status "completed" 0')),
        )
        self.assertIn('--fold-manifest "$fold_manifest"', source)
        self.assertIn('IAAA_CONFIG_PATH="$project_config"', source)

    def test_triage_publishes_only_after_staging_validation(self) -> None:
        source = TRIAGE.read_text(encoding="utf-8")
        self.assertIn("staging_dir", source)
        self.assertIn("os.replace(staging_dir, output_dir)", source)
        self.assertIn("IAAA_CONFIG_PATH to equal its sealed project configuration", source)
        self.assertIn('"schema_version": int(summary["schema_version"]) == 1', source)


if __name__ == "__main__":
    unittest.main()
