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
    AUDIT_BATCH_SIZE,
    _raw_dicom_binding,
)
from scripts.validate_mls_r1_replication_matrix import CHECKPOINT_PROVENANCE_SOURCE_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_vast_mls_r1r_three_seed_development_gate.sh"
TRIAGE = PROJECT_ROOT / "scripts" / "evaluate_mls_r1r_fold1_development_triage.py"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_mls_r1r2_replica.py"


class R1R2ContractHardeningTests(unittest.TestCase):
    def test_audit_surface_binds_config_fold_reader_and_evaluators(self) -> None:
        expected = {
            "three_seed_cuda_evaluator",
            "replica_launch_wrapper",
            "mlflow_environment_wrapper",
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

    def test_checkpoint_provenance_has_one_explicit_historical_exception(self) -> None:
        self.assertNotIn("fold_manifest", CHECKPOINT_PROVENANCE_SOURCE_KEYS)
        self.assertIn("dataset", CHECKPOINT_PROVENANCE_SOURCE_KEYS)
        self.assertIn("train_multitask", CHECKPOINT_PROVENANCE_SOURCE_KEYS)

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
        self.assertIn('audit_batch_size="$(read_contract_path \'.protocol.cuda_audit_batch_size\')"', source)
        self.assertIn('IAAA_CONFIG_PATH="$project_config"', source)

    def test_triage_publishes_only_after_staging_validation(self) -> None:
        source = TRIAGE.read_text(encoding="utf-8")
        self.assertIn("staging_dir", source)
        self.assertIn("os.replace(staging_dir, output_dir)", source)
        self.assertIn("IAAA_CONFIG_PATH to equal its sealed project configuration", source)
        self.assertIn('"schema_version": int(summary["schema_version"]) == 1', source)

    def test_replica_launcher_validates_contract_before_training(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("validate_contract(", source)
        self.assertIn("require_checkpoints=False", source)
        self.assertIn("contract-sha256-file", source)
        self.assertIn("IAAA_PROJECT_ROOT must equal", source)
        self.assertIn("IAAA_CONFIG_PATH must equal", source)
        self.assertIn("validated_before_cuda_training", source)

    def test_supervisor_configs_use_the_fail_closed_launcher(self) -> None:
        for path in sorted((PROJECT_ROOT / "config" / "supervisor").glob("mls_r1r2_*.conf")):
            source = path.read_text(encoding="utf-8")
            self.assertIn("launch_mls_r1r2_replica.py", source)
            self.assertIn("--contract-sha256-file", source)
            self.assertIn('IAAA_PROJECT_ROOT="/workspace/IAAA_BrainCtTriage_mls_r1r2"', source)
            self.assertIn('IAAA_CONFIG_PATH="/workspace/IAAA_BrainCtTriage_mls_r1r2/config/project.yaml"', source)

    def test_protocol_binds_the_audit_batch_size(self) -> None:
        self.assertEqual(AUDIT_BATCH_SIZE, 8)


if __name__ == "__main__":
    unittest.main()
