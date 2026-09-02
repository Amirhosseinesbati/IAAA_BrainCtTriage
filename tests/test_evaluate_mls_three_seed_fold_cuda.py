"""CPU-only contract tests for the pure parts of the three-seed CUDA audit."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.evaluate_mls_three_seed_fold_cuda import (
    IMMUTABLE_FOLDS,
    _config_difference,
    _metrics,
    _parse_checkpoint,
    _require_matching_resume_contract,
    _resume_contract,
)


class ThreeSeedFoldAuditTests(unittest.TestCase):
    def test_full_immutable_five_fold_scope_is_supported(self) -> None:
        self.assertEqual(IMMUTABLE_FOLDS, (0, 1, 2, 3, 4))

    def test_config_difference_accepts_seed_only(self) -> None:
        configs = {
            "seed42": {"fold": 0, "seed": 42, "backbone": "hrnet_w32"},
            "seed2026": {"fold": 0, "seed": 2026, "backbone": "hrnet_w32"},
            "seed3407": {"fold": 0, "seed": 3407, "backbone": "hrnet_w32"},
        }
        self.assertEqual(_config_difference(configs), {"seed"})

    def test_config_difference_detects_hidden_recipe_change(self) -> None:
        configs = {
            "a": {"fold": 0, "seed": 42, "weight": 0.1},
            "b": {"fold": 0, "seed": 2026, "weight": 0.2},
        }
        self.assertEqual(_config_difference(configs), {"seed", "weight"})

    def test_metrics_include_all_official_boundaries(self) -> None:
        truth = np.asarray([0.5, 2.0, 4.0, 6.0])
        metrics = _metrics(truth, truth.copy())
        self.assertEqual(metrics["mae_mm"], 0.0)
        self.assertEqual(metrics["f1_1mm"], 1.0)
        self.assertEqual(metrics["f1_3mm"], 1.0)
        self.assertEqual(metrics["f1_5mm"], 1.0)

    def test_checkpoint_label_rejects_shell_metacharacters(self) -> None:
        with self.assertRaisesRegex(Exception, "unsafe checkpoint label"):
            _parse_checkpoint("bad;label=model.pth")

    def test_resume_contract_binds_predictions_to_checkpoint_hashes(self) -> None:
        manifest = {
            "seed42": {"bytes": 10, "sha256": "a" * 64, "epoch": 15, "seed": 42},
            "seed2026": {"bytes": 11, "sha256": "b" * 64, "epoch": 15, "seed": 2026},
            "seed3407": {"bytes": 12, "sha256": "c" * 64, "epoch": 15, "seed": 3407},
        }
        first = _resume_contract(
            fold=0,
            expected_studies=2,
            fixed_epoch=15,
            checkpoint_manifest=manifest,
            study_ids=["study-a", "study-b"],
        )
        changed = {label: values.copy() for label, values in manifest.items()}
        changed["seed3407"]["sha256"] = "d" * 64
        second = _resume_contract(
            fold=0,
            expected_studies=2,
            fixed_epoch=15,
            checkpoint_manifest=changed,
            study_ids=["study-a", "study-b"],
        )
        self.assertNotEqual(first, second)

    def test_resume_requires_exact_contract(self) -> None:
        expected = {"schema_version": 1, "fold": 0}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume_contract.json"
            with self.assertRaisesRegex(RuntimeError, "without a resume contract"):
                _require_matching_resume_contract(path, expected)
            path.write_text('{"schema_version": 1, "fold": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _require_matching_resume_contract(path, expected)
            path.write_text('{"fold": 0, "schema_version": 1}\n', encoding="utf-8")
            _require_matching_resume_contract(path, expected)


if __name__ == "__main__":
    unittest.main()
