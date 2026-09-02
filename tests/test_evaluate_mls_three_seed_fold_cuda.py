"""CPU-only contract tests for the pure parts of the three-seed CUDA audit."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_mls_three_seed_fold_cuda import (
    IMMUTABLE_FOLDS,
    _config_difference,
    _metrics,
    _parse_checkpoint,
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


if __name__ == "__main__":
    unittest.main()
