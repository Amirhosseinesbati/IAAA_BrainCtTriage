from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ich_sampler_screen import compare_sampler_runs


def _summary(**updates):
    values = {
        "selection_score": 0.64,
        "mean_foreground_dice": 0.45,
        "any_ich_study_auc": 0.92,
        "macro_subtype_study_auc": 0.86,
        "presence_f1_at_0_1ml": 0.84,
        "normal_false_positive_rate_at_0_1ml": 0.20,
        "total_volume_mae_ml": 9.0,
        "subtypes": {
            "IVH": {
                "dice_known_pixels": 0.60,
                "study_auc": 0.90,
                "mae_ml": 0.50,
                "volume_strata": {
                    "small_le_2ml": {
                        "positive_studies": 2,
                        "dice_known_pixels": 0.10,
                        "presence_sensitivity_at_0_1ml": 0.50,
                        "mae_ml": 0.80,
                    }
                },
            }
        },
    }
    for key, value in updates.items():
        if key.startswith("ivh_"):
            values["subtypes"]["IVH"][key.removeprefix("ivh_")] = value
        elif key.startswith("small_"):
            values["subtypes"]["IVH"]["volume_strata"]["small_le_2ml"][
                key.removeprefix("small_")
            ] = value
        else:
            values[key] = value
    return values


def _run(root: Path, name: str, power: float, summary: dict) -> Path:
    run = root / name
    run.mkdir()
    config = {
        "run_name": name,
        "output_dir": str(run),
        "outer_fold": 2,
        "calibration_fold": 1,
        "sampler_study_balance_power": power,
        "evaluate_outer": power == 0.0,
    }
    (run / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "run_summary.json").write_text(
        json.dumps({"manifest_sha256": "manifest"}), encoding="utf-8"
    )
    (run / "calibration.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


class ICHSamplerScreenTests(unittest.TestCase):
    def test_clean_calibration_screen_advances_without_reading_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", 0.0, _summary())
            candidate = _run(
                root,
                "candidate",
                0.5,
                _summary(
                    selection_score=0.645,
                    total_volume_mae_ml=8.8,
                    ivh_mae_ml=0.45,
                    small_dice_known_pixels=0.18,
                    small_mae_ml=0.70,
                ),
            )
            result = compare_sampler_runs(
                baseline,
                candidate,
                baseline_calibration_summary=baseline / "calibration.json",
                candidate_calibration_summary=candidate / "calibration.json",
            )
            self.assertEqual(
                result["decision"],
                "advance_to_five_fold_oof_without_single_outer_tuning",
            )
            self.assertTrue(result["calibration"]["all_gates_passed"])
            self.assertIsNone(result["outer"])

    def test_ivh_mae_regression_rejects_before_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", 0.0, _summary())
            candidate = _run(
                root,
                "candidate",
                0.75,
                _summary(
                    selection_score=0.65,
                    ivh_mae_ml=0.60,
                    small_dice_known_pixels=0.18,
                    small_mae_ml=0.70,
                ),
            )
            result = compare_sampler_runs(
                baseline,
                candidate,
                baseline_calibration_summary=baseline / "calibration.json",
                candidate_calibration_summary=candidate / "calibration.json",
            )
            self.assertEqual(result["decision"], "reject_before_outer")
            self.assertFalse(
                result["calibration"]["gates"]["ivh_mae_not_worse"]["passed"]
            )

    def test_unrelated_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "baseline", 0.0, _summary())
            candidate = _run(root, "candidate", 0.5, _summary())
            config_path = candidate / "resolved_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["batch_size"] = 99
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected config"):
                compare_sampler_runs(
                    baseline,
                    candidate,
                    baseline_calibration_summary=baseline / "calibration.json",
                    candidate_calibration_summary=candidate / "calibration.json",
                )


if __name__ == "__main__":
    unittest.main()
