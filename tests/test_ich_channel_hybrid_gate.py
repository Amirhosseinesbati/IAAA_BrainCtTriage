from __future__ import annotations

import copy
import unittest

from scripts.evaluate_ich_channel_hybrid_gate import compare_channel_hybrid
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _summary() -> dict[str, object]:
    subtype = {
        "dice_known_pixels": 0.5,
        "study_auc": 0.8,
        "mae_ml": 1.0,
        "volume_strata": {
            "small_le_2ml": {
                "positive_studies": 1,
                "dice_known_pixels": 0.2,
                "presence_sensitivity_at_0_1ml": 1.0,
                "mae_ml": 0.4,
            }
        },
    }
    return {
        "selection_score": 0.6,
        "mean_foreground_dice": 0.5,
        "any_ich_study_auc": 0.8,
        "macro_subtype_study_auc": 0.8,
        "presence_f1_at_0_1ml": 0.8,
        "normal_false_positive_rate_at_0_1ml": 0.2,
        "total_volume_mae_ml": 2.0,
        "subtypes": {
            label: copy.deepcopy(subtype) for label in OUTPUT_LABELS[1:]
        },
    }


def _hybrid_payload(summary: dict[str, object]) -> dict[str, object]:
    return {
        "reference_labels": ["IVH", "SAH"],
        "candidate_labels": ["IPH", "SDH", "EDH"],
        "summary": summary,
    }


class ICHChannelHybridGateTests(unittest.TestCase):
    def test_material_safe_gain_advances(self):
        baseline = _summary()
        candidate = copy.deepcopy(baseline)
        candidate["selection_score"] = 0.62
        candidate["mean_foreground_dice"] = 0.52
        candidate["subtypes"]["IPH"]["dice_known_pixels"] = 0.6
        result = compare_channel_hybrid(baseline, _hybrid_payload(candidate))
        self.assertTrue(result["all_gates_passed"])
        self.assertEqual(
            result["decision"], "advance_to_cross_fitted_five_fold_oof"
        )

    def test_hidden_candidate_subtype_regression_rejects(self):
        baseline = _summary()
        candidate = copy.deepcopy(baseline)
        candidate["selection_score"] = 0.62
        candidate["subtypes"]["SDH"]["study_auc"] = 0.7
        result = compare_channel_hybrid(baseline, _hybrid_payload(candidate))
        self.assertFalse(result["all_gates_passed"])
        self.assertFalse(
            result["subtype_gates"]["SDH"]["study_auc_not_worse"]["passed"]
        )

    def test_reference_channel_drift_rejects(self):
        baseline = _summary()
        candidate = copy.deepcopy(baseline)
        candidate["selection_score"] = 0.62
        candidate["subtypes"]["IVH"]["volume_strata"]["small_le_2ml"][
            "mae_ml"
        ] = 0.5
        result = compare_channel_hybrid(baseline, _hybrid_payload(candidate))
        self.assertFalse(result["all_gates_passed"])
        self.assertFalse(
            result["reference_channel_exact_preservation"]["IVH"][
                "small_le_2ml.mae_ml"
            ]
        )


if __name__ == "__main__":
    unittest.main()
