from __future__ import annotations

import copy
import unittest

from scripts.select_ich_channel_sources import select_channel_sources
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _summary() -> dict[str, object]:
    return {
        "any_ich_study_auc": 0.8,
        "subtypes": {
            label: {
                "dice_known_pixels": 0.5,
                "study_auc": 0.8,
                "mae_ml": 1.0,
            }
            for label in OUTPUT_LABELS[1:]
        },
    }


class ICHChannelSelectionTests(unittest.TestCase):
    def test_candidate_requires_noninferiority_and_a_strict_gain(self):
        baseline = _summary()
        candidate = copy.deepcopy(baseline)
        candidate["subtypes"]["IPH"]["dice_known_pixels"] = 0.6
        candidate["subtypes"]["SDH"]["study_auc"] = 0.79
        selection = select_channel_sources(baseline, candidate)
        self.assertIn("IPH", selection["candidate_labels"])
        self.assertIn("SDH", selection["reference_labels"])
        self.assertIn("IVH", selection["reference_labels"])

    def test_any_score_changes_only_for_strict_auc_gain(self):
        baseline = _summary()
        tied = copy.deepcopy(baseline)
        self.assertEqual(
            select_channel_sources(baseline, tied)["any_ich_source"],
            "reference",
        )
        improved = copy.deepcopy(baseline)
        improved["any_ich_study_auc"] = 0.81
        self.assertEqual(
            select_channel_sources(baseline, improved)["any_ich_source"],
            "candidate",
        )

    def test_tolerance_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "tolerance"):
            select_channel_sources(_summary(), _summary(), tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
