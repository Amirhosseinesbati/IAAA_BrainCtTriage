from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_ich_channel_hybrid import (
    build_channel_hybrid,
    evaluate_channel_hybrid,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _predictions(offset: float) -> pd.DataFrame:
    row: dict[str, object] = {
        "study_id": "study-1",
        "patient_id": "patient-1",
        "slice_index": 0,
        "known": 1,
        "voxel_volume_ml": 0.002,
        "prob_any_ich": offset + 0.01,
    }
    for index, label in enumerate(OUTPUT_LABELS[1:], start=1):
        for prefix_index, prefix in enumerate((
            "prob",
            "pred_pixels",
            "intersection",
            "predicted_known_pixels",
            "observed_known_pixels",
        )):
            row[f"{prefix}_{label}"] = offset + index * 10 + prefix_index
    return pd.DataFrame([row])


class ICHChannelHybridTests(unittest.TestCase):
    def test_reference_ivh_and_candidate_other_channels_are_exact(self):
        reference = _predictions(100.0)
        candidate = _predictions(200.0)
        hybrid = build_channel_hybrid(reference, candidate)
        self.assertEqual(hybrid.loc[0, "prob_any_ich"], 200.01)
        self.assertEqual(hybrid.loc[0, "prob_IVH"], reference.loc[0, "prob_IVH"])
        self.assertEqual(hybrid.loc[0, "intersection_IVH"], reference.loc[0, "intersection_IVH"])
        self.assertEqual(hybrid.loc[0, "prob_IPH"], candidate.loc[0, "prob_IPH"])
        self.assertEqual(hybrid.loc[0, "pred_pixels_SDH"], candidate.loc[0, "pred_pixels_SDH"])

    def test_identity_mismatch_is_rejected(self):
        reference = _predictions(100.0)
        candidate = _predictions(200.0)
        candidate.loc[0, "slice_index"] = 1
        with self.assertRaisesRegex(ValueError, "slice_index"):
            build_channel_hybrid(reference, candidate)

    def test_unknown_channel_and_any_source_are_rejected(self):
        reference = _predictions(100.0)
        candidate = _predictions(200.0)
        with self.assertRaisesRegex(ValueError, "Unknown reference labels"):
            build_channel_hybrid(reference, candidate, reference_labels=("BAD",))
        with self.assertRaisesRegex(ValueError, "any_source"):
            build_channel_hybrid(reference, candidate, any_source="bad")

    def test_outer_hybrid_requires_fold_provenance(self):
        with self.assertRaisesRegex(ValueError, "outer_fold is required"):
            evaluate_channel_hybrid(
                Path("reference.csv"),
                Path("candidate.csv"),
                reference_run_summary=Path("reference.json"),
                candidate_run_summary=Path("candidate.json"),
                output_dir=Path("output"),
                evaluation_split="outer_fold",
            )


if __name__ == "__main__":
    unittest.main()
