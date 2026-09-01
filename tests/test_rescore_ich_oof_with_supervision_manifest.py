from __future__ import annotations

import unittest

import pandas as pd

from scripts.rescore_ich_oof_with_supervision_manifest import (
    apply_supervision_manifest,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _prediction_rows() -> pd.DataFrame:
    rows = []
    for study_id, known, fold in (("promote", 0, 0), ("unknown", 0, 1),
                                  ("known", 1, 2)):
        row = {
            "study_id": study_id,
            "slice_index": 0,
            "known": known,
            "outer_fold": fold,
        }
        for index, label in enumerate(OUTPUT_LABELS[1:], start=1):
            row[f"pred_pixels_{label}"] = index
            row[f"intersection_{label}"] = index if known else 0
            row[f"predicted_known_pixels_{label}"] = index if known else 0
            row[f"observed_known_pixels_{label}"] = index if known else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _manifest_rows() -> pd.DataFrame:
    rows = []
    for study_id, spatial, classification, supervision_type in (
        ("promote", 1, 1, "clean_negative"),
        ("unknown", 0, 0, "partial_json"),
        ("known", 1, 1, "partial_json"),
    ):
        row = {
            "study_id": study_id,
            "slice_index": 0,
            "segmentation_known": spatial,
            "classification_known": classification,
            "supervision_type": supervision_type,
            "any_ich": 0,
        }
        row.update({label: 0 for label in OUTPUT_LABELS[1:]})
        rows.append(row)
    return pd.DataFrame(rows)


class TestSupervisionRescore(unittest.TestCase):
    def test_promotes_only_clean_negative_spatial_accounting(self) -> None:
        result, audit = apply_supervision_manifest(
            _prediction_rows(), _manifest_rows(), expected_promotions=1
        )
        promoted = result.set_index("study_id").loc["promote"]
        self.assertEqual(promoted["known"], 1)
        for index, label in enumerate(OUTPUT_LABELS[1:], start=1):
            self.assertEqual(promoted[f"predicted_known_pixels_{label}"], index)
            self.assertEqual(promoted[f"intersection_{label}"], 0)
            self.assertEqual(promoted[f"observed_known_pixels_{label}"], 0)
        self.assertEqual(audit["promoted_slices"], 1)
        self.assertEqual(audit["promoted_slices_by_outer_fold"], {"0": 1})
        self.assertFalse(audit["model_predictions_changed"])

    def test_rejects_positive_promoted_target(self) -> None:
        manifest = _manifest_rows()
        manifest.loc[manifest["study_id"] == "promote", "IPH"] = 1
        with self.assertRaisesRegex(ValueError, "positive target"):
            apply_supervision_manifest(_prediction_rows(), manifest)

    def test_rejects_supervision_demotion(self) -> None:
        manifest = _manifest_rows()
        manifest.loc[manifest["study_id"] == "known", "segmentation_known"] = 0
        with self.assertRaisesRegex(ValueError, "demotes"):
            apply_supervision_manifest(_prediction_rows(), manifest)


if __name__ == "__main__":
    unittest.main()
