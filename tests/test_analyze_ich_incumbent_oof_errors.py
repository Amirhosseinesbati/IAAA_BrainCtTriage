from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analyze_ich_incumbent_oof_errors import (
    _build_hard_negative_slice_table,
    _max_consecutive,
    _presence_metrics,
    _quantiles,
)
from scripts.compare_ich_2p5d_segmentation_oof import VariantResult


class TestIncumbentOofErrorHelpers(unittest.TestCase):
    def test_presence_metrics(self) -> None:
        result = _presence_metrics(
            np.asarray([1, 1, 0, 0], dtype=bool),
            np.asarray([1, 0, 1, 0], dtype=bool),
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_quantiles_ignore_non_finite(self) -> None:
        result = _quantiles([1.0, 2.0, float("nan"), float("inf")])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["median"], 1.5)
        self.assertEqual(result["max"], 2.0)

    def test_max_consecutive(self) -> None:
        self.assertEqual(_max_consecutive([False, True, True, False, True]), 2)
        self.assertEqual(_max_consecutive([False, False]), 0)

    def test_hard_negative_manifest_keeps_only_oof_false_positive_slices(self) -> None:
        slices = pd.DataFrame({
            "study_id": ["fp", "fp", "tn", "tp"],
            "patient_id": ["p1", "p1", "p2", "p3"],
            "slice_index": [0, 1, 0, 0],
            "outer_fold": [2, 2, 1, 0],
            "voxel_volume_ml": [0.01] * 4,
            "prob_any_ich": [0.9, 0.8, 0.1, 0.95],
            "pred_pixels_IVH": [0, 2, 0, 3],
            "pred_pixels_IPH": [0, 1, 0, 0],
            "pred_pixels_SDH": [0, 0, 0, 0],
            "pred_pixels_EDH": [0, 0, 0, 0],
            "pred_pixels_SAH": [0, 0, 0, 0],
        })
        errors = pd.DataFrame({
            "study_id": ["fp", "tn", "tp"],
            "presence_error": ["false_positive", "true_negative", "true_positive"],
        })
        variant = VariantResult(
            name="test",
            slices=slices,
            studies=pd.DataFrame(),
            sufficient=pd.DataFrame(),
            summary={},
            fold_summaries=[],
            runs=[],
        )
        result = _build_hard_negative_slice_table(variant, errors)
        self.assertEqual(result["study_id"].tolist(), ["fp"])
        self.assertEqual(result["slice_index"].tolist(), [1])
        self.assertEqual(result["predicted_foreground_pixels"].tolist(), [3])
        self.assertAlmostEqual(result["predicted_foreground_volume_ml"].item(), 0.03)


if __name__ == "__main__":
    unittest.main()
