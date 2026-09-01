from __future__ import annotations

import unittest

import pandas as pd

from scripts.render_ich_oof_false_positive_montage import (
    _select_false_positive_slices,
)


class TestFalsePositiveMontageSelection(unittest.TestCase):
    def test_selects_largest_predicted_slice_for_top_false_positive(self) -> None:
        errors = pd.DataFrame({
            "study_id": ["a", "b", "c"],
            "presence_error": ["false_positive", "false_positive", "true_positive"],
            "pred_total_ml": [3.0, 8.0, 20.0],
        })
        slices = pd.DataFrame({
            "study_id": ["a", "a", "b", "b", "c"],
            "patient_id": ["pa", "pa", "pb", "pb", "pc"],
            "outer_fold": [0, 0, 1, 1, 2],
            "slice_index": [0, 1, 0, 1, 0],
            "pred_pixels_IVH": [0, 5, 2, 9, 100],
            "pred_pixels_IPH": [0, 0, 0, 0, 0],
            "pred_pixels_SDH": [0, 0, 0, 0, 0],
            "pred_pixels_EDH": [0, 0, 0, 0, 0],
            "pred_pixels_SAH": [0, 0, 0, 0, 0],
        })
        result = _select_false_positive_slices(errors, slices, top_studies=1)
        self.assertEqual(result["study_id"].tolist(), ["b"])
        self.assertEqual(result["slice_index"].tolist(), [1])
        self.assertEqual(result["predicted_foreground_pixels"].tolist(), [9])


if __name__ == "__main__":
    unittest.main()
