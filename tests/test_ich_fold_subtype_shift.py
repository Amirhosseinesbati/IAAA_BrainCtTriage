import unittest

import numpy as np
import pandas as pd

from scripts.analyze_ich_fold_subtype_shift import (
    _bootstrap_fold2_median_delta,
    _positive_summary,
)


class ICHFoldSubtypeShiftTests(unittest.TestCase):
    def test_positive_summary_counts_small_and_isolated_lesions(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["a", "b", "c"],
                "gt_V_IVH": [0.25, 3.0, 0.0],
                "gt_V_IPH": [0.0, 1.0, 0.0],
                "gt_V_SDH": [0.0, 0.0, 0.0],
                "gt_V_EDH": [0.0, 0.0, 0.0],
                "gt_V_SAH": [0.0, 0.0, 0.0],
                "positive_slices_IVH": [1, 4, 0],
            }
        )
        summary = _positive_summary(
            frame,
            subtype="IVH",
            volume_column="gt_V_IVH",
            positive_slice_column="positive_slices_IVH",
        )

        self.assertEqual(summary["positive_studies"], 2)
        self.assertEqual(summary["isolated_positive_studies"], 1)
        self.assertEqual(summary["positive_fraction_below_0p5ml"], 0.5)
        self.assertEqual(summary["positive_slices_median"], 2.5)

    def test_bootstrap_detects_clear_fold2_median_shift(self):
        result = _bootstrap_fold2_median_delta(
            np.asarray([8.0, 9.0, 10.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            samples=500,
            seed=7,
        )

        self.assertGreater(result["fold2_minus_other_median_ml"], 0.0)
        self.assertGreater(result["bootstrap_ci95_ml"][0], 0.0)
        self.assertEqual(result["bootstrap_probability_fold2_larger"], 1.0)


if __name__ == "__main__":
    unittest.main()
