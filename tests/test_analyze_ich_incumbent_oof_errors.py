from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_ich_incumbent_oof_errors import (
    _max_consecutive,
    _presence_metrics,
    _quantiles,
)


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


if __name__ == "__main__":
    unittest.main()
