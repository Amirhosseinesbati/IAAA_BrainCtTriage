from __future__ import annotations

import unittest

from src.fracture.pooling import aggregate_study_scores, compute_study_features


class TestFracturePooling(unittest.TestCase):
    def test_isolated_peak_is_penalized_by_consistency_features(self) -> None:
        features = compute_study_features([0.01, 0.95, 0.01, 0.01])
        self.assertAlmostEqual(features["max"], 0.95)
        self.assertLess(features["adjacent_pair"], 0.1)
        self.assertLess(features["window3_mean"], 0.4)

    def test_persistent_signal_survives_topk_and_adjacency(self) -> None:
        pooled = aggregate_study_scores([0.01, 0.7, 0.8, 0.6, 0.01])
        self.assertGreater(pooled["top3_mean"], 0.69)
        self.assertGreater(pooled["adjacent_pair"], 0.74)
        self.assertGreater(pooled["window3_mean"], 0.69)

    def test_invalid_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            compute_study_features([])
        with self.assertRaises(ValueError):
            compute_study_features([0.2, float("nan")])


if __name__ == "__main__":
    unittest.main()
