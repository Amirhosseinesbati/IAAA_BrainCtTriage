from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.strategies.mls_heatmap.dataset import build_mls_sampling_weights


def _mass_by_study_and_class(frame: pd.DataFrame, weights) -> pd.Series:
    weighted = frame.copy()
    weighted["weight"] = np.asarray(weights, dtype=float)
    return weighted.groupby(["is_target", "patient_id"])["weight"].sum()


class MLSStudySamplingTests(unittest.TestCase):
    def test_study_class_balanced_equalizes_studies_within_each_class(self) -> None:
        frame = pd.DataFrame({
            "patient_id": ["a"] * 11 + ["b"] * 11 + ["c"] * 5,
            "is_target": ([1] * 10 + [0]) + ([1] + [0] * 10) + [0] * 5,
        })

        weights = build_mls_sampling_weights(frame, "study_class_balanced")
        mass = _mass_by_study_and_class(frame, weights)

        self.assertTrue(np.isclose(mass.loc[1].sum(), 1.0))
        self.assertTrue(np.isclose(mass.loc[0].sum(), 1.0))
        self.assertTrue(np.allclose(mass.loc[1].to_numpy(), 0.5))
        self.assertTrue(np.allclose(mass.loc[0].to_numpy(), 1.0 / 3.0))

    def test_legacy_slice_class_balanced_preserves_row_weighting(self) -> None:
        frame = pd.DataFrame({
            "patient_id": ["a", "a", "b", "c", "c"],
            "is_target": [1, 1, 1, 0, 0],
        })

        weights = build_mls_sampling_weights(frame, "slice_class_balanced").numpy()

        self.assertTrue(np.allclose(weights[:3], 1.0 / 3.0))
        self.assertTrue(np.allclose(weights[3:], 1.0 / 2.0))
        self.assertTrue(np.isclose(weights[:3].sum(), 1.0))
        self.assertTrue(np.isclose(weights[3:].sum(), 1.0))

    def test_hybrid_sampler_uses_sqrt_study_exposure(self) -> None:
        frame = pd.DataFrame({
            "patient_id": ["a"] * 11 + ["b"] * 11 + ["c"] * 5,
            "is_target": ([1] * 10 + [0]) + ([1] + [0] * 10) + [0] * 5,
        })

        weights = build_mls_sampling_weights(
            frame, "hybrid_study_class_balanced"
        )
        mass = _mass_by_study_and_class(frame, weights)

        self.assertTrue(np.isclose(mass.loc[1].sum(), 1.0))
        self.assertTrue(np.isclose(mass.loc[0].sum(), 1.0))
        self.assertTrue(
            np.isclose(mass.loc[(1, "a")] / mass.loc[(1, "b")], np.sqrt(10.0))
        )
        self.assertTrue(
            np.isclose(mass.loc[(0, "b")] / mass.loc[(0, "a")], np.sqrt(10.0))
        )
        self.assertTrue(
            np.isclose(mass.loc[(0, "c")] / mass.loc[(0, "a")], np.sqrt(5.0))
        )
