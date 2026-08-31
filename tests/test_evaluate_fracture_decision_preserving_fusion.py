from __future__ import annotations

import numpy as np

from scripts.evaluate_fracture_decision_preserving_fusion import (
    _decision_preserving_score,
)


def test_decision_preserving_score_keeps_boundary_and_ranking() -> None:
    incumbent = np.asarray([0.7, 0.7, 0.9, 0.9])
    ranking = np.asarray([0.2, 0.8, 0.2, 0.8])
    actual = _decision_preserving_score(incumbent, ranking, 0.8)
    np.testing.assert_allclose(actual, [0.1, 0.4, 0.6, 0.9])
    np.testing.assert_array_equal(actual >= 0.5, incumbent >= 0.8)
