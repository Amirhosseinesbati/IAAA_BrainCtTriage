from __future__ import annotations

import numpy as np

from scripts.evaluate_fracture_rank_blend import (
    _average_percentile_rank,
    _rank_blend,
    _select_weight,
)


def test_average_percentile_rank_preserves_ties() -> None:
    actual = _average_percentile_rank(np.asarray([10.0, 20.0, 20.0, 40.0]))
    np.testing.assert_allclose(actual, [0.25, 0.625, 0.625, 1.0])


def test_rank_blend_endpoints_match_component_ranks() -> None:
    reference = np.asarray([0.1, 0.4, 0.2])
    candidate = np.asarray([0.9, 0.2, 0.5])
    np.testing.assert_allclose(
        _rank_blend(reference, candidate, 0.0),
        _average_percentile_rank(reference),
    )
    np.testing.assert_allclose(
        _rank_blend(reference, candidate, 1.0),
        _average_percentile_rank(candidate),
    )


def test_select_weight_prefers_less_candidate_on_auc_tie() -> None:
    curve = [
        {"candidate_weight": 0.5, "auc": 0.9},
        {"candidate_weight": 0.2, "auc": 0.9},
        {"candidate_weight": 0.1, "auc": 0.8},
    ]
    assert _select_weight(curve) == {"candidate_weight": 0.2, "auc": 0.9}
