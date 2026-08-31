from __future__ import annotations

import numpy as np

from submission.fracture_mil import (
    _adjacent_pair,
    _empirical_cdf,
    _topk_mean,
    _threshold_to_probability,
)


def test_adjacent_pair_preserves_slice_order() -> None:
    assert np.isclose(_adjacent_pair([0.9, 0.1, 0.8]), np.sqrt(0.09))


def test_empirical_cdf_matches_training_midrank() -> None:
    assert _empirical_cdf([1.0, 2.0, 2.0, 4.0], 2.0) == 0.5


def test_topk_mean_uses_largest_available_scores() -> None:
    assert np.isclose(_topk_mean([0.1, 0.9, 0.3], 2), 0.6)
    assert np.isclose(_topk_mean([0.1, 0.9, 0.3], 5), 1.3 / 3.0)


def test_threshold_mapping_places_decision_at_half() -> None:
    assert _threshold_to_probability(0.8, 0.8) == 0.5
    assert _threshold_to_probability(0.0, 0.8) == 0.0
    assert _threshold_to_probability(1.0, 0.8) == 1.0
