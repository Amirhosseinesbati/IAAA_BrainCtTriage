from __future__ import annotations

import numpy as np

from scripts.evaluate_fracture_mil_thresholds import (
    _map_threshold_to_half,
    _select_f1_threshold,
    _select_weight_and_threshold,
)


def test_threshold_mapping_preserves_decision_and_endpoints() -> None:
    score = np.asarray([0.0, 0.7, 1.0])
    actual = _map_threshold_to_half(score, 0.7)
    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0])


def test_f1_threshold_separates_simple_classes() -> None:
    truth = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = _select_f1_threshold(truth, score)
    np.testing.assert_array_equal(score >= threshold, truth)


def test_joint_selection_prefers_informative_candidate() -> None:
    truth = np.asarray([0, 0, 1, 1])
    reference = np.asarray([0.9, 0.8, 0.2, 0.1])
    candidate = np.asarray([0.1, 0.2, 0.8, 0.9])
    weight, threshold, metrics = _select_weight_and_threshold(
        truth, reference, candidate, [0.0, 1.0]
    )
    assert weight == 1.0
    assert metrics["f1"] == 1.0
    np.testing.assert_array_equal(candidate >= threshold, truth)
