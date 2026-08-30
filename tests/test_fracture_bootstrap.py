from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.compare_fracture_study_predictions import _sampled_auc, _score_columns


def test_sampled_auc_matches_sklearn_with_ties() -> None:
    scores = np.asarray([0.1, 0.2, 0.2, 0.8, 0.8, 0.9], dtype=np.float64)
    positive = np.asarray([[3, 4, 5], [4, 4, 3]], dtype=np.int64)
    negative = np.asarray([[0, 1, 2], [1, 2, 2]], dtype=np.int64)

    actual = _sampled_auc(scores, positive, negative)
    expected = []
    for positive_indices, negative_indices in zip(positive, negative, strict=True):
        indices = np.concatenate([negative_indices, positive_indices])
        truth = np.concatenate(
            [np.zeros(negative_indices.size), np.ones(positive_indices.size)]
        )
        expected.append(roc_auc_score(truth, scores[indices]))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_score_columns_support_asymmetric_pooling() -> None:
    assert _score_columns("prob_adjacent_pair", None, None) == (
        "prob_adjacent_pair",
        "prob_adjacent_pair",
    )
    assert _score_columns(
        "prob_adjacent_pair",
        "prob_noisy_or",
        "prob_top5_mean",
    ) == ("prob_noisy_or", "prob_top5_mean")
