from __future__ import annotations

import numpy as np
import pytest

from src.fracture.pooling import aggregate_study_scores
from submission.fracture_pooling import (
    aggregate_fracture_scores,
    select_fracture_score,
)


def test_packaged_pooling_matches_validated_evaluator() -> None:
    rng = np.random.default_rng(42)
    for length in (1, 2, 3, 17, 64):
        scores = rng.uniform(-0.2, 1.2, size=length)
        expected = aggregate_study_scores(scores)
        actual = aggregate_fracture_scores(scores)
        assert actual.keys() == expected.keys()
        for name in expected:
            assert actual[name] == pytest.approx(expected[name], abs=1e-12)


def test_select_fracture_score_uses_named_profile() -> None:
    scores = [0.01, 0.7, 0.8, 0.6, 0.01]
    assert select_fracture_score(scores, "top5_mean") == pytest.approx(0.424)

    with pytest.raises(ValueError, match="Unsupported fracture aggregation"):
        select_fracture_score(scores, "unknown")
