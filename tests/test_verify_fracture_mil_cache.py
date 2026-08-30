from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.verify_fracture_mil_cache import build_study_predictions


def test_build_study_predictions_respects_slice_order() -> None:
    slices = pd.DataFrame(
        {
            "study_id": ["a", "a", "a", "b"],
            "outer_fold": [2, 2, 2, 1],
            "slice_index": [2, 0, 1, 0],
            "study_fracture": [1, 1, 1, 0],
        }
    )
    scores = np.asarray([0.8, 0.1, 0.4, 0.2], dtype=np.float32)

    actual = build_study_predictions(slices, scores, validation_fold=2)

    assert actual["study_id"].tolist() == ["a"]
    assert actual["truth"].tolist() == [1]
    assert np.isclose(actual.loc[0, "prob_max"], 0.8)
    assert np.isclose(actual.loc[0, "prob_adjacent_pair"], np.sqrt(0.4 * 0.8))
