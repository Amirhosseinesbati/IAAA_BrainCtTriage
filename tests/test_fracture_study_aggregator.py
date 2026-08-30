from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.fit_fracture_study_aggregator import FEATURES, nested_group_oof


def test_nested_group_oof_scores_every_study() -> None:
    rows = []
    for fold in range(5):
        for index in range(8):
            truth = int(index < 2)
            signal = 0.75 if truth else 0.15
            row = {"fold": fold, "truth": truth}
            row.update({name: signal + 0.001 * index for name in FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)

    prediction, selections = nested_group_oof(frame, (0.03, 0.3))

    assert np.isfinite(prediction).all()
    assert set(selections) == {"0", "1", "2", "3", "4"}
    assert float(np.min(prediction)) >= 0.0
    assert float(np.max(prediction)) <= 1.0
    assert float(np.mean(prediction[:2])) > float(np.mean(prediction[2:8]))
