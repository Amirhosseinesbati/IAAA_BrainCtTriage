from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_fracture_mil_crossfit_blend import _select_weight


def test_select_weight_prefers_reference_only_on_exact_tie() -> None:
    development = pd.DataFrame(
        {
            "outer_fold": [0, 0, 0, 0],
            "truth": [0, 0, 1, 1],
            "reference_rank": [0.1, 0.2, 0.8, 0.9],
            "candidate_rank": [0.1, 0.2, 0.8, 0.9],
        }
    )
    result = _select_weight(development, np.asarray([0.0, 0.5, 1.0]))
    assert result["selected"]["candidate_weight"] == 0.0
