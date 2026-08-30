from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_fracture_mil_oof import _macro_paired_bootstrap


def test_macro_bootstrap_preserves_identical_candidates() -> None:
    predictions = pd.DataFrame(
        {
            "outer_fold": [0, 0, 0, 0, 1, 1, 1, 1],
            "truth": [0, 0, 1, 1, 0, 0, 1, 1],
            "reference": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8],
            "candidate": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8],
        }
    )
    reference, candidate = _macro_paired_bootstrap(
        predictions,
        "reference",
        "candidate",
        iterations=100,
        seed=7,
    )
    np.testing.assert_array_equal(reference, candidate)
