from __future__ import annotations

import pandas as pd

from scripts.benchmark_fracture_mil_package import _select_studies


def test_select_studies_covers_longest_positive_and_typical_negative() -> None:
    frame = pd.DataFrame(
        {
            "study_id": ["a", "a", "b", "c", "c", "c"],
            "slice_index": [0, 1, 0, 0, 1, 2],
            "study_fracture": [0, 0, 1, 1, 1, 1],
        }
    )
    selected = _select_studies(frame)
    assert "c" in selected
    assert "a" in selected
