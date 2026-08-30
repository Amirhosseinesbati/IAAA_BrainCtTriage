from __future__ import annotations

import pandas as pd

from scripts.search_fracture_snapshot_ensembles import (
    _macro_paired_bootstrap,
    _select_candidate,
)


def test_select_candidate_prefers_robust_then_cheaper() -> None:
    rows = [
        {"candidate": "high_macro", "worst_fold_auc": 0.70, "macro_auc": 0.95, "n_snapshots": 1},
        {"candidate": "robust_expensive", "worst_fold_auc": 0.80, "macro_auc": 0.90, "n_snapshots": 3},
        {"candidate": "robust_cheap", "worst_fold_auc": 0.80, "macro_auc": 0.90, "n_snapshots": 2},
    ]

    assert _select_candidate(rows)["candidate"] == "robust_cheap"


def test_macro_paired_bootstrap_detects_better_candidate() -> None:
    reference = pd.DataFrame({
        "study_id": ["a", "b", "c", "d"],
        "truth": [0, 0, 1, 1],
        "probability": [0.1, 0.8, 0.2, 0.9],
    })
    candidate = reference.copy()
    candidate["probability"] = [0.1, 0.2, 0.8, 0.9]

    result = _macro_paired_bootstrap(
        {0: (reference, candidate), 1: (reference, candidate)},
        iterations=1_000,
        seed=42,
    )

    assert result["observed_macro_difference"] > 0.0
    assert result["probability_candidate_not_better"] < 0.5
