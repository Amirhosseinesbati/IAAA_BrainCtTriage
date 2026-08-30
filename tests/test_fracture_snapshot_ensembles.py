from __future__ import annotations

from scripts.search_fracture_snapshot_ensembles import _select_candidate


def test_select_candidate_prefers_robust_then_cheaper() -> None:
    rows = [
        {"candidate": "high_macro", "worst_fold_auc": 0.70, "macro_auc": 0.95, "n_snapshots": 1},
        {"candidate": "robust_expensive", "worst_fold_auc": 0.80, "macro_auc": 0.90, "n_snapshots": 3},
        {"candidate": "robust_cheap", "worst_fold_auc": 0.80, "macro_auc": 0.90, "n_snapshots": 2},
    ]

    assert _select_candidate(rows)["candidate"] == "robust_cheap"
