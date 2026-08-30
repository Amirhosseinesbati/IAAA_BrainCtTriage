from __future__ import annotations

from scripts.select_fracture_pooling_crossfold import _best_profile


def test_best_profile_prioritizes_worst_fold_then_macro() -> None:
    table = {
        "prob_max": {"worst_fold_auc": 0.70, "macro_auc": 0.95},
        "prob_top2_mean": {"worst_fold_auc": 0.80, "macro_auc": 0.85},
        "prob_top3_mean": {"worst_fold_auc": 0.75, "macro_auc": 0.90},
        "prob_top5_mean": {"worst_fold_auc": 0.74, "macro_auc": 0.91},
        "prob_adjacent_pair": {"worst_fold_auc": 0.73, "macro_auc": 0.92},
        "prob_window3_mean": {"worst_fold_auc": 0.72, "macro_auc": 0.93},
        "prob_noisy_or": {"worst_fold_auc": 0.71, "macro_auc": 0.94},
    }

    assert _best_profile(table) == "prob_top2_mean"
