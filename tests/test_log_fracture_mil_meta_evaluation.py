from __future__ import annotations

from scripts.log_fracture_mil_meta_evaluation import _metric_payload


def test_metric_payload_flattens_bootstrap_and_folds() -> None:
    payload = {
        "blend_macro_auc": 0.9,
        "macro_difference": 0.03,
        "blend_worst_fold_auc": 0.8,
        "bootstrap": {
            "difference_95": [-0.01, 0.03, 0.07],
            "probability_blend_not_better": 0.05,
        },
        "per_fold": [{"fold": 2, "blend_auc": 0.91}],
    }
    actual = _metric_payload("candidate", payload)
    assert actual["candidate_macro_auc"] == 0.9
    assert actual["candidate_difference_ci95_lower"] == -0.01
    assert actual["candidate_fold2_auc"] == 0.91
