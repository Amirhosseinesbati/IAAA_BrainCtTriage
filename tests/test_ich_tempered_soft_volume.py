from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from scripts.screen_ich_tempered_soft_volume import (
    presence_locked_soft_predictions,
    soft_volume_screen_decision,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def _hard_rows() -> pd.DataFrame:
    rows = []
    for study_id, patient_id, ivh_pixels in (
        ("hard-negative", "p0", 0.0),
        ("soft-negative", "p1", 2.0),
        ("soft-positive", "p2", 2.0),
    ):
        row = {
            "study_id": study_id,
            "patient_id": patient_id,
            "slice_index": 0,
            "voxel_volume_ml": 0.1,
        }
        for label in OUTPUT_LABELS[1:]:
            row[f"pred_pixels_{label}"] = ivh_pixels if label == "IVH" else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _soft_rows() -> pd.DataFrame:
    rows = []
    for study_id, patient_id, ivh_pixels in (
        ("hard-negative", "p0", 10.0),
        ("soft-negative", "p1", 0.5),
        ("soft-positive", "p2", 3.0),
    ):
        row = {
            "study_id": study_id,
            "patient_id": patient_id,
            "slice_index": 0,
            "voxel_volume_ml": 0.1,
        }
        for label in OUTPUT_LABELS[1:]:
            row[f"soft_pixels_{label}"] = ivh_pixels if label == "IVH" else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_presence_lock_changes_only_double_positive_studies() -> None:
    candidate, diagnostics = presence_locked_soft_predictions(
        _hard_rows(), _soft_rows()
    )

    by_study = candidate.set_index("study_id")
    assert by_study.loc["hard-negative", "pred_pixels_IVH"] == 0.0
    assert by_study.loc["soft-negative", "pred_pixels_IVH"] == 2.0
    assert by_study.loc["soft-positive", "pred_pixels_IVH"] == 3.0
    assert diagnostics == {
        "studies_total": 3,
        "studies_using_soft": 1,
        "studies_using_hard": 2,
        "hard_positive_studies": 2,
        "soft_positive_studies_before_lock": 2,
    }


def test_presence_lock_rejects_misaligned_rows() -> None:
    soft = _soft_rows().copy()
    soft.loc[1, "slice_index"] = 3
    with pytest.raises(ValueError, match="misaligned"):
        presence_locked_soft_predictions(_hard_rows(), soft)


def _summary(*, mae: float, bias: float) -> dict:
    return {
        "selection_score": 0.6661624032001503,
        "mean_foreground_dice": 0.4591058994385365,
        "any_ich_study_auc": 0.9233870967741936,
        "macro_subtype_study_auc": 0.9109201965113145,
        "normal_false_positive_rate_at_0_1ml": 0.19444444444444445,
        "presence_f1_at_0_1ml": 0.8823529411764706,
        "total_volume_mae_ml": mae,
        "total_volume_bias_ml": bias,
        "subtypes": {
            label: {"mae_ml": 1.0} for label in OUTPUT_LABELS[1:]
        },
    }


def test_soft_volume_gate_requires_large_stable_volume_gain() -> None:
    baseline = _summary(mae=10.762715762077947, bias=-6.236356064112846)
    candidates = {
        0.05: _summary(mae=10.35, bias=-6.10),
        0.10: _summary(mae=10.20, bias=-5.80),
        0.20: _summary(mae=10.45, bias=-6.00),
    }

    decision = soft_volume_screen_decision(baseline, candidates)

    assert decision["decision"] == "advance_to_crossfit_oof"
    assert decision["best_temperature"] == pytest.approx(0.10)
    assert decision["stable_neighbor_temperatures"] == [0.05, 0.20]
    assert all(decision["gates"].values())


def test_soft_volume_gate_rejects_noninvariant_presence() -> None:
    baseline = _summary(mae=10.762715762077947, bias=-6.236356064112846)
    candidates = {
        0.05: _summary(mae=10.35, bias=-6.10),
        0.10: _summary(mae=10.20, bias=-5.80),
        0.20: _summary(mae=10.45, bias=-6.00),
    }
    candidates = deepcopy(candidates)
    candidates[0.10]["normal_false_positive_rate_at_0_1ml"] += 0.01

    decision = soft_volume_screen_decision(baseline, candidates)

    assert decision["decision"] == "reject_before_outer"
    assert not decision["gates"][
        "all_presence_spatial_and_auc_metrics_invariant"
    ]


def test_soft_volume_gate_rejects_subtype_mae_tradeoff() -> None:
    baseline = _summary(mae=10.762715762077947, bias=-6.236356064112846)
    candidates = {
        0.05: _summary(mae=10.35, bias=-6.10),
        0.10: _summary(mae=10.20, bias=-5.80),
        0.20: _summary(mae=10.45, bias=-6.00),
    }
    candidates[0.10]["subtypes"]["SDH"]["mae_ml"] = 1.30

    decision = soft_volume_screen_decision(baseline, candidates)

    assert decision["decision"] == "reject_before_outer"
    assert not decision["subtype_mae_noninferiority"]["SDH"]
