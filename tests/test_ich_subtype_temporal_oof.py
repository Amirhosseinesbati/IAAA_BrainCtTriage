from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.crossfit_ich_subtype_temporal_residual_oof import (
    FOLD_SPECS,
    fold_auc_delta,
    oof_promotion_decision,
    paired_patient_auc_bootstrap,
    validate_fold_specs,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


def test_fold_specs_cover_every_outer_once() -> None:
    validate_fold_specs(FOLD_SPECS)
    broken = tuple(copy.deepcopy(spec) for spec in FOLD_SPECS)
    broken[0]["outer_fold"] = 1
    with pytest.raises(ValueError, match="cover 0..4"):
        validate_fold_specs(broken)


def test_patient_bootstrap_is_zero_for_identical_scores() -> None:
    patient_ids = np.repeat(np.asarray(["p0", "p1", "p2", "p3"]), 2)
    binary = np.tile(np.asarray([0.0, 1.0]), 4)
    truth = np.repeat(binary[:, None], len(OUTPUT_LABELS), axis=1)
    scores = np.repeat(np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])[:, None], len(OUTPUT_LABELS), axis=1)
    result = paired_patient_auc_bootstrap(
        patient_ids, truth, scores, scores, samples=50, seed=42
    )
    assert result["accepted_samples"] == 50
    assert result["macro_subtype_auc"]["delta_ci95"] == [0.0, 0.0]
    assert result["macro_subtype_auc"][
        "bootstrap_probability_candidate_better"
    ] == 0.0


def test_fold_delta_preserves_undefined_rare_subtype() -> None:
    baseline = {
        "any_ich_auc": 0.8,
        "macro_subtype_auc": 0.7,
        "subtype_auc": {
            label: None if label == "EDH" else 0.7
            for label in OUTPUT_LABELS[1:]
        },
    }
    candidate = {
        "any_ich_auc": 0.8,
        "macro_subtype_auc": 0.72,
        "subtype_auc": {
            label: None if label == "EDH" else 0.72
            for label in OUTPUT_LABELS[1:]
        },
    }
    delta = fold_auc_delta(baseline, candidate)
    assert delta["subtype_auc"]["EDH"] is None
    assert delta["macro_subtype_auc"] == pytest.approx(0.02)
    assert delta["selection_proxy"] == pytest.approx(0.003)


def test_oof_gate_distinguishes_primary_and_strong_support() -> None:
    delta = {
        "any_ich_auc": 0.0,
        "macro_subtype_auc": 0.02,
        "selection_proxy": 0.003,
        "subtype_auc": {
            label: value
            for label, value in zip(
                OUTPUT_LABELS[1:], [0.01, 0.01, 0.01, 0.02, 0.05], strict=True
            )
        },
    }
    folds = [{"macro_subtype_auc": value} for value in [0.01, 0.02, 0.0, 0.03, -0.005]]
    bootstrap = {
        "valid_fraction": 1.0,
        "macro_subtype_auc": {
            "bootstrap_probability_candidate_better": 0.98,
            "delta_ci95": [0.001, 0.04],
        },
    }
    decision = oof_promotion_decision(
        delta,
        folds,
        bootstrap,
        any_logits_exact=True,
        baseline_locked_match=True,
        coverage_exact=True,
    )
    assert decision["primary_allowed"]
    assert decision["strong_support"]
    delta["subtype_auc"][OUTPUT_LABELS[1]] = -0.02
    rejected = oof_promotion_decision(
        delta,
        folds,
        bootstrap,
        any_logits_exact=True,
        baseline_locked_match=True,
        coverage_exact=True,
    )
    assert not rejected["primary_allowed"]
    assert not rejected["strong_support"]
