"""Apply a pre-calibrated 2.5D presence rule to 3D volume predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.ich_v2.evaluation import (
    VOLUME_KEYS,
    add_oracle_context_triage,
    summarize_ich_predictions,
)

from .evaluation import PresenceRule


def gate_volume_predictions(
    volume_predictions: pd.DataFrame,
    presence_studies: pd.DataFrame,
    rule: PresenceRule,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Zero all ICH volumes when the independently calibrated gate is negative."""
    score_column = f"score_{rule.pooling}"
    if score_column not in presence_studies:
        raise ValueError(f"Presence predictions lack {score_column}")
    left = volume_predictions.copy()
    right = presence_studies.loc[:, ["study_id", "truth_any_ich", score_column]].copy()
    left["study_id"] = left["study_id"].astype(str)
    right["study_id"] = right["study_id"].astype(str)
    if set(left["study_id"]) != set(right["study_id"]):
        raise ValueError("3D and 2.5D study sets do not match")
    merged = left.merge(right, on="study_id", how="inner", validate="one_to_one")
    merged["gate_positive"] = merged[score_column] >= rule.threshold
    for key in VOLUME_KEYS:
        merged.loc[~merged["gate_positive"], f"pred_{key}"] = 0.0
    merged = add_oracle_context_triage(merged)
    summary = summarize_ich_predictions(merged)
    truth = merged["truth_any_ich"].astype(bool)
    summary["presence_gate"] = {
        "pooling": rule.pooling,
        "threshold": rule.threshold,
        "positive_studies": int(merged["gate_positive"].sum()),
        "suppressed_studies": int((~merged["gate_positive"]).sum()),
        "false_negative_studies": int((truth & ~merged["gate_positive"]).sum()),
        "false_positive_studies": int((~truth & merged["gate_positive"]).sum()),
    }
    return merged, summary
