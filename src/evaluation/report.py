"""Build study-level triage error tables from OOF intermediate predictions."""

from __future__ import annotations

import pandas as pd

from src.evaluation.triage import VOLUME_KEYS, decision_margins, triage_rule_trace

INTERMEDIATE_KEYS = (*VOLUME_KEYS, "fracture_prob", "MLS_mm")


def build_error_table(frame: pd.DataFrame, *, prefix: str = "pred_") -> pd.DataFrame:
    missing = {f"{prefix}{key}" for key in INTERMEDIATE_KEYS} - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for _, source in frame.iterrows():
        values = {key: source[f"{prefix}{key}"] for key in INTERMEDIATE_KEYS}
        predicted, rule = triage_rule_trace(values)
        margins = decision_margins(values)
        row = source.to_dict()
        row.update({"pred_triage": predicted, "pred_primary_rule": rule, **{f"margin_{key}": value for key, value in margins.items()}})
        if "triage_class" in source:
            row["correct"] = int(predicted == int(source["triage_class"]))
            row["error_direction"] = predicted - int(source["triage_class"])
        rows.append(row)
    return pd.DataFrame(rows)
