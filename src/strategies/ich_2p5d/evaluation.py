"""Study pooling and calibration rules selected without touching the outer fold."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .cache import OUTPUT_LABELS


POOLING_METHODS = ("max", "top3_mean", "top5_mean", "adjacent_pair")


@dataclass(frozen=True)
class PresenceRule:
    pooling: str
    threshold: float
    calibration_f1: float
    calibration_sensitivity: float
    calibration_specificity: float


def pool_scores(values: np.ndarray, method: str) -> float:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Pooling requires a non-empty one-dimensional score array")
    if method == "max":
        return float(scores.max())
    if method in {"top3_mean", "top5_mean"}:
        count = 3 if method == "top3_mean" else 5
        return float(np.sort(scores)[-min(count, len(scores)):].mean())
    if method == "adjacent_pair":
        if len(scores) == 1:
            return float(scores[0])
        return float(np.sqrt(scores[:-1] * scores[1:]).max())
    raise ValueError(f"Unknown pooling method: {method}")


def aggregate_studies(slice_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study_id, group in slice_predictions.sort_values(
        ["study_id", "slice_index"]
    ).groupby("study_id", sort=True):
        row: dict[str, object] = {
            "study_id": str(study_id),
            "truth_any_ich": int(group["truth_any_ich"].max()),
        }
        any_scores = group["prob_any_ich"].to_numpy(dtype=np.float64)
        for method in POOLING_METHODS:
            row[f"score_{method}"] = pool_scores(any_scores, method)
        for label in OUTPUT_LABELS[1:]:
            row[f"score_{label}"] = float(group[f"prob_{label}"].max())
            row[f"truth_{label}"] = int(group[f"truth_{label}"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    truth = truth.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.count_nonzero(truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return float(f1_score(truth, predicted, zero_division=0)), sensitivity, specificity


def select_presence_rule(
    studies: pd.DataFrame,
    *,
    minimum_sensitivity: float = 0.95,
) -> PresenceRule:
    """Choose pooling and threshold using only the calibration fold."""
    truth = studies["truth_any_ich"].to_numpy(dtype=np.int64)
    candidates: list[PresenceRule] = []
    for method in POOLING_METHODS:
        scores = studies[f"score_{method}"].to_numpy(dtype=np.float64)
        thresholds = np.unique(np.concatenate([
            np.linspace(0.01, 0.99, 99), scores,
        ]))
        for threshold in thresholds:
            f1, sensitivity, specificity = _binary_metrics(truth, scores >= threshold)
            if sensitivity + 1e-12 >= minimum_sensitivity:
                candidates.append(PresenceRule(
                    pooling=method,
                    threshold=float(threshold),
                    calibration_f1=f1,
                    calibration_sensitivity=sensitivity,
                    calibration_specificity=specificity,
                ))
    if not candidates:
        raise ValueError("No calibration threshold satisfies minimum sensitivity")
    return max(
        candidates,
        key=lambda item: (
            item.calibration_f1,
            item.calibration_specificity,
            item.threshold,
            -POOLING_METHODS.index(item.pooling),
        ),
    )


def evaluate_presence_rule(studies: pd.DataFrame, rule: PresenceRule) -> dict[str, float | str]:
    truth = studies["truth_any_ich"].to_numpy(dtype=np.int64)
    scores = studies[f"score_{rule.pooling}"].to_numpy(dtype=np.float64)
    f1, sensitivity, specificity = _binary_metrics(truth, scores >= rule.threshold)
    return {
        "pooling": rule.pooling,
        "threshold": rule.threshold,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": float(roc_auc_score(truth, scores)),
    }


def rule_as_dict(rule: PresenceRule) -> dict[str, object]:
    return asdict(rule)
