"""Official Macro-F1 plus safety and diagnostic metrics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

from src.config import RANDOM_SEED, config_section


def _bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    units = np.unique(groups) if groups is not None else np.arange(len(y_true))
    scores: list[float] = []
    for _ in range(samples):
        selected = rng.choice(units, size=len(units), replace=True)
        if groups is None:
            indices = selected.astype(int)
        else:
            indices = np.concatenate([np.flatnonzero(groups == unit) for unit in selected])
        scores.append(float(f1_score(y_true[indices], y_pred[indices], labels=[0, 1, 2], average="macro", zero_division=0)))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(np.mean(scores)), "ci95_low": float(low), "ci95_high": float(high)}


def compute_competition_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    patient_ids: Iterable[str] | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if len(true) != len(pred) or not len(true):
        raise ValueError("y_true and y_pred must have equal non-zero length")
    if not set(np.unique(true)).issubset({0, 1, 2}) or not set(np.unique(pred)).issubset({0, 1, 2}):
        raise ValueError("Competition labels must be 0, 1, or 2")

    report = classification_report(true, pred, labels=[0, 1, 2], output_dict=True, zero_division=0)
    macro_f1 = float(f1_score(true, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    groups = np.asarray(list(patient_ids), dtype=str) if patient_ids is not None else None
    if groups is not None and len(groups) != len(true):
        raise ValueError("patient_ids must match prediction length")
    samples = bootstrap_samples
    if samples is None:
        samples = int(config_section("competition", "evaluation", "bootstrap_samples"))

    return {
        "official_metric": "macro_f1",
        "macro_f1": macro_f1,
        "qwk": float(cohen_kappa_score(true, pred, labels=[0, 1, 2], weights="quadratic")),
        "accuracy": float(accuracy_score(true, pred)),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in (0, 1, 2)
        },
        "catastrophic_errors": {
            "normal_to_critical": int(np.sum((true == 0) & (pred == 2))),
            "critical_to_normal": int(np.sum((true == 2) & (pred == 0))),
        },
        "bootstrap_macro_f1": _bootstrap_macro_f1(
            true, pred, groups, samples=samples,
            seed=int(config_section("competition", "evaluation", "bootstrap_seed")),
        ) if samples > 0 else None,
    }
