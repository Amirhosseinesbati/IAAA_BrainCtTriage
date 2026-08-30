"""Deterministic slice-to-study pooling for skull-fracture predictions."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _scores(values: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(values), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("At least one one-dimensional slice score is required")
    if not np.isfinite(scores).all():
        raise ValueError("Slice scores must be finite")
    return np.clip(scores, 0.0, 1.0)


def compute_study_features(values: Iterable[float]) -> dict[str, float]:
    """Return fixed, label-free study features from ordered slice scores."""
    scores = _scores(values)
    descending = np.sort(scores)[::-1]

    def topk_mean(k: int) -> float:
        return float(descending[: min(k, descending.size)].mean())

    adjacent_pair = 0.0
    window3 = 0.0
    if scores.size >= 2:
        adjacent_pair = float(np.sqrt(scores[:-1] * scores[1:]).max())
    if scores.size >= 3:
        window3 = float(
            np.convolve(scores, np.ones(3, dtype=np.float64) / 3.0, mode="valid").max()
        )

    return {
        "max": float(descending[0]),
        "top2_mean": topk_mean(2),
        "top3_mean": topk_mean(3),
        "top5_mean": topk_mean(5),
        "adjacent_pair": adjacent_pair,
        "window3_mean": window3,
        "count_ge_025": float(np.count_nonzero(scores >= 0.25)),
        "count_ge_050": float(np.count_nonzero(scores >= 0.50)),
        "fraction_ge_025": float(np.mean(scores >= 0.25)),
        "n_slices": float(scores.size),
    }


def aggregate_study_scores(values: Iterable[float]) -> dict[str, float]:
    """Return interpretable, label-free candidate study probabilities."""
    features = compute_study_features(values)
    scores = _scores(values)
    clipped = np.minimum(scores, 0.95)
    noisy_or = -math.expm1(float(np.log1p(-clipped).sum()))
    return {
        "max": features["max"],
        "top2_mean": features["top2_mean"],
        "top3_mean": features["top3_mean"],
        "top5_mean": features["top5_mean"],
        "adjacent_pair": features["adjacent_pair"],
        "window3_mean": features["window3_mean"],
        "noisy_or": float(np.clip(noisy_or, 0.0, 1.0)),
    }
