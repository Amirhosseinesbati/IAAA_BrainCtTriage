"""Self-contained fracture slice-to-study pooling for packaged inference."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def aggregate_fracture_scores(values: Iterable[float]) -> dict[str, float]:
    """Match the validated training/evaluation pooling definitions exactly."""
    scores = np.asarray(list(values), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("At least one one-dimensional slice score is required")
    if not np.isfinite(scores).all():
        raise ValueError("Slice scores must be finite")
    scores = np.clip(scores, 0.0, 1.0)
    descending = np.sort(scores)[::-1]

    def topk_mean(k: int) -> float:
        return float(descending[: min(k, descending.size)].mean())

    adjacent_pair = 0.0
    window3_mean = 0.0
    if scores.size >= 2:
        adjacent_pair = float(np.sqrt(scores[:-1] * scores[1:]).max())
    if scores.size >= 3:
        window3_mean = float(
            np.convolve(
                scores,
                np.ones(3, dtype=np.float64) / 3.0,
                mode="valid",
            ).max()
        )
    clipped = np.minimum(scores, 0.95)
    noisy_or = -math.expm1(float(np.log1p(-clipped).sum()))

    return {
        "max": float(descending[0]),
        "top2_mean": topk_mean(2),
        "top3_mean": topk_mean(3),
        "top5_mean": topk_mean(5),
        "adjacent_pair": adjacent_pair,
        "window3_mean": window3_mean,
        "noisy_or": float(np.clip(noisy_or, 0.0, 1.0)),
    }


def select_fracture_score(values: Iterable[float], aggregation: str) -> float:
    """Return one configured study score and fail on an unknown profile."""
    pooled = aggregate_fracture_scores(values)
    if aggregation not in pooled:
        raise ValueError(f"Unsupported fracture aggregation: {aggregation}")
    return pooled[aggregation]
