from __future__ import annotations

import numpy as np
import torch

from scripts.diagnose_ich_independent_sah_expert import (
    StreamingBinaryHistogram,
    masked_sah_expert_loss,
    separability_gate,
)


def test_streaming_histogram_separates_perfect_binary_scores() -> None:
    histogram = StreamingBinaryHistogram(bins=64)
    histogram.update(
        torch.tensor([0.95, 0.85, 0.10, 0.05]),
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
    )
    summary = histogram.summarize()
    assert summary["positive_pixels"] == 2
    assert summary["negative_pixels"] == 2
    assert np.isclose(summary["average_precision"], 1.0)
    assert np.isclose(summary["roc_auc"], 1.0)


def test_masked_sah_expert_loss_ignores_nonrecoverable_classes() -> None:
    expert = torch.zeros((1, 1, 2, 3), requires_grad=True)
    incumbent = torch.full((1, 6, 2, 3), -5.0)
    incumbent[:, 0] = 2.0
    incumbent[:, 3, 0, 0] = 6.0
    target = torch.tensor([[[5, 0, 0], [0, 0, 0]]])
    known = torch.ones(1)
    loss, statistics = masked_sah_expert_loss(
        expert, incumbent, target, known
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert statistics["eligible_pixels"] == 5.0
    assert statistics["positive_pixels"] == 0.0
    assert expert.grad is not None
    assert expert.grad[0, 0, 0, 0] == 0.0


def test_separability_gate_requires_material_threshold_free_gain() -> None:
    def metric(ap: float, auc: float, precision: float, positives: int = 800):
        return {
            "positive_pixels": positives,
            "negative_pixels": 100_000,
            "prevalence": positives / (positives + 100_000),
            "average_precision": ap,
            "roc_auc": auc,
            "precision_at_recall_0_10": precision,
            "precision_at_recall_0_25": precision,
        }

    metrics = {
        "background_or_iph": {
            "expert_gated": metric(0.060, 0.90, 0.20),
            "incumbent_gated": metric(0.040, 0.87, 0.10),
        },
        "near_incumbent_foreground": {
            "expert_gated": metric(0.12, 0.90, 0.20),
            "incumbent_gated": metric(0.08, 0.87, 0.10),
        },
    }
    assert separability_gate(metrics)["all_passed"]
    metrics["background_or_iph"]["expert_gated"]["average_precision"] = 0.041
    assert not separability_gate(metrics)["all_passed"]
