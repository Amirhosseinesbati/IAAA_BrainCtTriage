from __future__ import annotations

import torch

from scripts.train_ich_subtype_temporal_residual_local import (
    subtype_temporal_delta,
    subtype_temporal_promotion_decision,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.temporal_head import SubtypeTemporalResidualHead


def test_subtype_temporal_head_preserves_any_and_is_exact_identity() -> None:
    model = SubtypeTemporalResidualHead(
        12, projection_dim=8, hidden_dim=4, dropout=0.0
    ).eval()
    features = torch.randn((3, 7, 12))
    base_logits = torch.randn((3, 7, len(OUTPUT_LABELS)))
    lengths = torch.tensor([7, 5, 2])
    output = model(features, base_logits, lengths)
    torch.testing.assert_close(output, base_logits, rtol=0.0, atol=0.0)
    with torch.no_grad():
        model.subtype_residual.bias.fill_(0.5)
    changed = model(features, base_logits, lengths)
    torch.testing.assert_close(changed[:, :, 0], base_logits[:, :, 0])
    assert not torch.equal(changed[:, :, 1:], base_logits[:, :, 1:])


def test_subtype_temporal_gate_requires_macro_gain_and_subtype_safety() -> None:
    labels = OUTPUT_LABELS[1:]
    baseline = {
        "any_ich_auc": 0.90,
        "macro_subtype_auc": 0.80,
        "subtype_auc": {label: 0.80 for label in labels},
    }
    candidate = {
        "any_ich_auc": 0.90,
        "macro_subtype_auc": 0.82,
        "subtype_auc": {
            label: 0.82 if index < 4 else 0.80
            for index, label in enumerate(labels)
        },
    }
    delta = subtype_temporal_delta(baseline, candidate)
    assert subtype_temporal_promotion_decision(delta)["promotion_allowed"]
    delta["subtype_auc"][labels[-1]] = -0.02
    assert not subtype_temporal_promotion_decision(delta)["promotion_allowed"]
