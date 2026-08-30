from __future__ import annotations

import torch

from src.fracture.smooth_attention_mil import (
    SmoothAttentionMIL,
    SmoothAttentionMILConfig,
    smooth_attention_loss,
    smoothness_penalty,
)


def test_attention_weights_form_probability_distribution() -> None:
    torch.manual_seed(7)
    model = SmoothAttentionMIL(
        SmoothAttentionMILConfig(input_dim=5, hidden_dim=8, attention_dim=4, dropout=0.0)
    )
    study_logit, attention_logits, attention_weights = model(torch.randn(6, 5))

    assert study_logit.ndim == 0
    assert attention_logits.shape == (6,)
    assert attention_weights.shape == (6,)
    torch.testing.assert_close(attention_weights.sum(), torch.tensor(1.0))
    assert torch.all(attention_weights >= 0.0)


def test_first_order_smoothness_matches_adjacent_difference_energy() -> None:
    logits = torch.tensor([0.0, 1.0, 3.0])
    torch.testing.assert_close(smoothness_penalty(logits), torch.tensor(2.5))
    torch.testing.assert_close(
        smoothness_penalty(torch.ones(4)),
        torch.tensor(0.0),
    )


def test_zero_alpha_matches_classification_loss() -> None:
    total, classification, smoothness = smooth_attention_loss(
        torch.tensor(0.25),
        torch.tensor(1.0),
        torch.tensor([0.0, 1.0]),
        alpha=0.0,
    )
    torch.testing.assert_close(total, classification)
    assert smoothness.item() > 0.0
