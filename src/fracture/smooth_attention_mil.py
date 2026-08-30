"""Small gated-attention MIL head with ordered-slice smoothness."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SmoothAttentionMILConfig:
    input_dim: int
    hidden_dim: int = 64
    attention_dim: int = 32
    dropout: float = 0.10


class SmoothAttentionMIL(nn.Module):
    """Gated attention over one variable-length ordered CT study."""

    def __init__(self, config: SmoothAttentionMILConfig) -> None:
        super().__init__()
        if config.input_dim < 1 or config.hidden_dim < 1 or config.attention_dim < 1:
            raise ValueError("MIL dimensions must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.attention_value = nn.Linear(config.hidden_dim, config.attention_dim)
        self.attention_gate = nn.Linear(config.hidden_dim, config.attention_dim)
        self.attention_score = nn.Linear(config.attention_dim, 1, bias=False)
        self.classifier = nn.Linear(config.hidden_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [slices, input_dim]")
        if features.shape[0] < 1 or features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected non-empty [slices, {self.config.input_dim}] features, "
                f"received {tuple(features.shape)}"
            )
        hidden = self.encoder(features)
        gated = torch.tanh(self.attention_value(hidden)) * torch.sigmoid(
            self.attention_gate(hidden)
        )
        attention_logits = self.attention_score(gated).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=0)
        pooled = torch.sum(attention_weights[:, None] * hidden, dim=0)
        study_logit = self.classifier(pooled).squeeze(-1)
        return study_logit, attention_logits, attention_weights


def smoothness_penalty(attention_logits: Tensor, order: int = 1) -> Tensor:
    """Mean graph-difference energy for an ordered path of CT slices."""
    if attention_logits.ndim != 1:
        raise ValueError("attention_logits must be one-dimensional")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    required = order + 1
    if attention_logits.numel() < required:
        return attention_logits.sum() * 0.0
    difference = attention_logits
    for _ in range(order):
        difference = torch.diff(difference)
    return torch.mean(difference.square())


def smooth_attention_loss(
    study_logit: Tensor,
    target: Tensor,
    attention_logits: Tensor,
    *,
    alpha: float,
    positive_weight: Tensor | None = None,
    smoothness_order: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, classification, and normalized smoothness losses."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    classification = F.binary_cross_entropy_with_logits(
        study_logit.reshape(1),
        target.to(dtype=study_logit.dtype).reshape(1),
        pos_weight=positive_weight,
    )
    smoothness = smoothness_penalty(attention_logits, order=smoothness_order)
    total = (1.0 - alpha) * classification + alpha * smoothness
    return total, classification, smoothness
