"""Leaderboard-compatible timm model for 9-channel adjacent CT slices."""

from __future__ import annotations

from pathlib import Path

import timm
import torch

from .cache import OUTPUT_LABELS


DEFAULT_MODEL_NAME = "efficientnet_b0"


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> torch.nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=9,
        num_classes=len(OUTPUT_LABELS),
        drop_rate=dropout,
    )


def load_model_weights(model: torch.nn.Module, checkpoint: str | Path) -> dict[str, object]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("2.5D checkpoint must be a dictionary")
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    return payload
