"""Submission-compatible models for staged ICH-v2 experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import SegResNet


def build_seg_resnet(
    *,
    init_filters: int = 16,
    dropout_prob: float = 0.1,
) -> torch.nn.Module:
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        dropout_prob=dropout_prob,
    )


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Load legacy state dicts or v2 checkpoint payloads."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("ICH checkpoint must contain a state dictionary")
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=strict)
    return payload if "state_dict" in payload else {"state_dict": state}
