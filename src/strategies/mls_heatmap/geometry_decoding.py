"""Differentiable training geometry; the historical deployment decoder is unchanged."""

from __future__ import annotations

import torch

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.train import differentiable_keypoints_from_heatmaps


def local_softargmax_keypoints(
    logits: torch.Tensor, image_size: int, temperature: float, radius: int,
) -> torch.Tensor:
    """Expectation inside a fixed window about the detached discrete maximum.

    Geometry gradients act inside the selected window; the ordinary whole-map
    Gaussian cross-entropy still trains all locations. Coordinates use DARK's
    image-size/heatmap-size convention, not the historical endpoint linspace.
    """
    if logits.ndim != 4 or min(logits.shape) < 1:
        raise ValueError("Expected a nonempty [batch, landmarks, height, width] tensor")
    if image_size < 1 or temperature <= 0 or radius < 1:
        raise ValueError("Image size, temperature and radius must be positive")
    height, width = logits.shape[-2:]
    flat = logits.flatten(2)
    maxima = flat.detach().argmax(dim=-1, keepdim=True)
    positions = torch.arange(height * width, device=logits.device)
    xs = positions.remainder(width)
    ys = torch.div(positions, width, rounding_mode="floor")
    center_x = maxima.remainder(width)
    center_y = torch.div(maxima, width, rounding_mode="floor")
    local = (xs - center_x).abs().le(radius) & (ys - center_y).abs().le(radius)
    probabilities = torch.softmax((flat / temperature).masked_fill(~local, -torch.inf), dim=-1)
    x = (probabilities * xs.to(logits.dtype)).sum(dim=-1) * (image_size / width)
    y = (probabilities * ys.to(logits.dtype)).sum(dim=-1) * (image_size / height)
    return torch.stack((x, y), dim=-1)


def decode_training_keypoints(logits: torch.Tensor, config: MLSHeatmapConfig) -> torch.Tensor:
    if config.training_geometry_decoder == "global_softargmax":
        return differentiable_keypoints_from_heatmaps(
            logits, config.image_size, config.softargmax_temperature,
        )
    if config.training_geometry_decoder == "local_softargmax":
        return local_softargmax_keypoints(
            logits, config.image_size, config.softargmax_temperature,
            config.local_softargmax_radius,
        )
    raise ValueError("Unknown training geometry decoder")
