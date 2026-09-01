"""Pretrained 2.5D segmentation model with an auxiliary subtype head."""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from .cache import OUTPUT_LABELS


DEFAULT_SEGMENTATION_ARCHITECTURE = "unetplusplus"
DEFAULT_SEGMENTATION_ENCODER = "efficientnet-b2"


class HorizontalSymmetryInputAdapter(torch.nn.Module):
    """Add a zero-initialized learned residual from image/mirror pairs.

    The wrapped model still receives nine channels. At initialization the
    residual is exactly zero, so a legacy checkpoint produces identical
    outputs. Training can then learn a small symmetry-aware correction without
    changing the pretrained segmentation network.
    """

    def __init__(self, base_model: torch.nn.Module, *, input_channels: int = 9) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.base_model = base_model
        self.input_channels = int(input_channels)
        self.symmetry_residual = torch.nn.Conv2d(
            self.input_channels * 2,
            self.input_channels,
            kernel_size=1,
            bias=False,
        )
        torch.nn.init.zeros_(self.symmetry_residual.weight)

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                "Horizontal symmetry adapter expects "
                f"(N, {self.input_channels}, H, W) input"
            )
        paired = torch.cat([images, torch.flip(images, dims=(-1,))], dim=1)
        return self.base_model(images + self.symmetry_residual(paired))


def base_segmentation_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the legacy segmentation network inside an optional adapter."""
    if isinstance(model, HorizontalSymmetryInputAdapter):
        return model.base_model
    return model


def build_segmentation_model(
    *,
    architecture: str = DEFAULT_SEGMENTATION_ARCHITECTURE,
    encoder_name: str = DEFAULT_SEGMENTATION_ENCODER,
    pretrained: bool = False,
    dropout: float = 0.2,
    horizontal_symmetry_adapter: bool = False,
) -> torch.nn.Module:
    normalized = architecture.lower().replace("_", "").replace("+", "plus")
    architectures = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }
    if normalized not in architectures:
        raise ValueError(f"Unsupported ICH segmentation architecture: {architecture}")
    model = architectures[normalized](
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=9,
        classes=6,
        activation=None,
        aux_params={
            "pooling": "avg",
            "dropout": dropout,
            "activation": None,
            "classes": len(OUTPUT_LABELS),
        },
    )
    if horizontal_symmetry_adapter:
        return HorizontalSymmetryInputAdapter(model, input_channels=9)
    return model


def load_segmentation_weights(
    model: torch.nn.Module, checkpoint: str | Path
) -> dict[str, object]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("2.5D segmentation checkpoint must be a dictionary")
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    return payload
