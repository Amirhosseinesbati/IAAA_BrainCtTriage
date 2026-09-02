"""Pretrained 2.5D segmentation model with an auxiliary subtype head."""

from __future__ import annotations

import copy
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


class FiveSliceContextInputAdapter(torch.nn.Module):
    """Inject five-slice context while preserving a legacy three-slice model.

    Inputs contain five ordered slices with three CT windows each.  The wrapped
    incumbent receives the middle three slices exactly as before, plus a
    zero-initialized local residual learned from all five slices.  Consequently
    a legacy checkpoint is bit-identical at initialization while a small
    adapter can learn through-plane continuity without retraining the backbone.
    """

    windows_per_slice = 3
    context_slices = 5
    legacy_slices = 3

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_channels = self.windows_per_slice * self.context_slices
        self.base_input_channels = self.windows_per_slice * self.legacy_slices
        self.core_start = self.windows_per_slice
        self.core_stop = self.core_start + self.base_input_channels
        self.context_residual = torch.nn.Conv2d(
            self.input_channels,
            self.base_input_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        torch.nn.init.zeros_(self.context_residual.weight)

    def forward(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                "Five-slice context adapter expects "
                f"(N, {self.input_channels}, H, W) input"
            )
        core = images[:, self.core_start:self.core_stop]
        return self.base_model(core + self.context_residual(images))


def _segmentation_head_input_channels(model: torch.nn.Module) -> int:
    head = getattr(model, "segmentation_head", None)
    if not isinstance(head, torch.nn.Module):
        raise ValueError("Base model does not expose an SMP segmentation head")
    for module in head.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    raise ValueError("Could not infer decoder channels from segmentation head")


class SahBackgroundExpansionAdapter(torch.nn.Module):
    """Recover missed SAH from a tightly controlled incumbent support.

    The incumbent network is permanently used without gradients.  A tiny
    zero-initialized head sees its detached decoder features and mask logits,
    then adds a bounded residual only to the SAH logit at supported incumbent
    pixels.  Background is always supported; IPH can be enabled explicitly for
    a preregistered selectivity probe.  Consequently initialization is an exact
    identity and incumbent SAH can never be removed.  With the default settings,
    IVH/IPH/SDH/EDH argmax masks cannot be changed by construction.
    """

    background_class_id = 0
    iph_class_id = 2
    sah_class_id = 5

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        hidden_channels: int = 16,
        maximum_logit_residual: float = 8.0,
        include_incumbent_iph: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if maximum_logit_residual <= 0:
            raise ValueError("maximum_logit_residual must be positive")
        self.base_model = base_model
        self.hidden_channels = int(hidden_channels)
        self.maximum_logit_residual = float(maximum_logit_residual)
        self.include_incumbent_iph = bool(include_incumbent_iph)
        decoder_channels = _segmentation_head_input_channels(base_model)
        input_channels = decoder_channels + 6
        groups = 4 if self.hidden_channels % 4 == 0 else 1
        self.sah_residual_head = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.GroupNorm(groups, self.hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )
        final = self.sah_residual_head[-1]
        if not isinstance(final, torch.nn.Conv2d):
            raise TypeError("SAH residual head must end in a convolution")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def incumbent_support_mask(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """Return the only incumbent pixels where SAH may receive a residual."""
        incumbent = mask_logits.argmax(dim=1, keepdim=True)
        support = incumbent == self.background_class_id
        if self.include_incumbent_iph:
            support = support | (incumbent == self.iph_class_id)
        return support

    def _frozen_base_forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.base_model.encoder(images)
            if not isinstance(features, (list, tuple)) or not features:
                raise TypeError("Base encoder must return a feature sequence")
            feature_list = list(features)
            decoded = self.base_model.decoder(feature_list)
            mask_logits = self.base_model.segmentation_head(decoded)
            class_logits = self.base_model.classification_head(feature_list[-1])
        return decoded.detach(), mask_logits.detach(), class_logits.detach()

    def forward(self, images: torch.Tensor):
        decoded, mask_logits, class_logits = self._frozen_base_forward(images)
        residual_input = torch.cat([decoded, mask_logits], dim=1)
        raw_residual = self.sah_residual_head(residual_input)
        residual = self.maximum_logit_residual * torch.tanh(raw_residual)
        support = self.incumbent_support_mask(mask_logits)
        sah_residual = residual * support.to(residual.dtype)
        adjustment = torch.cat(
            [torch.zeros_like(mask_logits[:, : self.sah_class_id]), sah_residual],
            dim=1,
        )
        return mask_logits + adjustment, class_logits


class ConditionalSubtypeRefinementModel(torch.nn.Module):
    """Refine hemorrhage subtype while preserving incumbent foreground support.

    The incumbent encoder, decoder, mask head and classification head are
    permanently frozen.  A trainable decoder and segmentation-head copy starts
    from the same checkpoint and chooses only among the five foreground
    subtypes at pixels where the incumbent already predicts hemorrhage.  Thus
    the foreground/background hard mask, total predicted hemorrhage volume,
    Any-ICH scores and auxiliary subtype scores cannot change by construction.

    Initialization preserves the incumbent hard subtype mask exactly.  Logits
    inside incumbent foreground are reparameterized so their foreground winner
    is guaranteed to remain above background; outside foreground the incumbent
    logits are returned bit-for-bit.
    """

    background_class_id = 0
    foreground_class_ids = (1, 2, 3, 4, 5)

    def __init__(
        self,
        incumbent_model: torch.nn.Module,
        *,
        conditional_margin: float = 1.0,
    ) -> None:
        super().__init__()
        if conditional_margin <= 0:
            raise ValueError("conditional_margin must be positive")
        for name in (
            "encoder",
            "decoder",
            "segmentation_head",
            "classification_head",
        ):
            if not isinstance(getattr(incumbent_model, name, None), torch.nn.Module):
                raise ValueError(f"Incumbent model does not expose {name}")
        self.incumbent_model = incumbent_model
        self.subtype_decoder = copy.deepcopy(incumbent_model.decoder)
        self.subtype_segmentation_head = copy.deepcopy(
            incumbent_model.segmentation_head
        )
        self.conditional_margin = float(conditional_margin)
        self.incumbent_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.incumbent_model.eval()
        return self

    def _frozen_incumbent_forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.incumbent_model.encoder(images)
            if not isinstance(features, (list, tuple)) or not features:
                raise TypeError("Incumbent encoder must return a feature sequence")
            feature_list = list(features)
            decoded = self.incumbent_model.decoder(feature_list)
            mask_logits = self.incumbent_model.segmentation_head(decoded)
            class_logits = self.incumbent_model.classification_head(feature_list[-1])
        return (
            [feature.detach() for feature in feature_list],
            mask_logits.detach(),
            class_logits.detach(),
        )

    def forward_components(
        self, images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features, incumbent_logits, class_logits = self._frozen_incumbent_forward(
            images
        )
        subtype_decoded = self.subtype_decoder(features)
        subtype_logits = self.subtype_segmentation_head(subtype_decoded)
        if subtype_logits.shape != incumbent_logits.shape:
            raise ValueError("Conditional subtype logits do not match incumbent shape")

        support = incumbent_logits.argmax(dim=1) != self.background_class_id
        foreground_logits = subtype_logits[:, 1:]
        relative_foreground = foreground_logits - foreground_logits.amax(
            dim=1, keepdim=True
        )
        incumbent_background = incumbent_logits[:, :1]
        supported_logits = torch.cat(
            [
                incumbent_background,
                incumbent_background
                + self.conditional_margin
                + relative_foreground,
            ],
            dim=1,
        )
        mask_logits = torch.where(
            support[:, None], supported_logits, incumbent_logits
        )
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "subtype_logits": subtype_logits,
            "incumbent_mask_logits": incumbent_logits,
            "incumbent_foreground_support": support,
        }

    def forward(self, images: torch.Tensor):
        components = self.forward_components(images)
        return components["mask_logits"], components["class_logits"]


def conditional_subtype_trainable_parameters(
    model: ConditionalSubtypeRefinementModel,
) -> list[torch.nn.Parameter]:
    """Expose the copied decoder/head and keep the incumbent fully frozen."""
    if not isinstance(model, ConditionalSubtypeRefinementModel):
        raise TypeError("Expected ConditionalSubtypeRefinementModel")
    model.incumbent_model.requires_grad_(False)
    model.subtype_decoder.requires_grad_(True)
    model.subtype_segmentation_head.requires_grad_(True)
    parameters = [
        *model.subtype_decoder.parameters(),
        *model.subtype_segmentation_head.parameters(),
    ]
    if not parameters:
        raise ValueError("Conditional subtype refiner has no trainable parameters")
    return parameters


def set_conditional_subtype_training_mode(
    model: ConditionalSubtypeRefinementModel,
) -> None:
    """Train only the copied decoder/head with the incumbent held in eval mode."""
    if not isinstance(model, ConditionalSubtypeRefinementModel):
        raise TypeError("Expected ConditionalSubtypeRefinementModel")
    model.train()
    model.incumbent_model.eval()
    model.subtype_decoder.train()
    model.subtype_segmentation_head.train()


def base_segmentation_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the legacy segmentation network inside an optional adapter."""
    if isinstance(
        model,
        (
            HorizontalSymmetryInputAdapter,
            FiveSliceContextInputAdapter,
            SahBackgroundExpansionAdapter,
        ),
    ):
        return model.base_model
    return model


def input_adapter_residual(model: torch.nn.Module) -> torch.nn.Module:
    """Return the only trainable residual module of a supported adapter."""
    if isinstance(model, HorizontalSymmetryInputAdapter):
        return model.symmetry_residual
    if isinstance(model, FiveSliceContextInputAdapter):
        return model.context_residual
    if isinstance(model, SahBackgroundExpansionAdapter):
        return model.sah_residual_head
    raise TypeError("Model is not a supported ICH adapter")


def build_segmentation_model(
    *,
    architecture: str = DEFAULT_SEGMENTATION_ARCHITECTURE,
    encoder_name: str = DEFAULT_SEGMENTATION_ENCODER,
    pretrained: bool = False,
    dropout: float = 0.2,
    horizontal_symmetry_adapter: bool = False,
    five_slice_context_adapter: bool = False,
    sah_residual_adapter: bool = False,
    sah_residual_hidden_channels: int = 16,
    sah_maximum_logit_residual: float = 8.0,
    sah_include_incumbent_iph: bool = False,
) -> torch.nn.Module:
    adapter_count = sum(
        bool(value)
        for value in (
            horizontal_symmetry_adapter,
            five_slice_context_adapter,
            sah_residual_adapter,
        )
    )
    if adapter_count > 1:
        raise ValueError("Only one ICH adapter can be enabled")
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
    if five_slice_context_adapter:
        return FiveSliceContextInputAdapter(model)
    if sah_residual_adapter:
        return SahBackgroundExpansionAdapter(
            model,
            hidden_channels=sah_residual_hidden_channels,
            maximum_logit_residual=sah_maximum_logit_residual,
            include_incumbent_iph=sah_include_incumbent_iph,
        )
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
