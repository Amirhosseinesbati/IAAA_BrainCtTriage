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


def compose_factorized_mask_logits(
    foreground_logit: torch.Tensor,
    conditional_subtype_logits: torch.Tensor,
) -> torch.Tensor:
    """Compose six-class logits from P(foreground) and P(subtype|foreground).

    The background logit is the binary reference value zero. Foreground logits
    are the binary foreground log-odds plus normalized conditional subtype
    log-probabilities. Consequently softmax over the returned tensor represents
    the exact product ``P(foreground) * P(subtype | foreground)``.
    """
    if foreground_logit.ndim != 4 or foreground_logit.shape[1] != 1:
        raise ValueError("foreground_logit must have shape (N, 1, H, W)")
    if (
        conditional_subtype_logits.ndim != 4
        or conditional_subtype_logits.shape[1] != 5
        or conditional_subtype_logits.shape[0] != foreground_logit.shape[0]
        or conditional_subtype_logits.shape[-2:] != foreground_logit.shape[-2:]
    ):
        raise ValueError(
            "conditional_subtype_logits must have shape (N, 5, H, W) "
            "matching foreground_logit"
        )
    subtype_log_probabilities = torch.nn.functional.log_softmax(
        conditional_subtype_logits, dim=1
    )
    background_logit = torch.zeros_like(foreground_logit)
    return torch.cat(
        [background_logit, foreground_logit + subtype_log_probabilities], dim=1
    )


def compose_factorized_residual_logits(
    legacy_mask_logits: torch.Tensor,
    foreground_residual: torch.Tensor,
    subtype_residual: torch.Tensor,
) -> torch.Tensor:
    """Apply factorized residuals with bit-exact zero-residual legacy logits.

    The subtype adjustment is centered in log-sum-exp space so it cannot change
    total foreground mass. The foreground residual then shifts that mass relative
    to background. Computing the composition in FP32 avoids BF16 cancellation at
    near-tied pixels while returning the original logits exactly when both
    residual tensors are zero.
    """
    if legacy_mask_logits.ndim != 4 or legacy_mask_logits.shape[1] != 6:
        raise ValueError("legacy_mask_logits must have shape (N, 6, H, W)")
    if foreground_residual.shape != legacy_mask_logits[:, :1].shape:
        raise ValueError("foreground_residual must match the background-logit shape")
    if subtype_residual.shape != legacy_mask_logits[:, 1:].shape:
        raise ValueError("subtype_residual must match the five foreground logits")
    legacy = legacy_mask_logits.float()
    foreground_residual = foreground_residual.float()
    subtype_residual = subtype_residual.float()
    legacy_subtype_logits = legacy[:, 1:]
    adjusted_subtype_logits = legacy_subtype_logits + subtype_residual
    legacy_log_normalizer = torch.logsumexp(
        legacy_subtype_logits, dim=1, keepdim=True
    )
    adjusted_log_normalizer = torch.logsumexp(
        adjusted_subtype_logits, dim=1, keepdim=True
    )
    centered_subtype_logits = adjusted_subtype_logits + (
        legacy_log_normalizer - adjusted_log_normalizer
    )
    foreground_logits = centered_subtype_logits + foreground_residual
    return torch.cat([legacy[:, :1], foreground_logits], dim=1)


class FactorizedForegroundSubtypeModel(torch.nn.Module):
    """Factorize ICH support and subtype with exact warm-start probability identity."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        for name in (
            "encoder",
            "decoder",
            "segmentation_head",
            "classification_head",
        ):
            if not isinstance(getattr(base_model, name, None), torch.nn.Module):
                raise ValueError(f"Base model does not expose {name}")
        self.base_model = base_model
        decoder_channels = _segmentation_head_input_channels(base_model)
        self.foreground_residual_head = torch.nn.Conv2d(
            decoder_channels, 1, kernel_size=3, padding=1
        )
        self.subtype_residual_head = torch.nn.Conv2d(
            decoder_channels, 5, kernel_size=3, padding=1
        )
        for head in (self.foreground_residual_head, self.subtype_residual_head):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    @staticmethod
    def legacy_factorization(
        mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact foreground log-odds and conditional subtype logits."""
        if mask_logits.ndim != 4 or mask_logits.shape[1] != 6:
            raise ValueError("Legacy mask logits must have shape (N, 6, H, W)")
        mask_logits = mask_logits.float()
        conditional_subtype_logits = mask_logits[:, 1:]
        foreground_logit = (
            torch.logsumexp(conditional_subtype_logits, dim=1, keepdim=True)
            - mask_logits[:, :1]
        )
        return foreground_logit, conditional_subtype_logits

    def forward_components(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        check_input_shape = getattr(self.base_model, "check_input_shape", None)
        if callable(check_input_shape):
            check_input_shape(images)
        features = self.base_model.encoder(images)
        if not isinstance(features, (list, tuple)) or not features:
            raise TypeError("Base encoder must return a feature sequence")
        feature_list = list(features)
        decoded = self.base_model.decoder(feature_list)
        legacy_mask_logits = self.base_model.segmentation_head(decoded)
        class_logits = self.base_model.classification_head(feature_list[-1])
        residual_features = decoded
        if residual_features.shape[-2:] != legacy_mask_logits.shape[-2:]:
            residual_features = torch.nn.functional.interpolate(
                residual_features,
                size=legacy_mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        legacy_foreground_logit, legacy_subtype_logits = self.legacy_factorization(
            legacy_mask_logits
        )
        foreground_residual = self.foreground_residual_head(residual_features)
        subtype_residual = self.subtype_residual_head(residual_features)
        foreground_logit = legacy_foreground_logit + foreground_residual
        conditional_subtype_logits = legacy_subtype_logits + subtype_residual
        mask_logits = compose_factorized_residual_logits(
            legacy_mask_logits, foreground_residual, subtype_residual
        )
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "legacy_mask_logits": legacy_mask_logits,
            "foreground_logit": foreground_logit,
            "conditional_subtype_logits": conditional_subtype_logits,
            "foreground_residual": foreground_residual,
            "subtype_residual": subtype_residual,
        }

    def forward(self, images: torch.Tensor):
        components = self.forward_components(images)
        return components["mask_logits"], components["class_logits"]


def factorized_trainable_parameters(
    model: FactorizedForegroundSubtypeModel,
) -> list[torch.nn.Parameter]:
    """Train decoder/output branches while freezing encoder and classifier."""
    if not isinstance(model, FactorizedForegroundSubtypeModel):
        raise TypeError("Expected FactorizedForegroundSubtypeModel")
    model.requires_grad_(False)
    modules = (
        model.base_model.decoder,
        model.base_model.segmentation_head,
        model.foreground_residual_head,
        model.subtype_residual_head,
    )
    for module in modules:
        module.requires_grad_(True)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    if not parameters:
        raise ValueError("Factorized foreground/subtype model has no trainable parameters")
    return parameters


def set_factorized_training_mode(model: FactorizedForegroundSubtypeModel) -> None:
    """Train spatial branches with frozen encoder/classifier and stable BN state."""
    if not isinstance(model, FactorizedForegroundSubtypeModel):
        raise TypeError("Expected FactorizedForegroundSubtypeModel")
    model.train()
    model.base_model.encoder.eval()
    model.base_model.classification_head.eval()
    model.base_model.decoder.train()
    model.base_model.segmentation_head.train()
    model.foreground_residual_head.train()
    model.subtype_residual_head.train()
    for module in (
        *model.base_model.decoder.modules(),
        *model.base_model.segmentation_head.modules(),
    ):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


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

    Initialization preserves the incumbent hard subtype mask exactly.  Native
    stage-two foreground logits are retained, while only the background logit
    is lowered when needed to guarantee the foreground winner remains above it;
    outside foreground the incumbent logits are returned bit-for-bit.
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
            # SMP decoders may apply in-place activations to encoder features.
            # Preserve an independent pre-decoder copy so the incumbent and
            # subtype decoder receive identical tensors at initialization.
            subtype_features = [feature.detach().clone() for feature in feature_list]
            decoded = self.incumbent_model.decoder(feature_list)
            mask_logits = self.incumbent_model.segmentation_head(decoded)
            class_logits = self.incumbent_model.classification_head(feature_list[-1])
        return (
            subtype_features,
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
        incumbent_background = incumbent_logits[:, :1]
        maximum_foreground = foreground_logits.amax(dim=1, keepdim=True)
        supported_background = torch.minimum(
            incumbent_background,
            maximum_foreground - self.conditional_margin,
        )
        supported_logits = torch.cat(
            [supported_background, foreground_logits],
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


class ConditionalSubtypeResidualAdapter(torch.nn.Module):
    """Apply a small bounded subtype residual without changing ICH support.

    The audited incumbent remains fully frozen.  A zero-initialized residual
    head sees detached decoder features and incumbent logits, then adjusts only
    the five foreground logits at pixels where the incumbent already predicts
    hemorrhage.  The hard foreground/background mask and auxiliary
    classification logits are therefore immutable by construction.
    """

    background_class_id = 0

    def __init__(
        self,
        incumbent_model: torch.nn.Module,
        *,
        hidden_channels: int = 16,
        maximum_logit_residual: float = 4.0,
        conditional_margin: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if maximum_logit_residual <= 0:
            raise ValueError("maximum_logit_residual must be positive")
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
        self.hidden_channels = int(hidden_channels)
        self.maximum_logit_residual = float(maximum_logit_residual)
        self.conditional_margin = float(conditional_margin)
        decoder_channels = _segmentation_head_input_channels(incumbent_model)
        input_channels = decoder_channels + 6
        groups = 4 if self.hidden_channels % 4 == 0 else 1
        self.subtype_residual_head = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.GroupNorm(groups, self.hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(self.hidden_channels, 5, kernel_size=1),
        )
        final = self.subtype_residual_head[-1]
        if not isinstance(final, torch.nn.Conv2d):
            raise TypeError("Subtype residual head must end in a convolution")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        self.incumbent_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.incumbent_model.eval()
        return self

    def _frozen_incumbent_forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.incumbent_model.encoder(images)
            if not isinstance(features, (list, tuple)) or not features:
                raise TypeError("Incumbent encoder must return a feature sequence")
            feature_list = list(features)
            decoded = self.incumbent_model.decoder(feature_list)
            mask_logits = self.incumbent_model.segmentation_head(decoded)
            class_logits = self.incumbent_model.classification_head(feature_list[-1])
        if decoded.shape[-2:] != mask_logits.shape[-2:]:
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return decoded.detach(), mask_logits.detach(), class_logits.detach()

    def forward_components(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded, incumbent_logits, class_logits = self._frozen_incumbent_forward(
            images
        )
        residual_input = torch.cat([decoded, incumbent_logits], dim=1)
        raw_residual = self.subtype_residual_head(residual_input)
        residual = self.maximum_logit_residual * torch.tanh(raw_residual)
        foreground_logits = incumbent_logits[:, 1:] + residual
        subtype_logits = torch.cat([incumbent_logits[:, :1], foreground_logits], dim=1)

        support = incumbent_logits.argmax(dim=1) != self.background_class_id
        maximum_foreground = foreground_logits.amax(dim=1, keepdim=True)
        supported_background = torch.minimum(
            incumbent_logits[:, :1],
            maximum_foreground - self.conditional_margin,
        )
        supported_logits = torch.cat([supported_background, foreground_logits], dim=1)
        mask_logits = torch.where(support[:, None], supported_logits, incumbent_logits)
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "subtype_logits": subtype_logits,
            "incumbent_mask_logits": incumbent_logits,
            "incumbent_foreground_support": support,
            "subtype_residual": residual,
        }

    def forward(self, images: torch.Tensor):
        components = self.forward_components(images)
        return components["mask_logits"], components["class_logits"]


class ConditionalSubtypeSelectiveResidualAdapter(torch.nn.Module):
    """Route a bounded subtype residual through a learned error-selection gate."""

    background_class_id = 0

    def __init__(
        self,
        incumbent_model: torch.nn.Module,
        *,
        hidden_channels: int = 16,
        maximum_logit_residual: float = 4.0,
        conditional_margin: float = 1.0,
        gate_threshold: float = 0.5,
        initial_gate_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if maximum_logit_residual <= 0:
            raise ValueError("maximum_logit_residual must be positive")
        if conditional_margin <= 0:
            raise ValueError("conditional_margin must be positive")
        if not 0 < gate_threshold < 1:
            raise ValueError("gate_threshold must be between zero and one")
        if not 0 < initial_gate_probability < 1:
            raise ValueError("initial_gate_probability must be between zero and one")
        for name in (
            "encoder",
            "decoder",
            "segmentation_head",
            "classification_head",
        ):
            if not isinstance(getattr(incumbent_model, name, None), torch.nn.Module):
                raise ValueError(f"Incumbent model does not expose {name}")
        self.incumbent_model = incumbent_model
        self.hidden_channels = int(hidden_channels)
        self.maximum_logit_residual = float(maximum_logit_residual)
        self.conditional_margin = float(conditional_margin)
        self.gate_threshold = float(gate_threshold)
        self.initial_gate_probability = float(initial_gate_probability)
        decoder_channels = _segmentation_head_input_channels(incumbent_model)
        input_channels = decoder_channels + 6
        groups = 4 if self.hidden_channels % 4 == 0 else 1
        self.selective_stem = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.GroupNorm(groups, self.hidden_channels),
            torch.nn.SiLU(),
        )
        self.subtype_residual_output = torch.nn.Conv2d(
            self.hidden_channels, 5, kernel_size=1
        )
        self.selection_gate_output = torch.nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1
        )
        torch.nn.init.zeros_(self.subtype_residual_output.weight)
        torch.nn.init.zeros_(self.subtype_residual_output.bias)
        torch.nn.init.zeros_(self.selection_gate_output.weight)
        gate_bias = torch.logit(torch.tensor(self.initial_gate_probability))
        torch.nn.init.constant_(self.selection_gate_output.bias, float(gate_bias))
        self.incumbent_model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.incumbent_model.eval()
        return self

    def _frozen_incumbent_forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.incumbent_model.encoder(images)
            if not isinstance(features, (list, tuple)) or not features:
                raise TypeError("Incumbent encoder must return a feature sequence")
            feature_list = list(features)
            decoded = self.incumbent_model.decoder(feature_list)
            mask_logits = self.incumbent_model.segmentation_head(decoded)
            class_logits = self.incumbent_model.classification_head(feature_list[-1])
        if decoded.shape[-2:] != mask_logits.shape[-2:]:
            decoded = torch.nn.functional.interpolate(
                decoded,
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return decoded.detach(), mask_logits.detach(), class_logits.detach()

    def forward_components(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded, incumbent_logits, class_logits = self._frozen_incumbent_forward(
            images
        )
        hidden = self.selective_stem(torch.cat([decoded, incumbent_logits], dim=1))
        raw_residual = self.subtype_residual_output(hidden)
        residual = self.maximum_logit_residual * torch.tanh(raw_residual)
        gate_logits = self.selection_gate_output(hidden)
        gate_probability = torch.sigmoid(gate_logits)
        if self.training:
            effective_gate = gate_probability
        else:
            effective_gate = (gate_probability >= self.gate_threshold).to(
                gate_probability.dtype
            )
        gated_residual = residual * effective_gate
        foreground_logits = incumbent_logits[:, 1:] + gated_residual
        subtype_logits = torch.cat([incumbent_logits[:, :1], foreground_logits], dim=1)

        support = incumbent_logits.argmax(dim=1) != self.background_class_id
        maximum_foreground = foreground_logits.amax(dim=1, keepdim=True)
        supported_background = torch.minimum(
            incumbent_logits[:, :1],
            maximum_foreground - self.conditional_margin,
        )
        supported_logits = torch.cat([supported_background, foreground_logits], dim=1)
        mask_logits = torch.where(support[:, None], supported_logits, incumbent_logits)
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "subtype_logits": subtype_logits,
            "incumbent_mask_logits": incumbent_logits,
            "incumbent_foreground_support": support,
            "subtype_residual": residual,
            "selection_gate_logits": gate_logits,
            "selection_gate_probability": gate_probability,
            "selection_gate_active": gate_probability >= self.gate_threshold,
        }

    def forward(self, images: torch.Tensor):
        components = self.forward_components(images)
        return components["mask_logits"], components["class_logits"]


def conditional_subtype_selective_trainable_parameters(
    model: ConditionalSubtypeSelectiveResidualAdapter,
) -> list[torch.nn.Parameter]:
    """Expose only the selective stem, residual output and gate output."""
    if not isinstance(model, ConditionalSubtypeSelectiveResidualAdapter):
        raise TypeError("Expected ConditionalSubtypeSelectiveResidualAdapter")
    model.incumbent_model.requires_grad_(False)
    modules = (
        model.selective_stem,
        model.subtype_residual_output,
        model.selection_gate_output,
    )
    for module in modules:
        module.requires_grad_(True)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    if not parameters:
        raise ValueError("Selective subtype residual adapter has no parameters")
    return parameters


def set_conditional_subtype_selective_training_mode(
    model: ConditionalSubtypeSelectiveResidualAdapter,
) -> None:
    """Use the soft gate during training and keep the incumbent in eval mode."""
    if not isinstance(model, ConditionalSubtypeSelectiveResidualAdapter):
        raise TypeError("Expected ConditionalSubtypeSelectiveResidualAdapter")
    model.train()
    model.incumbent_model.eval()
    model.selective_stem.train()
    model.subtype_residual_output.train()
    model.selection_gate_output.train()


def conditional_subtype_residual_trainable_parameters(
    model: ConditionalSubtypeResidualAdapter,
) -> list[torch.nn.Parameter]:
    """Expose only the bounded residual head parameters."""
    if not isinstance(model, ConditionalSubtypeResidualAdapter):
        raise TypeError("Expected ConditionalSubtypeResidualAdapter")
    model.incumbent_model.requires_grad_(False)
    model.subtype_residual_head.requires_grad_(True)
    parameters = list(model.subtype_residual_head.parameters())
    if not parameters:
        raise ValueError("Conditional subtype residual adapter has no parameters")
    return parameters


def set_conditional_subtype_residual_training_mode(
    model: ConditionalSubtypeResidualAdapter,
) -> None:
    """Train the stateless residual path while keeping the incumbent frozen."""
    if not isinstance(model, ConditionalSubtypeResidualAdapter):
        raise TypeError("Expected ConditionalSubtypeResidualAdapter")
    model.train()
    model.incumbent_model.eval()
    model.subtype_residual_head.train()


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
    """Train only copied decoder/head weights with all normalization stats frozen.

    The refiner starts as an exact incumbent copy.  Updating BatchNorm running
    statistics would create an unregularized second source of drift that is not
    represented in the optimizer parameter delta, so normalization modules use
    their incumbent statistics while their affine parameters remain trainable.
    """
    if not isinstance(model, ConditionalSubtypeRefinementModel):
        raise TypeError("Expected ConditionalSubtypeRefinementModel")
    model.train()
    model.incumbent_model.eval()
    model.subtype_decoder.train()
    model.subtype_segmentation_head.train()
    for module in (
        *model.subtype_decoder.modules(),
        *model.subtype_segmentation_head.modules(),
    ):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def base_segmentation_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the legacy segmentation network inside an optional adapter."""
    if isinstance(
        model,
        (
            HorizontalSymmetryInputAdapter,
            FiveSliceContextInputAdapter,
            SahBackgroundExpansionAdapter,
            FactorizedForegroundSubtypeModel,
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
    factorized_output_head: bool = False,
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
            factorized_output_head,
        )
    )
    if adapter_count > 1:
        raise ValueError("Only one ICH adapter or factorized head can be enabled")
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
    if factorized_output_head:
        return FactorizedForegroundSubtypeModel(model)
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
