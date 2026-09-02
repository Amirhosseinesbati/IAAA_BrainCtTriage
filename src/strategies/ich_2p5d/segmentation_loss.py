"""Multi-task segmentation and slice-subtype loss for the 2.5D ICH model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.strategies.ich_v2.losses import MaskedDiceFocalLoss


DIFFUSE_HEMORRHAGE_CLASS_IDS = (3, 5)  # SDH and SAH
SAH_CLASS_ID = 5
SEGMENTATION_OBJECTIVES = ("multiclass", "hierarchical_foreground_subtype")
CONDITIONAL_SUBTYPE_MODES = ("cross_entropy", "balanced_softmax")


def foreground_logit_from_multiclass(mask_logits: torch.Tensor) -> torch.Tensor:
    """Return the exact foreground-vs-background logit of a multiclass head.

    ``sigmoid(result)`` equals the sum of all foreground softmax probabilities.
    The hierarchy can therefore improve support while preserving the existing
    six-channel inference contract.
    """
    if mask_logits.ndim < 3 or mask_logits.shape[1] != 6:
        raise ValueError("ICH foreground logit expects [B, 6, ...] logits")
    stable = mask_logits.float()
    return torch.logsumexp(stable[:, 1:], dim=1) - stable[:, 0]


class HierarchicalForegroundSubtypeLoss(nn.Module):
    """Separate hemorrhage support from conditional subtype discrimination."""

    def __init__(
        self,
        *,
        foreground_class_weights: torch.Tensor | None = None,
        foreground_class_counts: torch.Tensor | None = None,
        conditional_subtype_mode: str = "cross_entropy",
        foreground_dice_weight: float = 0.40,
        foreground_focal_weight: float = 0.20,
        conditional_subtype_weight: float = 0.30,
        subtype_ovr_weight: float = 0.10,
        focal_gamma: float = 2.0,
        background_weight: float = 0.15,
        empty_foreground_weight: float = 0.0,
        empty_foreground_top_fraction: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        objective_weights = (
            foreground_dice_weight,
            foreground_focal_weight,
            conditional_subtype_weight,
            subtype_ovr_weight,
        )
        if any(weight < 0 for weight in objective_weights) or sum(objective_weights) <= 0:
            raise ValueError(
                "Hierarchical objective weights must be non-negative with a positive sum"
            )
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if background_weight <= 0:
            raise ValueError("background_weight must be positive")
        if empty_foreground_weight < 0:
            raise ValueError("empty_foreground_weight must be non-negative")
        if not 0 < empty_foreground_top_fraction <= 1:
            raise ValueError("empty_foreground_top_fraction must be in (0, 1]")
        weights = torch.ones(5, dtype=torch.float32)
        if foreground_class_weights is not None:
            weights = foreground_class_weights.detach().float().flatten().clone()
            if weights.numel() != 5:
                raise ValueError("foreground_class_weights must contain five values")
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError(
                    "foreground_class_weights must be finite and positive"
                )
        self.register_buffer("foreground_class_weights", weights)
        if conditional_subtype_mode not in CONDITIONAL_SUBTYPE_MODES:
            raise ValueError(
                "conditional_subtype_mode must be one of: "
                f"{', '.join(CONDITIONAL_SUBTYPE_MODES)}"
            )
        counts = torch.ones(5, dtype=torch.float32)
        if foreground_class_counts is not None:
            counts = foreground_class_counts.detach().float().flatten().clone()
            if counts.numel() != 5:
                raise ValueError("foreground_class_counts must contain five values")
            if not torch.isfinite(counts).all() or torch.any(counts <= 0):
                raise ValueError("foreground_class_counts must be finite and positive")
        if conditional_subtype_mode == "balanced_softmax" and foreground_class_counts is None:
            raise ValueError(
                "balanced_softmax requires foreground_class_counts"
            )
        self.register_buffer("foreground_class_counts", counts)
        self.conditional_subtype_mode = conditional_subtype_mode
        self.foreground_dice_weight = float(foreground_dice_weight)
        self.foreground_focal_weight = float(foreground_focal_weight)
        self.conditional_subtype_weight = float(conditional_subtype_weight)
        self.subtype_ovr_weight = float(subtype_ovr_weight)
        self.focal_gamma = float(focal_gamma)
        self.background_weight = float(background_weight)
        self.empty_foreground_weight = float(empty_foreground_weight)
        self.empty_foreground_top_fraction = float(empty_foreground_top_fraction)
        self.smooth = float(smooth)

    def components(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        supervision: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hasattr(logits, "as_tensor"):
            logits = logits.as_tensor()
        if hasattr(target, "as_tensor"):
            target = target.as_tensor()
        if hasattr(supervision, "as_tensor"):
            supervision = supervision.as_tensor()
        if logits.ndim < 3 or logits.shape[1] != 6:
            raise ValueError("Hierarchical ICH loss expects [B, 6, ...] logits")
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target = target.squeeze(1)
        if supervision.ndim == logits.ndim and supervision.shape[1] == 1:
            supervision = supervision.squeeze(1)
        expected = logits.shape[:1] + logits.shape[2:]
        if target.shape != expected or supervision.shape != expected:
            raise ValueError(
                "Target or supervision is incompatible with hierarchical logits"
            )
        target = target.long()
        valid = supervision > 0.5
        if not torch.any(valid):
            raise ValueError("Batch contains no supervised voxels")
        if target.min() < 0 or target.max() >= 6:
            raise ValueError("Target contains an invalid ICH class")

        stable = logits.float()
        foreground_logit = foreground_logit_from_multiclass(stable)
        foreground_probability = torch.sigmoid(foreground_logit)
        foreground_target = target > 0
        binary_target = foreground_target.float()
        binary_bce = F.binary_cross_entropy_with_logits(
            foreground_logit, binary_target, reduction="none"
        )
        correct_probability = torch.where(
            foreground_target,
            foreground_probability,
            1.0 - foreground_probability,
        )
        voxel_weights = torch.where(
            foreground_target,
            torch.ones_like(foreground_probability),
            torch.full_like(foreground_probability, self.background_weight),
        )
        focal_values = binary_bce * (
            1.0 - correct_probability
        ).pow(self.focal_gamma)
        foreground_focal = (
            focal_values[valid] * voxel_weights[valid]
        ).sum() / voxel_weights[valid].sum().clamp_min(1e-8)

        foreground_valid = valid & foreground_target
        if torch.any(foreground_valid):
            intersection = (
                foreground_probability * binary_target * valid.float()
            ).sum()
            predicted = (foreground_probability * valid.float()).sum()
            observed = (binary_target * valid.float()).sum()
            foreground_dice = 1.0 - (
                2.0 * intersection + self.smooth
            ) / (predicted + observed + self.smooth)

            subtype_logits = stable[:, 1:].movedim(1, -1)[foreground_valid]
            subtype_target = target[foreground_valid] - 1
            if self.conditional_subtype_mode == "balanced_softmax":
                balanced_logits = subtype_logits + torch.log(
                    self.foreground_class_counts
                )
                conditional_subtype = F.cross_entropy(
                    balanced_logits,
                    subtype_target,
                )
            else:
                conditional_subtype = F.cross_entropy(
                    subtype_logits,
                    subtype_target,
                    weight=self.foreground_class_weights,
                )
            one_hot = F.one_hot(subtype_target, num_classes=5).float()
            subtype_bce = F.binary_cross_entropy_with_logits(
                subtype_logits, one_hot, reduction="none"
            )
            subtype_probability = torch.sigmoid(subtype_logits)
            subtype_correct_probability = torch.where(
                one_hot > 0.5,
                subtype_probability,
                1.0 - subtype_probability,
            )
            subtype_focal = subtype_bce * (
                1.0 - subtype_correct_probability
            ).pow(self.focal_gamma)
            positive_focal = (subtype_focal * one_hot).sum(dim=1)
            negative_focal = (
                subtype_focal * (1.0 - one_hot)
            ).sum(dim=1) / 4.0
            sample_weights = self.foreground_class_weights[subtype_target]
            subtype_ovr = (
                (0.5 * positive_focal + 0.5 * negative_focal) * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1e-8)
        else:
            zero = stable.sum() * 0.0
            foreground_dice = zero
            conditional_subtype = zero
            subtype_ovr = zero

        valid_per_sample = valid.flatten(start_dim=1).any(dim=1)
        foreground_per_sample = foreground_valid.flatten(start_dim=1).any(dim=1)
        empty_rows = valid_per_sample & ~foreground_per_sample
        if torch.any(empty_rows):
            empty_values = F.softplus(foreground_logit)
            if self.empty_foreground_top_fraction >= 1.0:
                empty_valid = valid & empty_rows.reshape(
                    (-1,) + (1,) * (valid.ndim - 1)
                )
                empty_foreground = empty_values[empty_valid].mean()
            else:
                hard_losses = []
                for index in torch.nonzero(empty_rows, as_tuple=False).flatten():
                    sample = empty_values[index][valid[index]]
                    count = max(
                        1,
                        math.ceil(
                            sample.numel() * self.empty_foreground_top_fraction
                        ),
                    )
                    hard_losses.append(
                        torch.topk(sample, count, sorted=False).values.mean()
                    )
                empty_foreground = torch.stack(hard_losses).mean()
        else:
            empty_foreground = stable.sum() * 0.0

        total = (
            self.foreground_dice_weight * foreground_dice
            + self.foreground_focal_weight * foreground_focal
            + self.conditional_subtype_weight * conditional_subtype
            + self.subtype_ovr_weight * subtype_ovr
            + self.empty_foreground_weight * empty_foreground
        )
        return {
            "loss": total,
            "dice": foreground_dice,
            "focal": foreground_focal,
            "empty_foreground": empty_foreground,
            "foreground_dice": foreground_dice,
            "foreground_focal": foreground_focal,
            "conditional_subtype": conditional_subtype,
            "subtype_ovr": subtype_ovr,
        }

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        supervision: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(logits, target, supervision)["loss"]


def conditional_subtype_loss_components(
    subtype_logits: torch.Tensor,
    incumbent_mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    foreground_class_weights: torch.Tensor | None = None,
    stability_weight: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Train foreground subtype decisions without changing hemorrhage support.

    Spatially known true-foreground pixels already supported by the incumbent
    receive a five-way class-weighted cross-entropy objective.  Incumbent
    foreground pixels outside that supervised set (known background or rows
    without spatial labels) receive a smaller hard-distillation objective so
    the refiner does not freely redistribute existing false-positive volume
    across subtypes.  No background/foreground objective is present because the
    two-stage model locks that decision by construction.
    """
    if subtype_logits.ndim != 4 or subtype_logits.shape[1] != 6:
        raise ValueError("Conditional subtype loss expects [B, 6, H, W] logits")
    if incumbent_mask_logits.shape != subtype_logits.shape:
        raise ValueError("Incumbent logits must match conditional subtype logits")
    if masks.ndim == subtype_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("Conditional subtype masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (subtype_logits.shape[0], *subtype_logits.shape[-2:]):
        raise ValueError("Conditional subtype masks are incompatible with logits")
    if segmentation_known.numel() != subtype_logits.shape[0]:
        raise ValueError("Conditional subtype supervision flags are incompatible")
    if stability_weight < 0:
        raise ValueError("stability_weight must be non-negative")
    weights = None
    if foreground_class_weights is not None:
        if foreground_class_weights.numel() != 5:
            raise ValueError("Conditional subtype class weights must contain five values")
        weights = foreground_class_weights.reshape(5).to(
            device=subtype_logits.device, dtype=torch.float32
        )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Conditional subtype class weights must be finite and positive")

    incumbent_prediction = incumbent_mask_logits.detach().argmax(dim=1)
    incumbent_foreground = incumbent_prediction > 0
    known = (segmentation_known.reshape(-1) > 0.5)[:, None, None]
    supervised_pixels = known & incumbent_foreground & (masks > 0)
    stability_pixels = incumbent_foreground & ~supervised_pixels
    foreground_logits = subtype_logits.float()[:, 1:].permute(0, 2, 3, 1)
    zero = subtype_logits.float().sum() * 0.0

    if torch.any(supervised_pixels):
        supervised = F.cross_entropy(
            foreground_logits[supervised_pixels],
            masks[supervised_pixels].long() - 1,
            weight=weights,
        )
    else:
        supervised = zero
    if stability_weight > 0 and torch.any(stability_pixels):
        stability = F.cross_entropy(
            foreground_logits[stability_pixels],
            incumbent_prediction[stability_pixels].long() - 1,
        )
    else:
        stability = zero
    return {
        "loss": supervised + float(stability_weight) * stability,
        "supervised": supervised,
        "stability": stability,
    }


def conditional_subtype_correction_loss_components(
    subtype_logits: torch.Tensor,
    incumbent_mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    correction_class_weights: torch.Tensor | None = None,
    stability_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Correct incumbent subtype errors while preserving all other decisions.

    Ground-truth cross-entropy is applied only where the incumbent already
    predicts foreground but assigns the wrong foreground subtype.  Everywhere
    else inside incumbent foreground, a soft KL teacher keeps the copied
    decoder/head at the incumbent distribution.  The KL term is exactly zero
    at initialization, unlike hard-label distillation, so batches without a
    correction target cannot sharpen or otherwise drift the incumbent output.

    Foreground/background support remains outside this loss and is locked by
    :class:`ConditionalSubtypeRefinementModel`.
    """
    if subtype_logits.ndim != 4 or subtype_logits.shape[1] != 6:
        raise ValueError("Conditional correction loss expects [B, 6, H, W] logits")
    if incumbent_mask_logits.shape != subtype_logits.shape:
        raise ValueError("Incumbent logits must match conditional subtype logits")
    if masks.ndim == subtype_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("Conditional correction masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (subtype_logits.shape[0], *subtype_logits.shape[-2:]):
        raise ValueError("Conditional correction masks are incompatible with logits")
    if segmentation_known.numel() != subtype_logits.shape[0]:
        raise ValueError("Conditional correction supervision flags are incompatible")
    if stability_weight < 0:
        raise ValueError("stability_weight must be non-negative")

    weights = None
    if correction_class_weights is not None:
        if correction_class_weights.numel() != 5:
            raise ValueError("Conditional correction class weights must contain five values")
        weights = correction_class_weights.reshape(5).to(
            device=subtype_logits.device, dtype=torch.float32
        )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError(
                "Conditional correction class weights must be finite and positive"
            )

    incumbent_logits = incumbent_mask_logits.detach().float()[:, 1:]
    candidate_logits = subtype_logits.float()[:, 1:]
    incumbent_prediction = incumbent_mask_logits.detach().argmax(dim=1)
    incumbent_foreground = incumbent_prediction > 0
    known = (segmentation_known.reshape(-1) > 0.5)[:, None, None]
    supported_true_foreground = known & incumbent_foreground & (masks > 0)
    correction_pixels = supported_true_foreground & (
        incumbent_prediction != masks
    )
    stability_pixels = incumbent_foreground & ~correction_pixels
    candidate_last = candidate_logits.permute(0, 2, 3, 1)
    incumbent_last = incumbent_logits.permute(0, 2, 3, 1)
    zero = candidate_logits.sum() * 0.0

    if torch.any(correction_pixels):
        correction = F.cross_entropy(
            candidate_last[correction_pixels],
            masks[correction_pixels].long() - 1,
            weight=weights,
        )
    else:
        correction = zero
    if stability_weight > 0 and torch.any(stability_pixels):
        teacher = F.softmax(incumbent_last[stability_pixels], dim=-1)
        student = F.log_softmax(candidate_last[stability_pixels], dim=-1)
        stability = F.kl_div(student, teacher, reduction="batchmean")
    else:
        stability = zero
    return {
        "loss": correction + float(stability_weight) * stability,
        "correction": correction,
        "stability": stability,
        "correction_pixel_count": correction_pixels.sum(),
        "stability_pixel_count": stability_pixels.sum(),
    }


def conditional_subtype_population_loss_components(
    subtype_logits: torch.Tensor,
    incumbent_mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    correction_class_weights: torch.Tensor | None = None,
    correction_weight: float = 4.0,
    stability_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Use one population denominator for correction and preservation.

    Exp69 averaged its rare correction pixels and abundant preservation pixels
    independently, so a small error set received the same component mass as
    millions of incumbent-correct pixels.  Here both terms are summed and
    divided by the total incumbent-foreground population.  Their influence
    therefore follows their observed prevalence while soft KL still provides
    an identity-preserving trust region.
    """
    if subtype_logits.ndim != 4 or subtype_logits.shape[1] != 6:
        raise ValueError("Population subtype loss expects [B, 6, H, W] logits")
    if incumbent_mask_logits.shape != subtype_logits.shape:
        raise ValueError("Incumbent logits must match population subtype logits")
    if masks.ndim == subtype_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("Population subtype masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (subtype_logits.shape[0], *subtype_logits.shape[-2:]):
        raise ValueError("Population subtype masks are incompatible with logits")
    if segmentation_known.numel() != subtype_logits.shape[0]:
        raise ValueError("Population subtype supervision flags are incompatible")
    if correction_weight < 0 or stability_weight < 0:
        raise ValueError("Population subtype weights must be non-negative")

    weights = None
    if correction_class_weights is not None:
        if correction_class_weights.numel() != 5:
            raise ValueError("Population correction class weights need five values")
        weights = correction_class_weights.reshape(5).to(
            device=subtype_logits.device, dtype=torch.float32
        )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError(
                "Population correction class weights must be finite and positive"
            )

    incumbent_foreground_logits = incumbent_mask_logits.detach().float()[:, 1:]
    candidate_foreground_logits = subtype_logits.float()[:, 1:]
    incumbent_prediction = incumbent_mask_logits.detach().argmax(dim=1)
    incumbent_foreground = incumbent_prediction > 0
    known = (segmentation_known.reshape(-1) > 0.5)[:, None, None]
    supported_true_foreground = known & incumbent_foreground & (masks > 0)
    correction_pixels = supported_true_foreground & (incumbent_prediction != masks)
    stability_pixels = incumbent_foreground & ~correction_pixels
    candidate_last = candidate_foreground_logits.permute(0, 2, 3, 1)
    incumbent_last = incumbent_foreground_logits.permute(0, 2, 3, 1)
    zero = candidate_foreground_logits.sum() * 0.0
    denominator = incumbent_foreground.sum().clamp_min(1).to(
        candidate_foreground_logits.dtype
    )

    if torch.any(correction_pixels):
        correction_per_pixel = F.cross_entropy(
            candidate_last[correction_pixels],
            masks[correction_pixels].long() - 1,
            weight=weights,
            reduction="none",
        )
        correction_sum = correction_per_pixel.sum()
        correction_mean = correction_per_pixel.mean()
    else:
        correction_sum = zero
        correction_mean = zero

    if stability_weight > 0 and torch.any(stability_pixels):
        teacher = F.softmax(incumbent_last[stability_pixels], dim=-1)
        student = F.log_softmax(candidate_last[stability_pixels], dim=-1)
        stability_per_pixel = F.kl_div(
            student,
            teacher,
            reduction="none",
        ).sum(dim=-1)
        stability_sum = stability_per_pixel.sum()
        stability_mean = stability_per_pixel.mean()
    else:
        stability_sum = zero
        stability_mean = zero

    correction_population = correction_sum / denominator
    stability_population = stability_sum / denominator
    loss = (
        float(correction_weight) * correction_population
        + float(stability_weight) * stability_population
    )
    return {
        "loss": loss,
        "correction": correction_mean,
        "stability": stability_mean,
        "correction_population": correction_population,
        "stability_population": stability_population,
        "correction_pixel_count": correction_pixels.sum(),
        "stability_pixel_count": stability_pixels.sum(),
        "population_pixel_count": incumbent_foreground.sum(),
    }


def conditional_subtype_selective_loss_components(
    subtype_logits: torch.Tensor,
    selection_gate_logits: torch.Tensor,
    incumbent_mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    correction_class_weights: torch.Tensor | None = None,
    correction_weight: float = 4.0,
    stability_weight: float = 1.0,
    gate_weight: float = 0.25,
    gate_positive_weight: float = 200.0,
) -> dict[str, torch.Tensor]:
    """Jointly learn subtype correction and whether to accept that correction."""
    if selection_gate_logits.shape != (
        subtype_logits.shape[0],
        1,
        *subtype_logits.shape[-2:],
    ):
        raise ValueError("Selection gate logits are incompatible with subtype logits")
    if gate_weight < 0 or gate_positive_weight <= 0:
        raise ValueError("Selection gate weights must be positive")

    components = conditional_subtype_population_loss_components(
        subtype_logits,
        incumbent_mask_logits,
        masks,
        segmentation_known,
        correction_class_weights=correction_class_weights,
        correction_weight=correction_weight,
        stability_weight=stability_weight,
    )
    if masks.ndim == subtype_logits.ndim:
        masks = masks.squeeze(1)
    incumbent_prediction = incumbent_mask_logits.detach().argmax(dim=1)
    incumbent_foreground = incumbent_prediction > 0
    known = (segmentation_known.reshape(-1) > 0.5)[:, None, None]
    supported_true_foreground = known & incumbent_foreground & (masks > 0)
    correction_pixels = supported_true_foreground & (incumbent_prediction != masks)
    gate_targets = correction_pixels.to(selection_gate_logits.dtype)
    gate_logits = selection_gate_logits.float().squeeze(1)
    zero = gate_logits.sum() * 0.0
    if torch.any(incumbent_foreground):
        gate_per_pixel = F.binary_cross_entropy_with_logits(
            gate_logits[incumbent_foreground],
            gate_targets[incumbent_foreground].float(),
            pos_weight=torch.tensor(
                float(gate_positive_weight),
                device=gate_logits.device,
                dtype=gate_logits.dtype,
            ),
            reduction="none",
        )
        gate = gate_per_pixel.mean()
    else:
        gate = zero
    components["gate"] = gate
    components["loss"] = components["loss"] + float(gate_weight) * gate
    components["gate_positive_pixel_count"] = correction_pixels.sum()
    components["gate_negative_pixel_count"] = (
        incumbent_foreground & ~correction_pixels
    ).sum()
    return components


def _positive_tversky_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    class_ids: tuple[int, ...],
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Recall-weighted overlap for selected positive-only subtype rows."""
    if mask_logits.ndim != 4 or mask_logits.shape[1] != 6:
        raise ValueError("Diffuse Tversky loss expects [B, 6, H, W] mask logits")
    if masks.ndim == mask_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("Diffuse Tversky masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (mask_logits.shape[0], *mask_logits.shape[-2:]):
        raise ValueError("Diffuse Tversky masks are incompatible with logits")
    if segmentation_known.numel() != mask_logits.shape[0]:
        raise ValueError("Diffuse Tversky supervision flags are incompatible")
    if alpha < 0 or beta < 0 or alpha + beta <= 0:
        raise ValueError("Diffuse Tversky alpha/beta must be non-negative")
    if smooth <= 0:
        raise ValueError("Diffuse Tversky smooth must be positive")
    if not class_ids or any(class_id < 1 or class_id >= 6 for class_id in class_ids):
        raise ValueError("Tversky class_ids must contain foreground classes")

    known_rows = segmentation_known.reshape(-1) > 0.5
    if not torch.any(known_rows):
        return mask_logits.float().sum() * 0.0
    selected_logits = mask_logits[known_rows].float()
    selected_masks = masks[known_rows].long()
    probabilities = torch.softmax(selected_logits, dim=1)
    class_losses = []
    for class_id in class_ids:
        target = selected_masks == class_id
        positive_rows = target.flatten(start_dim=1).any(dim=1)
        if not torch.any(positive_rows):
            continue
        probability = probabilities[positive_rows, class_id]
        target = target[positive_rows].float()
        true_positive = (probability * target).sum()
        false_positive = (probability * (1.0 - target)).sum()
        false_negative = ((1.0 - probability) * target).sum()
        score = (true_positive + smooth) / (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + smooth
        )
        class_losses.append(1.0 - score)
    if not class_losses:
        return mask_logits.float().sum() * 0.0
    return torch.stack(class_losses).mean()


def positive_diffuse_tversky_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Recall-weighted SDH/SAH overlap on positive subtype rows only."""
    return _positive_tversky_loss(
        mask_logits,
        masks,
        segmentation_known,
        class_ids=DIFFUSE_HEMORRHAGE_CLASS_IDS,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
    )


def positive_sah_tversky_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Recall-weighted SAH overlap without introducing an SDH gradient."""
    return _positive_tversky_loss(
        mask_logits,
        masks,
        segmentation_known,
        class_ids=(SAH_CLASS_ID,),
        alpha=alpha,
        beta=beta,
        smooth=smooth,
    )


def positive_sah_pixel_nll_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
) -> torch.Tensor:
    """Increase SAH probability only at spatially known true-SAH pixels.

    Unlike the row-positive Tversky objective, this term contains no
    false-positive contribution from background pixels.  The ordinary
    segmentation objective remains responsible for false-positive control.
    """
    if mask_logits.ndim != 4 or mask_logits.shape[1] != 6:
        raise ValueError("SAH positive-pixel NLL expects [B, 6, H, W] logits")
    if masks.ndim == mask_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("SAH positive-pixel masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (mask_logits.shape[0], *mask_logits.shape[-2:]):
        raise ValueError("SAH positive-pixel masks are incompatible with logits")
    if segmentation_known.numel() != mask_logits.shape[0]:
        raise ValueError("SAH positive-pixel supervision flags are incompatible")
    known = (segmentation_known.reshape(-1) > 0.5)[:, None, None]
    positive_pixels = known & (masks == SAH_CLASS_ID)
    if not torch.any(positive_pixels):
        return mask_logits.float().sum() * 0.0
    sah_log_probability = F.log_softmax(mask_logits.float(), dim=1)[:, SAH_CLASS_ID]
    return -sah_log_probability[positive_pixels].mean()


def soft_physical_volume_components(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    segmentation_known: torch.Tensor,
    voxel_volume_ml: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Align differentiable subtype and total slice volumes in physical mL.

    A log transform prevents a few large haemorrhages from dominating small
    lesions. Only spatially supervised rows participate; classification-only
    rows must never be interpreted as empty masks.
    """
    if mask_logits.ndim != 4 or mask_logits.shape[1] != 6:
        raise ValueError("Physical-volume loss expects [B, 6, H, W] mask logits")
    if masks.ndim == mask_logits.ndim:
        if masks.shape[1] != 1:
            raise ValueError("Physical-volume masks need one channel")
        masks = masks.squeeze(1)
    if masks.shape != (mask_logits.shape[0], *mask_logits.shape[-2:]):
        raise ValueError("Physical-volume masks are incompatible with logits")
    if segmentation_known.numel() != mask_logits.shape[0]:
        raise ValueError(
            "Physical-volume supervision flags are incompatible with logits"
        )
    if voxel_volume_ml.numel() != mask_logits.shape[0]:
        raise ValueError("Physical-volume voxel sizes are incompatible with logits")

    voxel_volume_ml = voxel_volume_ml.reshape(-1).to(
        device=mask_logits.device, dtype=torch.float32
    )
    if not torch.isfinite(voxel_volume_ml).all() or torch.any(voxel_volume_ml <= 0):
        raise ValueError("Physical-volume voxel sizes must be finite and positive")
    segmentation_rows = segmentation_known.reshape(-1) > 0.5
    if not torch.any(segmentation_rows):
        zero = mask_logits.float().sum() * 0.0
        return {"loss": zero, "subtype": zero, "total": zero}

    selected_logits = mask_logits[segmentation_rows].float()
    selected_masks = masks[segmentation_rows].long()
    selected_voxels = voxel_volume_ml[segmentation_rows, None]
    foreground_probabilities = torch.softmax(selected_logits, dim=1)[:, 1:]
    predicted_ml = foreground_probabilities.sum(dim=(-2, -1)) * selected_voxels
    target_ml = torch.stack(
        [
            (selected_masks == class_id).sum(dim=(-2, -1))
            for class_id in range(1, 6)
        ],
        dim=1,
    ).float() * selected_voxels

    predicted_log_ml = torch.log1p(predicted_ml)
    target_log_ml = torch.log1p(target_ml)
    subtype = F.smooth_l1_loss(
        predicted_log_ml, target_log_ml, beta=0.5, reduction="mean"
    )
    total = F.smooth_l1_loss(
        torch.log1p(predicted_ml.sum(dim=1)),
        torch.log1p(target_ml.sum(dim=1)),
        beta=0.5,
        reduction="mean",
    )
    return {"loss": 0.5 * (subtype + total), "subtype": subtype, "total": total}


class ICH25DSegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        classification_pos_weight: torch.Tensor,
        segmentation_class_weights: torch.Tensor | None = None,
        segmentation_class_counts: torch.Tensor | None = None,
        segmentation_weight: float = 1.0,
        classification_weight: float = 0.25,
        classification_focal_gamma: float = 1.0,
        background_weight: float = 0.15,
        empty_foreground_weight: float = 0.0,
        empty_foreground_top_fraction: float = 1.0,
        ivh_center_loss_weight: float = 0.0,
        physical_volume_loss_weight: float = 0.0,
        diffuse_tversky_loss_weight: float = 0.0,
        sah_tversky_loss_weight: float = 0.0,
        sah_positive_pixel_loss_weight: float = 0.0,
        segmentation_objective: str = "multiclass",
        foreground_dice_weight: float = 0.40,
        foreground_focal_weight: float = 0.20,
        conditional_subtype_weight: float = 0.30,
        subtype_ovr_weight: float = 0.10,
        conditional_subtype_mode: str = "cross_entropy",
    ) -> None:
        super().__init__()
        if segmentation_weight <= 0 or classification_weight < 0:
            raise ValueError("Invalid multi-task loss weights")
        if classification_focal_gamma < 0:
            raise ValueError("classification_focal_gamma must be non-negative")
        if ivh_center_loss_weight < 0:
            raise ValueError("ivh_center_loss_weight must be non-negative")
        if physical_volume_loss_weight < 0:
            raise ValueError("physical_volume_loss_weight must be non-negative")
        if diffuse_tversky_loss_weight < 0:
            raise ValueError("diffuse_tversky_loss_weight must be non-negative")
        if sah_tversky_loss_weight < 0:
            raise ValueError("sah_tversky_loss_weight must be non-negative")
        if sah_positive_pixel_loss_weight < 0:
            raise ValueError("sah_positive_pixel_loss_weight must be non-negative")
        if segmentation_objective not in SEGMENTATION_OBJECTIVES:
            raise ValueError(
                "segmentation_objective must be one of: "
                f"{', '.join(SEGMENTATION_OBJECTIVES)}"
            )
        self.segmentation_objective = segmentation_objective
        if segmentation_objective == "multiclass":
            self.segmentation = MaskedDiceFocalLoss(
                num_classes=6,
                dice_weight=0.65,
                focal_weight=0.35,
                focal_gamma=2.0,
                background_weight=background_weight,
                foreground_weights=segmentation_class_weights,
                empty_foreground_weight=empty_foreground_weight,
                empty_foreground_top_fraction=empty_foreground_top_fraction,
            )
        else:
            self.segmentation = HierarchicalForegroundSubtypeLoss(
                foreground_class_weights=segmentation_class_weights,
                foreground_class_counts=segmentation_class_counts,
                conditional_subtype_mode=conditional_subtype_mode,
                foreground_dice_weight=foreground_dice_weight,
                foreground_focal_weight=foreground_focal_weight,
                conditional_subtype_weight=conditional_subtype_weight,
                subtype_ovr_weight=subtype_ovr_weight,
                background_weight=background_weight,
                empty_foreground_weight=empty_foreground_weight,
                empty_foreground_top_fraction=empty_foreground_top_fraction,
            )
        self.segmentation_weight = float(segmentation_weight)
        self.classification_weight = float(classification_weight)
        self.classification_focal_gamma = float(classification_focal_gamma)
        self.ivh_center_loss_weight = float(ivh_center_loss_weight)
        self.physical_volume_loss_weight = float(physical_volume_loss_weight)
        self.diffuse_tversky_loss_weight = float(diffuse_tversky_loss_weight)
        self.sah_tversky_loss_weight = float(sah_tversky_loss_weight)
        self.sah_positive_pixel_loss_weight = float(sah_positive_pixel_loss_weight)
        self.register_buffer(
            "classification_pos_weight",
            classification_pos_weight.detach().float().clone(),
        )

    def components(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        masks: torch.Tensor,
        class_targets: torch.Tensor,
        segmentation_known: torch.Tensor,
        classification_known: torch.Tensor | None = None,
        ivh_center_targets: torch.Tensor | None = None,
        voxel_volume_ml: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        segmentation_rows = segmentation_known > 0.5
        classification_rows = (
            segmentation_rows
            if classification_known is None
            else classification_known > 0.5
        )
        if torch.any(segmentation_rows):
            supervision = segmentation_rows[:, None, None].expand_as(masks)
            segmentation = self.segmentation.components(
                mask_logits, masks, supervision
            )
        else:
            zero = mask_logits.float().sum() * 0.0
            segmentation = {
                "loss": zero,
                "dice": zero,
                "focal": zero,
                "empty_foreground": zero,
                "foreground_dice": zero,
                "foreground_focal": zero,
                "conditional_subtype": zero,
                "subtype_ovr": zero,
            }
        for name in (
            "foreground_dice",
            "foreground_focal",
            "conditional_subtype",
            "subtype_ovr",
        ):
            segmentation.setdefault(name, mask_logits.float().sum() * 0.0)
        if not torch.any(classification_rows):
            raise ValueError("Multi-task batch contains no classification supervision")

        selected_logits = class_logits[classification_rows].float()
        selected_targets = class_targets[classification_rows].float()
        bce = F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_targets,
            pos_weight=self.classification_pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(selected_logits)
        correct_probability = (
            selected_targets * probabilities
            + (1.0 - selected_targets) * (1.0 - probabilities)
        )
        classification = (
            bce * (1.0 - correct_probability).pow(self.classification_focal_gamma)
        ).mean()
        if self.ivh_center_loss_weight > 0:
            if ivh_center_targets is None:
                raise ValueError(
                    "ivh_center_targets are required when IVH center loss is enabled"
                )
            if hasattr(ivh_center_targets, "as_tensor"):
                ivh_center_targets = ivh_center_targets.as_tensor()
            if ivh_center_targets.ndim == mask_logits.ndim:
                if ivh_center_targets.shape[1] != 1:
                    raise ValueError("IVH center targets need one channel")
                ivh_center_targets = ivh_center_targets.squeeze(1)
            if ivh_center_targets.shape != masks.shape:
                raise ValueError("IVH center targets are incompatible with masks")
            center_pixels = (ivh_center_targets > 0.5) & segmentation_rows.reshape(
                (-1,) + (1,) * (masks.ndim - 1)
            )
            if torch.any(center_pixels):
                ivh_log_probability = F.log_softmax(
                    mask_logits.float(), dim=1
                )[:, 1]
                ivh_center = -ivh_log_probability[center_pixels].mean()
            else:
                ivh_center = mask_logits.float().sum() * 0.0
        else:
            ivh_center = mask_logits.float().sum() * 0.0
        if self.physical_volume_loss_weight > 0:
            if voxel_volume_ml is None:
                raise ValueError(
                    "voxel_volume_ml is required when physical-volume loss is enabled"
                )
            physical_volume = soft_physical_volume_components(
                mask_logits,
                masks,
                segmentation_known,
                voxel_volume_ml,
            )
        else:
            zero = mask_logits.float().sum() * 0.0
            physical_volume = {"loss": zero, "subtype": zero, "total": zero}
        if self.diffuse_tversky_loss_weight > 0:
            diffuse_tversky = positive_diffuse_tversky_loss(
                mask_logits,
                masks,
                segmentation_known,
            )
        else:
            diffuse_tversky = mask_logits.float().sum() * 0.0
        if self.sah_tversky_loss_weight > 0:
            sah_tversky = positive_sah_tversky_loss(
                mask_logits,
                masks,
                segmentation_known,
            )
        else:
            sah_tversky = mask_logits.float().sum() * 0.0
        if self.sah_positive_pixel_loss_weight > 0:
            sah_positive_pixel = positive_sah_pixel_nll_loss(
                mask_logits,
                masks,
                segmentation_known,
            )
        else:
            sah_positive_pixel = mask_logits.float().sum() * 0.0
        total = (
            self.segmentation_weight * segmentation["loss"]
            + self.classification_weight * classification
            + self.ivh_center_loss_weight * ivh_center
            + self.physical_volume_loss_weight * physical_volume["loss"]
            + self.diffuse_tversky_loss_weight * diffuse_tversky
            + self.sah_tversky_loss_weight * sah_tversky
            + self.sah_positive_pixel_loss_weight * sah_positive_pixel
        )
        return {
            "loss": total,
            "segmentation": segmentation["loss"],
            "dice": segmentation["dice"],
            "focal": segmentation["focal"],
            "empty_foreground": segmentation["empty_foreground"],
            "foreground_dice": segmentation["foreground_dice"],
            "foreground_focal": segmentation["foreground_focal"],
            "conditional_subtype": segmentation["conditional_subtype"],
            "subtype_ovr": segmentation["subtype_ovr"],
            "classification": classification,
            "ivh_center": ivh_center,
            "physical_volume": physical_volume["loss"],
            "physical_volume_subtype": physical_volume["subtype"],
            "physical_volume_total": physical_volume["total"],
            "diffuse_tversky": diffuse_tversky,
            "sah_tversky": sah_tversky,
            "sah_positive_pixel": sah_positive_pixel,
        }

    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        masks: torch.Tensor,
        class_targets: torch.Tensor,
        segmentation_known: torch.Tensor,
        classification_known: torch.Tensor | None = None,
        ivh_center_targets: torch.Tensor | None = None,
        voxel_volume_ml: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.components(
            mask_logits,
            class_logits,
            masks,
            class_targets,
            segmentation_known,
            classification_known,
            ivh_center_targets,
            voxel_volume_ml,
        )["loss"]
