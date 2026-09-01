"""Multi-task segmentation and slice-subtype loss for the 2.5D ICH model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.strategies.ich_v2.losses import MaskedDiceFocalLoss


DIFFUSE_HEMORRHAGE_CLASS_IDS = (3, 5)  # SDH and SAH
SAH_CLASS_ID = 5


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
            }
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
