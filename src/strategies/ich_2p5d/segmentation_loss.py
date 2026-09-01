"""Multi-task segmentation and slice-subtype loss for the 2.5D ICH model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.strategies.ich_v2.losses import MaskedDiceFocalLoss


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
    ) -> None:
        super().__init__()
        if segmentation_weight <= 0 or classification_weight < 0:
            raise ValueError("Invalid multi-task loss weights")
        if classification_focal_gamma < 0:
            raise ValueError("classification_focal_gamma must be non-negative")
        if ivh_center_loss_weight < 0:
            raise ValueError("ivh_center_loss_weight must be non-negative")
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
        total = (
            self.segmentation_weight * segmentation["loss"]
            + self.classification_weight * classification
            + self.ivh_center_loss_weight * ivh_center
        )
        return {
            "loss": total,
            "segmentation": segmentation["loss"],
            "dice": segmentation["dice"],
            "focal": segmentation["focal"],
            "empty_foreground": segmentation["empty_foreground"],
            "classification": classification,
            "ivh_center": ivh_center,
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
    ) -> torch.Tensor:
        return self.components(
            mask_logits,
            class_logits,
            masks,
            class_targets,
            segmentation_known,
            classification_known,
            ivh_center_targets,
        )["loss"]
