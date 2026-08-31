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
        segmentation_weight: float = 1.0,
        classification_weight: float = 0.25,
        classification_focal_gamma: float = 1.0,
        background_weight: float = 0.15,
    ) -> None:
        super().__init__()
        if segmentation_weight <= 0 or classification_weight < 0:
            raise ValueError("Invalid multi-task loss weights")
        if classification_focal_gamma < 0:
            raise ValueError("classification_focal_gamma must be non-negative")
        self.segmentation = MaskedDiceFocalLoss(
            num_classes=6,
            dice_weight=0.65,
            focal_weight=0.35,
            focal_gamma=2.0,
            background_weight=background_weight,
        )
        self.segmentation_weight = float(segmentation_weight)
        self.classification_weight = float(classification_weight)
        self.classification_focal_gamma = float(classification_focal_gamma)
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
        known: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_rows = known > 0.5
        if not torch.any(valid_rows):
            raise ValueError("Multi-task batch contains no supervised slices")
        supervision = valid_rows[:, None, None].expand_as(masks)
        segmentation = self.segmentation.components(
            mask_logits, masks, supervision
        )

        selected_logits = class_logits[valid_rows].float()
        selected_targets = class_targets[valid_rows].float()
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
        total = (
            self.segmentation_weight * segmentation["loss"]
            + self.classification_weight * classification
        )
        return {
            "loss": total,
            "segmentation": segmentation["loss"],
            "dice": segmentation["dice"],
            "focal": segmentation["focal"],
            "classification": classification,
        }

    def forward(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        masks: torch.Tensor,
        class_targets: torch.Tensor,
        known: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(
            mask_logits, class_logits, masks, class_targets, known
        )["loss"]
