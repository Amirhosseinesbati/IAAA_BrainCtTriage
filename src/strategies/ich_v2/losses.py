"""Losses that never learn from unknown ICH voxels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDiceFocalLoss(nn.Module):
    """Foreground Dice plus focal CE over explicitly supervised voxels.

    Empty-foreground patches contribute only focal background loss.  This is
    important because 140 clean-negative studies are real supervision, while
    missing JSON slices in annotated studies must contribute no gradient.
    """

    def __init__(
        self,
        num_classes: int = 6,
        *,
        dice_weight: float = 0.6,
        focal_weight: float = 0.4,
        focal_gamma: float = 2.0,
        background_weight: float = 0.2,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("MaskedDiceFocalLoss requires at least two classes")
        if dice_weight < 0 or focal_weight < 0 or dice_weight + focal_weight <= 0:
            raise ValueError("Loss weights must be non-negative with a positive sum")
        if background_weight <= 0:
            raise ValueError("background_weight must be positive")
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)
        weights = torch.ones(self.num_classes, dtype=torch.float32)
        weights[0] = float(background_weight)
        self.register_buffer("class_weights", weights)

    @staticmethod
    def _squeeze_channel(value: torch.Tensor, logits: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == logits.ndim and value.shape[1] == 1:
            value = value.squeeze(1)
        if value.ndim != logits.ndim - 1:
            raise ValueError(
                f"{name} must have {logits.ndim - 1} dimensions for logits "
                f"{tuple(logits.shape)}, got {tuple(value.shape)}"
            )
        return value

    def components(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        supervision: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # MONAI propagates per-sample affine metadata through model outputs.
        # Loss arithmetic must operate on plain tensors because slicing a
        # multi-crop MetaTensor tries to collate heterogeneous transform traces.
        if hasattr(logits, "as_tensor"):
            logits = logits.as_tensor()
        if hasattr(target, "as_tensor"):
            target = target.as_tensor()
        if hasattr(supervision, "as_tensor"):
            supervision = supervision.as_tensor()
        target = self._squeeze_channel(target, logits, "target").long()
        supervision = self._squeeze_channel(supervision, logits, "supervision")
        valid = supervision > 0.5
        if not torch.any(valid):
            raise ValueError("Batch contains no supervised voxels")
        if target.min() < 0 or target.max() >= self.num_classes:
            raise ValueError("Target contains an invalid ICH class")

        stable_logits = logits.float()
        log_probs = F.log_softmax(stable_logits, dim=1)
        probs = log_probs.exp()
        target_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_log_prob = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        voxel_weights = self.class_weights[target]
        focal = -((1.0 - target_prob).pow(self.focal_gamma) * target_log_prob)
        focal = (focal[valid] * voxel_weights[valid]).sum() / voxel_weights[valid].sum().clamp_min(1e-8)

        one_hot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        valid_channel = valid.unsqueeze(1).float()
        dims = (0, *range(2, logits.ndim))
        intersection = (probs * one_hot * valid_channel).sum(dim=dims)
        predicted = (probs * valid_channel).sum(dim=dims)
        observed = (one_hot * valid_channel).sum(dim=dims)
        present_foreground = observed[1:] > 0
        if torch.any(present_foreground):
            dice_scores = (
                2.0 * intersection[1:] + self.smooth
            ) / (predicted[1:] + observed[1:] + self.smooth)
            dice = 1.0 - dice_scores[present_foreground].mean()
        else:
            dice = stable_logits.sum() * 0.0

        total = self.dice_weight * dice + self.focal_weight * focal
        return {"loss": total, "dice": dice, "focal": focal}

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        supervision: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(logits, target, supervision)["loss"]
