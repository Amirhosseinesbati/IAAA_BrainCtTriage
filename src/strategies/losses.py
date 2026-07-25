"""
losses.py — Custom loss functions and composite loss builder.

Provides:
- ``FocalLoss`` — standard multi-class focal loss
- ``TverskyLoss`` — asymmetric similarity loss (tunable FP/FN penalty)
- ``build_composite_loss()`` — factory for weighted combinations

All losses follow the same interface as ``smp.losses``::

    loss = loss_fn(logits, target)   # logits: (B, C, H, W), target: (B, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.strategies.loss_config import LossConfig


# ═════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═════════════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Formula::

        FL(p_t) = -α * (1 - p_t)^γ * log(p_t)

    where ``p_t`` is the softmax probability of the ground-truth class.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values down-weight easy examples.
    alpha : float
        Balancing weight per class (``None`` = uniform).
    reduction : {"mean", "sum", "none"}
    ignore_index : int or None
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, H, W) raw logits (before softmax)
        target : (B, H, W) integer class indices
        """
        num_classes = logits.shape[1]

        # Softmax over class dim
        probs = F.softmax(logits, dim=1)                 # (B, C, H, W)

        # Gather probabilities of the target class
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()                   # (B, C, H, W)
        p_t = (probs * target_one_hot).sum(dim=1)        # (B, H, W)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Cross-entropy part: -log(p_t)
        ce = -torch.log(p_t.clamp(min=1e-8))             # (B, H, W)

        loss = focal_weight * ce                          # (B, H, W)

        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
            alpha_factor = alpha_t.sum(dim=1)
            loss = alpha_factor * loss

        # Ignore index mask
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            loss = loss * mask.float()

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═════════════════════════════════════════════════════════════════════════
# Tversky Loss
# ═════════════════════════════════════════════════════════════════════════


class TverskyLoss(nn.Module):
    """
    Multi-class Tversky Loss.

    The Tversky index is an asymmetric similarity measure::

        TI_c = TP_c / (TP_c + α * FP_c + β * FN_c)

    and the loss per class is ``1 - TI_c``.

    Parameters
    ----------
    alpha : float
        Weight for false positives (default 0.7 = penalise FP more than FN).
    beta : float
        Weight for false negatives (default 0.3).
    smooth : float
        Smoothing factor to avoid division by zero.
    reduction : {"mean", "sum"}
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, H, W) raw logits
        target : (B, H, W) integer class indices (0 = background, 1..C-1)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                  # (B, C, H, W)

        # One-hot target
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()                    # (B, C, H, W)

        # Per-class Tversky
        dims = (0, 2, 3)  # sum over batch, height, width
        tp = (probs * target_one_hot).sum(dim=dims)       # (C,)
        fp = (probs * (1 - target_one_hot)).sum(dim=dims)  # (C,)
        fn = ((1 - probs) * target_one_hot).sum(dim=dims)  # (C,)

        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss_per_class = 1.0 - tversky_idx                # (C,)

        if self.reduction == "mean":
            return loss_per_class.mean()
        return loss_per_class.sum()


# ═════════════════════════════════════════════════════════════════════════
# Composite Loss Builder
# ═════════════════════════════════════════════════════════════════════════


def build_composite_loss(
    loss_config: LossConfig,
    num_classes: int,
) -> nn.Module:
    """
    Build a weighted composite loss function from a ``LossConfig``.

    Returns a callable ``nn.Module`` whose ``forward(logits, target)``
    computes the weighted sum of all enabled loss components.

    Usage::

        loss_fn = build_composite_loss(config.loss_config, num_classes=6)
        loss = loss_fn(logits, masks)
    """
    components: list[tuple[float, nn.Module]] = []

    if loss_config.use_dice:
        import segmentation_models_pytorch as smp
        components.append((
            loss_config.weight_dice,
            smp.losses.DiceLoss(mode="multiclass"),
        ))
    if loss_config.use_focal:
        components.append((
            loss_config.weight_focal,
            FocalLoss(gamma=2.0, alpha=0.25),
        ))
    if loss_config.use_tversky:
        components.append((
            loss_config.weight_tversky,
            TverskyLoss(alpha=0.7, beta=0.3),
        ))
    if loss_config.use_cross_entropy:
        import segmentation_models_pytorch as smp
        components.append((
            loss_config.weight_cross_entropy,
            smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1),
        ))

    if not components:
        raise ValueError(
            "No loss functions enabled in LossConfig. "
            "Enable at least one of: Dice, Focal, Tversky, CrossEntropy."
        )

    class CompositeLoss(nn.Module):
        """Weighted sum of multiple loss terms."""

        def __init__(self, components: list[tuple[float, nn.Module]]):
            super().__init__()
            self.components = components

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            total = 0.0
            for weight, loss_fn in self.components:
                total = total + weight * loss_fn(logits, target)
            return total

    return CompositeLoss(components)
