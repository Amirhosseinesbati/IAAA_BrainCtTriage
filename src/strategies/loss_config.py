"""
loss_config.py — Pydantic model for weighted combination of loss functions.

Provides a type-safe configuration model that lets users compose multiple
loss functions (Dice, Focal, Tversky, Cross-Entropy) with arbitrary weights.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    """
    Weighted combination of loss functions for ICH segmentation.

    Each loss type has a boolean toggle and a weight multiplier.
    The effective loss is::

        Σ (weight_i * loss_i)   for each enabled loss i

    Default: Dice + Cross-Entropy (weights 1:1) — same as the previous
    "Combined" preset.
    """

    # ── Toggles ──────────────────────────────────────────────────
    use_dice: bool = Field(
        default=True,
        description="Enable Dice loss (region-based, handles class imbalance)",
    )
    use_focal: bool = Field(
        default=False,
        description="Enable Focal loss (focuses on hard misclassified examples)",
    )
    use_tversky: bool = Field(
        default=False,
        description=(
            "Enable Tversky loss (asymmetric similarity, tunable FP/FN "
            "penalty via alpha/beta in the loss implementation)"
        ),
    )
    use_cross_entropy: bool = Field(
        default=True,
        description="Enable Cross-Entropy loss (pixel-wise, smooth gradient)",
    )

    # ── Weights ──────────────────────────────────────────────────
    # Only effective when the corresponding toggle is True.
    weight_dice: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight multiplier for Dice loss",
    )
    weight_focal: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight multiplier for Focal loss",
    )
    weight_tversky: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight multiplier for Tversky loss",
    )
    weight_cross_entropy: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight multiplier for Cross-Entropy loss",
    )

    # ── Helper properties ───────────────────────────────────────┬───────────────
    @property
    def active_losses(self) -> list[str]:
        """Return list of enabled loss names."""
        active = []
        if self.use_dice:
            active.append("Dice")
        if self.use_focal:
            active.append("Focal")
        if self.use_tversky:
            active.append("Tversky")
        if self.use_cross_entropy:
            active.append("CrossEntropy")
        return active

    @property
    def combination_string(self) -> str:
        """
        Human-readable description of the loss composition.

        Example: ``"1.0*Dice + 1.0*CrossEntropy"``
        """
        parts = []
        name_map = {
            "Dice": ("use_dice", "weight_dice"),
            "Focal": ("use_focal", "weight_focal"),
            "Tversky": ("use_tversky", "weight_tversky"),
            "CrossEntropy": ("use_cross_entropy", "weight_cross_entropy"),
        }
        for display_name, (toggle_attr, weight_attr) in name_map.items():
            if getattr(self, toggle_attr):
                w = getattr(self, weight_attr)
                parts.append(f"{w}*{display_name}")
        return " + ".join(parts) if parts else "No loss selected"
