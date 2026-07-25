"""
augmentation_config.py — Pydantic models for per-transform augmentation config.

Each augmentation has an ``enabled`` toggle and a ``prob`` (probability),
plus optional transform-specific parameters.  Recommended defaults follow
brain-CT best-practices: moderate intensity, no large rotations, left-right
flip carries an anatomical warning for MLS tasks.

Two top-level configs are provided — one for SMP (Albumentations-based 2D)
and one for MONAI (MONAI-transforms based 3D).
"""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════
# Shared sub-models
# ═════════════════════════════════════════════════════════════════════════


class AugToggle(BaseModel):
    """Base toggle for a single augmentation."""

    enabled: bool = Field(
        default=True,
        description="Enable this augmentation",
    )
    prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying the augmentation",
    )


class AugShiftScaleRotate(AugToggle):
    """Shift, scale, and small rotation — useful for affine variation."""

    shift_limit: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Maximum fraction of shift (width/height)",
    )
    scale_limit: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Maximum scaling factor (+/-)",
    )
    rotate_limit: int = Field(
        default=10,
        ge=0,
        le=30,
        description="Maximum rotation angle in degrees (+/-)",
    )


class AugNoise(AugToggle):
    """Gaussian noise injection."""

    var_limit: Tuple[float, float] = Field(
        default=(0.0, 0.01),
        description="Variance range for Gaussian noise (lower, upper)",
    )


# ═════════════════════════════════════════════════════════════════════════
# SMP (Albumentations-based) Augmentation Config
# ═════════════════════════════════════════════════════════════════════════


class SMPAugmentationConfig(BaseModel):
    """
    Per-transform augmentation configuration for SMP 2D / 2.5D training.

    Uses Albumentations under the hood.  Every transform has a recommended
    default that can be overridden by the user.

    ⚠️  ``left_right_flip`` carries a risk for MLS (Midline Shift) tasks
    because flipping can invert the sign of the MLS measurement.  The UI
    should display a warning when this is enabled.
    """

    # ── Master switch ─────────────────────────────────────────────
    enabled: bool = Field(
        default=True,
        description="Master switch — disables all augmentations when False",
    )

    # ── Individual transforms ─────────────────────────────────────
    top_bottom_flip: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description="Random vertical flip (top-bottom). Safe for brain CT.",
    )
    left_right_flip: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description=(
            "Random horizontal flip (left-right). ⚠️  Inverts MLS sign — "
            "use with caution when MLS prediction is enabled."
        ),
    )
    rotate90: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description="Random 90° rotation. Preserves anatomy orientation.",
    )
    shift_scale_rotate: AugShiftScaleRotate = Field(
        default_factory=lambda: AugShiftScaleRotate(prob=0.5),
        description="Small affine shifts, scaling, and limited rotation (≤10°).",
    )
    brightness_contrast: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.3),
        description="Random brightness & contrast adjustment",
    )
    gauss_noise: AugNoise = Field(
        default_factory=lambda: AugNoise(prob=0.2),
        description="Gaussian noise injection for robustness",
    )
    scale_intensity: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.3),
        description="Scale pixel intensities by a random factor (simulates CT variation)",
    )
    adjust_contrast: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.3),
        description="Adjust image contrast (gamma correction style)",
    )


# ═════════════════════════════════════════════════════════════════════════
# MONAI (MONAI-transforms-based) Augmentation Config
# ═════════════════════════════════════════════════════════════════════════


class MONAIAugmentationConfig(BaseModel):
    """
    Per-transform augmentation configuration for MONAI 3D / 2.5D training.

    Uses MONAI ``Rand*`` transforms under the hood.  Each spatial axis can
    be flipped independently.
    """

    enabled: bool = Field(
        default=True,
        description="Master switch — disables all MONAI augmentations when False",
    )

    # ── Spatial flips (per axis) ──────────────────────────────────
    flip_axis_0: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description="Random flip along axis 0 (coronal / left-right). "
        "⚠️  Inverts MLS sign.",
    )
    flip_axis_1: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description="Random flip along axis 1 (sagittal / anterior-posterior). Safe.",
    )
    flip_axis_2: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.5),
        description="Random flip along axis 2 (axial / top-bottom). Safe.",
    )

    # ── Intensity transforms ──────────────────────────────────────
    scale_intensity: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.3),
        description="Random intensity scaling by a factor around 1.0",
    )
    shift_intensity: AugToggle = Field(
        default_factory=lambda: AugToggle(prob=0.3),
        description="Random intensity shift (offset)",
    )
    gaussian_noise: AugNoise = Field(
        default_factory=lambda: AugNoise(prob=0.2),
        description="Gaussian noise injection for robustness",
    )
