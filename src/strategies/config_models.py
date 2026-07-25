"""
config_models.py — Pydantic configuration models for every ICH strategy.

Each model defines the trainable hyper-parameters exposed to the user.
The JSON Schema of each model is used by the Streamlit UI to dynamically
render configuration forms without hard-coding field widgets.

Adding a new field here automatically surfaces it in the UI.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.strategies.loss_config import LossConfig
from src.strategies.augmentation_config import SMPAugmentationConfig, MONAIAugmentationConfig


# ═════════════════════════════════════════════════════════════════════════
# nnU-Net Strategy Config
# ═════════════════════════════════════════════════════════════════════════

class NNUNetConfig(BaseModel):
    """Configuration for the nnU-Net v2 ICH segmentation strategy."""

    configuration: Literal["2d", "3d_fullres", "3d_lowres"] = Field(
        default="2d",
        description="nnU-Net configuration (2d for slice-based, 3d_fullres for volume-based)",
    )
    fold: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Cross-validation fold index (0-4 for 5-fold CV)",
    )
    early_stopping_patience: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Stop training if val_loss does not improve for this many epochs",
    )
    save_every: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Save and upload checkpoint every N epochs",
    )


# ═════════════════════════════════════════════════════════════════════════
# SMP (Segmentation Models PyTorch) Strategy Config
# ═════════════════════════════════════════════════════════════════════════

_SMP_ARCHITECTURES = Literal[
    "Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN",
    "PAN", "Linknet", "MAnet", "PSPNet",
]

_SMP_ENCODERS = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101",
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
    "efficientnet-b3", "efficientnet-b4",
    "timm-mobilenetv3_large_100", "timm-efficientnet-b0",
    "timm-res2net50_26w_4s",
    "mit_b0", "mit_b1", "mit_b2",  # Mix Vision Transformer
    "mobileone_s0", "mobileone_s1",
]

_SMP_OPTIMIZERS = Literal["AdamW", "Adam", "SGD"]

_SMP_SCHEDULERS = Literal["CosineAnnealing", "ReduceLROnPlateau", "OneCycleLR"]

# Retained for backward compatibility with existing saved configs / env vars.
# New code should use `loss_config` instead.
_SMP_LOSSES = Literal["Dice", "CrossEntropy", "Combined"]

_SMP_DIMENSIONS = Literal["2D", "2.5D"]


class SMPConfig(BaseModel):
    """Configuration for Segmentation Models PyTorch (SMP) strategy."""

    # ── Model architecture ─────────────────────────────────────────
    architecture: _SMP_ARCHITECTURES = Field(
        default="Unet",
        description="Model architecture (U-Net, U-Net++, DeepLabV3+, etc.)",
    )
    encoder: _SMP_ENCODERS = Field(
        default="resnet34",
        description="Encoder backbone (from timm / pretrained-models)",
    )
    encoder_weights: str = Field(
        default="imagenet",
        description="Pretrained weights: 'imagenet', 'ssl', or leave empty for random init",
    )

    # ── Model dimension ────────────────────────────────────────────
    model_dimension: _SMP_DIMENSIONS = Field(
        default="2D",
        description=(
            "Input dimension: '2D' for single-slice, '2.5D' for stacked "
            "consecutive slices as input channels"
        ),
    )
    slices_per_stack: Optional[Literal[3, 5]] = Field(
        default=3,
        description=(
            "Number of consecutive slices stacked as input channels "
            "(only used when model_dimension='2.5D')"
        ),
    )

    # ── Training hyper-parameters ──────────────────────────────────
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-1,
        description="Initial learning rate",
    )
    optimizer: _SMP_OPTIMIZERS = Field(
        default="AdamW",
        description="Optimizer algorithm",
    )
    scheduler: _SMP_SCHEDULERS = Field(
        default="CosineAnnealing",
        description="Learning rate scheduler",
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of training epochs",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Training batch size (adjust based on GPU memory)",
    )
    image_size: int = Field(
        default=512,
        ge=128,
        le=1024,
        description="Input image size in pixels (square)",
    )

    # ── Loss function (new: weighted composite) ────────────────────
    loss_config: LossConfig = Field(
        default_factory=LossConfig,
        description="Weighted combination of loss functions (Dice, Focal, Tversky, CE)",
    )

    # ── Augmentation (new: per-transform config) ───────────────────
    augmentation_config: SMPAugmentationConfig = Field(
        default_factory=SMPAugmentationConfig,
        description="Per-transform data augmentation settings",
    )

    # ── Legacy fields (kept for backward compatibility) ────────────
    augmentation: bool = Field(
        default=True,
        description=(
            "[DEPRECATED — use augmentation_config.enabled instead] "
            "Enable on-the-fly data augmentation"
        ),
    )
    loss: _SMP_LOSSES = Field(
        default="Combined",
        description=(
            "[DEPRECATED — use loss_config instead] "
            "Loss function: Dice, CrossEntropy, or Combined"
        ),
    )

    # ── Training utilities ─────────────────────────────────────────
    early_stopping_patience: int = Field(
        default=30,
        ge=5,
        le=200,
        description="Stop training if val_loss doesn't improve for N epochs",
    )
    use_amp: bool = Field(
        default=True,
        description="Use Automatic Mixed Precision (AMP) for faster training",
    )


# ═════════════════════════════════════════════════════════════════════════
# MONAI Strategy Config
# ═════════════════════════════════════════════════════════════════════════

_MONAI_MODELS = Literal["UNETR", "SwinUNETR", "SegResNet", "DynUNet"]
_MONAI_DIMENSIONS = Literal["2.5D", "3D"]


class MONAIConfig(BaseModel):
    """Configuration for MONAI-based segmentation strategy."""

    # ── Model ──────────────────────────────────────────────────────
    model: _MONAI_MODELS = Field(
        default="UNETR",
        description="MONAI network architecture for 3D segmentation",
    )

    # ── Model dimension ────────────────────────────────────────────
    model_dimension: _MONAI_DIMENSIONS = Field(
        default="3D",
        description=(
            "Input dimension: '3D' for full-volume patches, "
            "'2.5D' for small-depth stacked slices"
        ),
    )
    slices_per_stack: Optional[Literal[3, 5]] = Field(
        default=3,
        description=(
            "Number of consecutive slices in the depth dimension "
            "(only used when model_dimension='2.5D')"
        ),
    )

    # ── Training hyper-parameters ──────────────────────────────────
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-1,
        description="Initial learning rate",
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of training epochs",
    )
    batch_size: int = Field(
        default=2,
        ge=1,
        le=16,
        description="3D batch size (keep small — 3D volumes are memory-heavy)",
    )
    roi_size: int = Field(
        default=128,
        ge=64,
        le=256,
        description="Crop/patch size for 3D volumes (cubic)",
    )
    val_split: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Fraction of data reserved for validation",
    )

    # ── Loss function (new: weighted composite) ────────────────────
    loss_config: LossConfig = Field(
        default_factory=LossConfig,
        description="Weighted combination of loss functions (Dice, Focal, Tversky, CE)",
    )

    # ── Augmentation (new: per-transform config) ───────────────────
    augmentation_config: MONAIAugmentationConfig = Field(
        default_factory=MONAIAugmentationConfig,
        description="Per-transform 3D augmentation settings",
    )

    # ── Legacy fields ──────────────────────────────────────────────
    augmentation: bool = Field(
        default=True,
        description=(
            "[DEPRECATED — use augmentation_config.enabled instead] "
            "Enable MONAI's 3D data augmentation transforms"
        ),
    )

    # ── Training utilities ─────────────────────────────────────────
    early_stopping_patience: int = Field(
        default=30,
        ge=5,
        le=200,
        description="Stop training if val_loss doesn't improve for N epochs",
    )


# ═════════════════════════════════════════════════════════════════════════
# YOLO Segmentation Strategy Config
# ═════════════════════════════════════════════════════════════════════════

_YOLO_SIZES = Literal["n", "s", "m", "l", "x"]


class YOLOSegConfig(BaseModel):
    """Configuration for Ultralytics YOLO segmentation strategy."""

    model_size: _YOLO_SIZES = Field(
        default="s",
        description="YOLO model scale: n (nano), s (small), m (medium), l (large), x (xlarge)",
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of training epochs",
    )
    imgsz: int = Field(
        default=640,
        ge=256,
        le=1280,
        description="Input image size for training",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Training batch size",
    )
    lr: float = Field(
        default=0.001,
        ge=1e-5,
        le=1e-1,
        description="Initial learning rate",
    )
    patience: int = Field(
        default=50,
        ge=10,
        le=300,
        description="Early stopping patience (epochs without improvement)",
    )


# ═════════════════════════════════════════════════════════════════════════
# MLS Heatmap Strategy Config
# ═════════════════════════════════════════════════════════════════════════

_HRNET_BACKBONES = Literal["hrnet_w32", "hrnet_w18"]
_MLS_AGGREGATION = Literal["max", "p90"]
_MLS_INPUT_CHANNELS = Literal[1, 3]


class MLSHeatmapConfig(BaseModel):
    """Configuration for HRNet heatmap-based MLS regression strategy."""

    # ── Model architecture ─────────────────────────────────────────
    backbone: _HRNET_BACKBONES = Field(
        default="hrnet_w32",
        description="HRNet backbone: 'hrnet_w32' (higher accuracy) or 'hrnet_w18' (faster/lighter)",
    )
    input_channels: _MLS_INPUT_CHANNELS = Field(
        default=3,
        description="Number of input channels: 3 (brain+subdural+bone) or 1 (single window)",
    )
    image_size: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Input image size in pixels (square). Output heatmap = image_size / 4",
    )

    # ── Heatmap generation ─────────────────────────────────────────
    heatmap_sigma: float = Field(
        default=2.0,
        ge=0.5,
        le=8.0,
        description="Standard deviation (px) of the Gaussian heatmap target",
    )

    # ── Training hyper-parameters ──────────────────────────────────
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Initial learning rate for AdamW optimizer",
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of training epochs",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Training batch size (adjust based on GPU memory)",
    )
    val_split: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Fraction of data reserved for validation",
    )

    # ── Slice selection & aggregation ──────────────────────────────
    top_k_slices: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top candidate slices for MLS aggregation",
    )
    aggregation: _MLS_AGGREGATION = Field(
        default="max",
        description=(
            "Aggregation method across top-K slices: "
            "'max' (conservative, picks largest MLS) or "
            "'p90' (90th percentile, robust to outliers)"
        ),
    )

    # ── Data augmentation ──────────────────────────────────────────
    rotation_deg: float = Field(
        default=10.0,
        ge=0.0,
        le=45.0,
        description="Maximum rotation ±degrees for augmentation (0 = disabled)",
    )
    translation: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        description="Maximum translation as fraction of image size (0 = disabled)",
    )
    intensity_jitter: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Maximum intensity brightness/contrast jitter (0 = disabled)",
    )
    augment_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying augmentation to a given sample",
    )

    # ── Training utilities ─────────────────────────────────────────
    early_stopping_patience: int = Field(
        default=15,
        ge=5,
        le=100,
        description="Stop training if val_mls_mae_mm doesn't improve for N epochs",
    )
    lr_scheduler_patience: int = Field(
        default=5,
        ge=2,
        le=50,
        description="ReduceLROnPlateau patience (epochs without improvement)",
    )
    use_amp: bool = Field(
        default=True,
        description="Use Automatic Mixed Precision (AMP) for faster training",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=16,
        description="Number of DataLoader worker processes",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
