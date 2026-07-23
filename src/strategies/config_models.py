"""
config_models.py — Pydantic configuration models for every ICH strategy.

Each model defines the trainable hyper-parameters exposed to the user.
The JSON Schema of each model is used by the Streamlit UI to dynamically
render configuration forms without hard-coding field widgets.

Adding a new field here automatically surfaces it in the UI.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

_SMP_LOSSES = Literal["Dice", "CrossEntropy", "Combined"]


class SMPConfig(BaseModel):
    """Configuration for Segmentation Models PyTorch (SMP) strategy."""

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
    augmentation: bool = Field(
        default=True,
        description="Enable on-the-fly data augmentation (Albumentations)",
    )
    loss: _SMP_LOSSES = Field(
        default="Combined",
        description="Loss function: Dice, CrossEntropy, or Combined (Dice + CE)",
    )
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


class MONAIConfig(BaseModel):
    """Configuration for MONAI-based 3D segmentation strategy."""

    model: _MONAI_MODELS = Field(
        default="UNETR",
        description="MONAI network architecture for 3D segmentation",
    )
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
    augmentation: bool = Field(
        default=True,
        description="Enable MONAI's 3D data augmentation transforms",
    )
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
