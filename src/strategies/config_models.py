"""
config_models.py — Pydantic configuration models for every ICH strategy.

Each model defines the trainable hyper-parameters exposed to the user.
The JSON Schema of each model is used by the Streamlit UI to dynamically
render configuration forms without hard-coding field widgets.

Adding a new field here automatically surfaces it in the UI.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.strategies.loss_config import LossConfig
from src.strategies.augmentation_config import SMPAugmentationConfig, MONAIAugmentationConfig
from src.config import config_section


class CompetitionFoldConfig(BaseModel):
    """Fields shared by strategies participating in comparable OOF runs."""

    fold: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Validation fold from config/folds.csv (patient-grouped, 0-4)",
    )
    use_competition_folds: bool = Field(
        default=True,
        description=(
            "Use the immutable patient-grouped fold manifest. Disable only for "
            "quick local smoke tests whose metrics are not used for model selection."
        ),
    )


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


class SMPConfig(CompetitionFoldConfig):
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

_MONAI_MODELS = Literal["SegResNet"]
_MONAI_DIMENSIONS = Literal["3D"]


class MONAIConfig(CompetitionFoldConfig):
    """Configuration for MONAI-based segmentation strategy."""

    # ── Model ──────────────────────────────────────────────────────
    model: _MONAI_MODELS = Field(
        default="SegResNet",
        description="Fixed competition architecture: compact 3D SegResNet",
    )

    # ── Model dimension ────────────────────────────────────────────
    model_dimension: _MONAI_DIMENSIONS = Field(
        default="3D",
        description=(
            "Fixed 3D input used by the competition SegResNet path"
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
        description="Fallback validation fraction when use_competition_folds=false",
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


class YOLOSegConfig(CompetitionFoldConfig):
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


_FRACTURE_DEFAULTS = config_section("training", "yolo")


class FractureYOLOConfig(CompetitionFoldConfig):
    """Validated fracture detector config sourced from ``project.yaml``."""

    image_size: int = Field(default=int(_FRACTURE_DEFAULTS["image_size"]), ge=256, le=1280)
    epochs: int = Field(default=int(_FRACTURE_DEFAULTS["epochs"]), ge=10, le=500)
    batch_size: int = Field(default=int(_FRACTURE_DEFAULTS["batch_size"]), ge=1, le=128)
    patience: int = Field(default=int(_FRACTURE_DEFAULTS["patience"]), ge=10, le=300)
    optimizer: Literal["AdamW", "Adam", "SGD"] = _FRACTURE_DEFAULTS["optimizer"]
    lr: float = Field(default=float(_FRACTURE_DEFAULTS["lr"]), ge=1e-6, le=1e-1)
    pretrained: str = Field(default=str(_FRACTURE_DEFAULTS["pretrained"]), min_length=3)


# ═════════════════════════════════════════════════════════════════════════
# MLS Heatmap Strategy Config
# ═════════════════════════════════════════════════════════════════════════

_HRNET_BACKBONES = Literal["hrnet_w32", "hrnet_w18"]
_MLS_AGGREGATION = Literal[
    "max", "p90", "median", "quantile", "relative_component", "anchor_window",
    "joint_component", "severity_window",
]
_MLS_INPUT_CHANNELS = Literal[1, 3, 9]
_MLS_HEATMAP_DEFAULTS = config_section("training", "mls_heatmap")


class MLSHeatmapConfig(CompetitionFoldConfig):
    """Configuration for HRNet heatmap-based MLS regression strategy."""

    # ── Model architecture ─────────────────────────────────────────
    backbone: _HRNET_BACKBONES = Field(
        default="hrnet_w32",
        description="HRNet backbone: 'hrnet_w32' (higher accuracy) or 'hrnet_w18' (faster/lighter)",
    )
    input_channels: _MLS_INPUT_CHANNELS = Field(
        default=3,
        description=(
            "Number of input channels: 1 (brain), 3 (brain+subdural+bone), or "
            "9 (three adjacent slices, each with those three windows)."
        ),
    )
    context_cache_manifest_sha256: str | None = Field(
        default=None,
        description=(
            "Required immutable cache-manifest SHA-256 for the opt-in 2.5D MLS "
            "dataset. It binds the checkpoint configuration to its input contract."
        ),
    )
    context_cache_validation_receipt_sha256: str | None = Field(
        default=None,
        description=(
            "Required SHA-256 of the passing raw-DICOM cache-validation receipt "
            "for the opt-in 2.5D MLS dataset."
        ),
    )
    image_size: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Input image size in pixels (square). Output heatmap = image_size / 4",
    )

    # ── Heatmap generation ─────────────────────────────────────────
    heatmap_sigma: float = Field(
        default=3.5,
        ge=0.5,
        le=8.0,
        description=(
            "Standard deviation (px) of the Gaussian heatmap target. "
            "Larger values give smoother, easier-to-learn targets on small "
            "datasets; DARK sub-pixel decoding recovers precision at inference"
        ),
    )
    heatmap_sigma_anneal_end: float | None = Field(
        default=None,
        ge=0.5,
        le=8.0,
        description=(
            "Optional final training-target sigma for a symmetric coarse-to-fine "
            "linear schedule. heatmap_sigma remains the fixed validation sigma and "
            "the schedule midpoint; the first-epoch sigma is "
            "2 * heatmap_sigma - heatmap_sigma_anneal_end. Leave empty to preserve "
            "the historical fixed-sigma target."
        ),
    )

    @model_validator(mode="after")
    def validate_heatmap_sigma_schedule(self) -> "MLSHeatmapConfig":
        end = self.heatmap_sigma_anneal_end
        if end is None:
            return self
        if end > self.heatmap_sigma:
            raise ValueError(
                "heatmap_sigma_anneal_end must not exceed heatmap_sigma; "
                "only a preregistered coarse-to-fine schedule is supported"
            )
        start = 2.0 * self.heatmap_sigma - end
        if start > 8.0:
            raise ValueError(
                "symmetric heatmap sigma schedule starts above the supported 8 px maximum"
            )
        return self

    # ── Competition-aware auxiliary losses ───────────────────────
    mls_loss_weight: float = Field(
        default=float(_MLS_HEATMAP_DEFAULTS["loss"]["mls_weight"]),
        ge=0.0, le=5.0,
        description="Weight of differentiable MLS millimetre regression loss",
    )
    threshold_loss_weight: float = Field(
        default=float(_MLS_HEATMAP_DEFAULTS["loss"]["threshold_weight"]),
        ge=0.0, le=5.0,
        description="Weight of ordinal loss at the official 1/3/5 mm boundaries",
    )
    softargmax_temperature: float = Field(
        default=float(_MLS_HEATMAP_DEFAULTS["loss"]["softargmax_temperature"]),
        gt=0.0, le=1.0,
        description="Temperature for differentiable heatmap coordinate decoding",
    )
    training_geometry_decoder: Literal["global_softargmax", "local_softargmax"] = Field(
        default="global_softargmax",
        description="Coordinate decoder for primary multitask geometry losses; deployment remains unchanged.",
    )
    local_softargmax_radius: int = Field(
        default=6, ge=1, le=16,
        description="Peak-centered local training window radius in heatmap pixels.",
    )
    threshold_temperature_mm: float = Field(
        default=float(_MLS_HEATMAP_DEFAULTS["loss"]["threshold_temperature_mm"]),
        gt=0.0, le=5.0,
        description="Smoothness of ordinal logits around official MLS thresholds",
    )
    signed_offset_loss_weight: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="Weight of signed perpendicular falx-offset supervision.",
    )
    study_bag_loss_weight: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description=(
            "Weight of the positive-study auxiliary loss that couples peak-slice "
            "selection with the official study-level maximum MLS target."
        ),
    )
    study_bag_threshold_loss_weight: float = Field(
        default=0.5, ge=0.0, le=5.0,
        description=(
            "Relative 1/3/5-mm boundary BCE weight inside the study-bag loss."
        ),
    )
    study_bag_peak_temperature: float = Field(
        default=1.0, gt=0.0, le=5.0,
        description=(
            "Softmax temperature used to form a differentiable peak-head-weighted "
            "study MLS surrogate from an annotated positive study bag."
        ),
    )
    study_bag_every_n_steps: int = Field(
        default=4, ge=1, le=64,
        description=(
            "Run one auxiliary positive-study bag update every N ordinary slice "
            "batches; the primary slice sampler remains unchanged."
        ),
    )
    within_study_rank_loss_weight: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description=(
            "Weight of a sparse RankNet-style selector loss over pairs of "
            "annotated target slices from the same study. It supervises the "
            "deployed top-k selector ordering without changing study pooling."
        ),
    )
    within_study_rank_min_gap_mm: float = Field(
        default=1.0, gt=0.0, le=20.0,
        description=(
            "Only same-study pairs whose annotated local MLS differs by at "
            "least this amount participate in the ranking auxiliary."
        ),
    )
    within_study_rank_temperature: float = Field(
        default=1.0, gt=0.0, le=5.0,
        description="Temperature applied to selector-logit differences in the ranking loss.",
    )
    within_study_rank_every_n_steps: int = Field(
        default=4, ge=1, le=64,
        description=(
            "Run one same-study pair auxiliary update every N ordinary slice "
            "batches; the primary slice sampler remains unchanged."
        ),
    )
    within_study_rank_detach_backbone: bool = Field(
        default=False,
        description=(
            "When true, run the extra same-study rank update through the selector "
            "head only: backbone activations and running statistics are frozen for "
            "that auxiliary forward. Ordinary slice losses still train the complete "
            "multitask model."
        ),
    )
    use_reference_refinement: bool = Field(
        default=False,
        description="Opt-in predicted-reference-conditioned residual outer heatmap head.",
    )
    use_ordinal_aux_head: bool = Field(
        default=False,
        description=(
            "Attach an independent pooled-feature ordinal head for the official "
            "1/3/5 mm boundaries. The head is training-only and does not replace "
            "the keypoint-derived MLS inference path."
        ),
    )
    ordinal_head_loss_weight: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="Weight of the independent three-logit ordinal auxiliary BCE",
    )
    ordinal_boundary_weights: tuple[float, float, float] = Field(
        default=(0.75, 1.0, 1.25),
        description=(
            "Fixed BCE weights for the 1/3/5 mm logits. The 5 mm boundary is "
            "slightly emphasized because it was the failed critical-edge gate."
        ),
    )
    ordinal_monotonic_penalty_weight: float = Field(
        default=0.1, ge=0.0, le=5.0,
        description=(
            "Within-head penalty for violating P(MLS>=1)>=P(MLS>=3)>=P(MLS>=5)"
        ),
    )

    @model_validator(mode="after")
    def validate_ordinal_auxiliary_head(self) -> "MLSHeatmapConfig":
        is_2p5d = self.dataset_variant == "multitask_2p5d_v1"
        cache_hash = self.context_cache_manifest_sha256
        receipt_hash = self.context_cache_validation_receipt_sha256
        if is_2p5d:
            if self.input_channels not in {3, 9}:
                raise ValueError(
                    "multitask_2p5d_v1 requires 3-channel central control or "
                    "9-channel adjacent-slice context"
                )
            if (
                not isinstance(cache_hash, str)
                or len(cache_hash) != 64
                or any(char not in "0123456789abcdef" for char in cache_hash.lower())
            ):
                raise ValueError(
                    "multitask_2p5d_v1 requires a 64-character "
                    "context_cache_manifest_sha256"
                )
            if (
                not isinstance(receipt_hash, str)
                or len(receipt_hash) != 64
                or any(char not in "0123456789abcdef" for char in receipt_hash.lower())
            ):
                raise ValueError(
                    "multitask_2p5d_v1 requires a 64-character "
                    "context_cache_validation_receipt_sha256"
                )
            if self.image_size != 512:
                raise ValueError(
                    "multitask_2p5d_v1 is locked to the native 512x512 cache contract"
                )
            if not self.use_selector:
                raise ValueError(
                    "multitask_2p5d_v1 requires use_selector=true for its CUDA multitask trainer"
                )
        elif self.input_channels == 9:
            raise ValueError(
                "input_channels=9 requires dataset_variant=multitask_2p5d_v1 "
                "so training and CUDA inference share the same adjacent-slice contract"
            )
        elif cache_hash is not None or receipt_hash is not None:
            raise ValueError(
                "2.5D cache manifest/validation receipt hashes are valid only "
                "for multitask_2p5d_v1"
            )
        if self.use_reference_refinement and not self.use_selector:
            raise ValueError("Reference refinement requires the multitask selector path")
        if self.ordinal_head_loss_weight > 0.0 and not self.use_ordinal_aux_head:
            raise ValueError(
                "ordinal_head_loss_weight>0 requires use_ordinal_aux_head=true"
            )
        if any(weight <= 0.0 for weight in self.ordinal_boundary_weights):
            raise ValueError("ordinal_boundary_weights must contain three positive values")
        if self.study_bag_loss_weight > 0.0 and not self.use_selector:
            raise ValueError(
                "study_bag_loss_weight>0 requires use_selector=true because it "
                "uses the peak-selection logits"
            )
        if (
            self.study_bag_loss_weight > 0.0
            and self.dataset_variant not in {"multitask_v2", "multitask_2p5d_v1"}
        ):
            raise ValueError(
                "study_bag_loss_weight>0 requires a supported multitask dataset"
            )
        if self.within_study_rank_loss_weight > 0.0 and not self.use_selector:
            raise ValueError(
                "within_study_rank_loss_weight>0 requires use_selector=true "
                "because it supervises peak-selection logits"
            )
        if (
            self.within_study_rank_loss_weight > 0.0
            and self.dataset_variant not in {"multitask_v2", "multitask_2p5d_v1"}
        ):
            raise ValueError(
                "within_study_rank_loss_weight>0 requires "
                "a supported multitask dataset"
            )
        if (
            self.within_study_rank_detach_backbone
            and self.within_study_rank_loss_weight <= 0.0
        ):
            raise ValueError(
                "within_study_rank_detach_backbone=true requires "
                "within_study_rank_loss_weight>0"
            )
        return self

    use_selector: bool = Field(
        default=False,
        description=(
            "Train an explicit target-slice selector jointly with the keypoint "
            "heatmaps. Required for multitask_v2 experiments."
        ),
    )
    selector_loss_weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Weight of the target-vs-nontarget slice BCE objective",
    )
    selector_head_mode: Literal["single", "dual"] = Field(
        default="single",
        description=(
            "single preserves the historical selector logit. dual predicts "
            "independent target-presence and within-target peak-severity logits "
            "so study gating and slice ranking do not share one calibration."
        ),
    )
    selector_peak_loss_weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description=(
            "Relative peak-head BCE weight in dual mode. The two selector losses "
            "are normalized by 1 + this value so the total selector-loss scale "
            "remains comparable with the historical single-head recipe."
        ),
    )
    selector_target_mode: Literal["binary", "peak_aware_soft"] = Field(
        default="binary",
        description=(
            "Binary labels treat every annotated slice equally. peak_aware_soft "
            "retains target-slice supervision while assigning larger targets to "
            "slices closer to the official study-level maximum MLS."
        ),
    )
    selector_peak_base: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum selector target assigned to an annotated non-peak slice",
    )
    selector_peak_power: float = Field(
        default=1.0, gt=0.0, le=4.0,
        description="Exponent applied to slice-MLS / study-max MLS for peak-aware labels",
    )
    spatial_loss_weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Weight of spatial heatmap distribution cross-entropy",
    )
    coordinate_loss_weight: float = Field(
        default=0.5, ge=0.0, le=10.0,
        description="Weight of normalized differentiable keypoint coordinate loss",
    )
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, le=32,
        description="Accumulate CUDA gradients to emulate a larger batch on low VRAM",
    )
    dataset_variant: Literal["positive_only", "multitask_v2", "multitask_2p5d_v1"] = Field(
        default="positive_only",
        description="Prepared MLS dataset contract used by the trainer",
    )

    # ── Training hyper-parameters ──────────────────────────────────
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Initial learning rate for AdamW optimizer",
    )
    weight_decay: float = Field(
        default=1e-3,
        ge=0.0,
        le=0.1,
        description="L2 weight decay for AdamW (higher = stronger regularization)",
    )
    head_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout2d probability in the heatmap head (0 = disabled)",
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
        description="Fallback validation fraction when use_competition_folds=false",
    )
    sampling_mode: Literal[
        "slice_class_balanced",
        "hybrid_study_class_balanced",
        "study_class_balanced",
    ] = Field(
        default="slice_class_balanced",
        description=(
            "Training sampler policy. slice_class_balanced reproduces legacy 50/50 "
            "target-vs-nontarget row sampling. hybrid_study_class_balanced keeps "
            "study mass proportional to the square root of its row count. "
            "study_class_balanced gives each study equal mass within a class."
        ),
    )

    # ── Slice selection & aggregation ──────────────────────────────
    top_k_slices: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top candidate slices for MLS aggregation",
    )
    selector_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Study is treated as MLS-negative when the maximum selector "
            "probability is below this threshold"
        ),
    )
    negative_value_mm: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="MLS value returned for selector-negative studies",
    )
    aggregation: _MLS_AGGREGATION = Field(
        default="max",
        description=(
            "Aggregation method across top-K slices: "
            "'max' (conservative, picks largest MLS) or "
            "'p90' (90th percentile, robust to outliers), 'median', 'quantile', "
            "'relative_component' (contiguous selector component around its peak), "
            "'anchor_window' (fixed contiguous radius around the selector peak), "
            "'joint_component' (selector and heatmap-confidence component), or "
            "'severity_window' (target/heatmap/severity anchored neighbourhood)"
        ),
    )
    selector_relative_ratio: float = Field(
        default=0.3, ge=0.05, le=1.0,
        description=(
            "For relative_component pooling, keep the contiguous component around "
            "the selector peak whose probabilities are at least this fraction of the peak"
        ),
    )
    aggregation_quantile: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description=(
            "Quantile of slice MLS values used by quantile, relative_component, "
            "anchor_window, joint_component, and severity_window pooling"
        ),
    )
    aggregation_probability_weighted: bool = Field(
        default=False,
        description=(
            "Weight the study-level MLS quantile by selector probability. "
            "Useful when peak-aware selector supervision is enabled."
        ),
    )
    anchor_window_radius: int = Field(
        default=2, ge=0, le=10,
        description=(
            "For anchor_window pooling, include this many adjacent slices on each "
            "side of the maximum-selector slice"
        ),
    )
    min_active_slices: int = Field(
        default=1, ge=1, le=10,
        description=(
            "Minimum number of slices at or above selector_threshold required to "
            "treat a study as MLS-positive"
        ),
    )
    heatmap_guard_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Within a selected component, discard slices whose heatmap confidence "
            "is below this fraction of the component maximum; 0 disables the guard"
        ),
    )

    # ── Data augmentation ──────────────────────────────────────────
    rotation_deg: float = Field(
        default=15.0,
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
        default=0.9,
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
    snapshot_start_epoch: int = Field(
        default=0,
        ge=0,
        le=500,
        description=(
            "First epoch eligible for a local audit snapshot; 0 disables periodic "
            "snapshots. Snapshots are intended for post-training full-study CUDA selection."
        ),
    )
    snapshot_every_n_epochs: int = Field(
        default=0,
        ge=0,
        le=100,
        description=(
            "Save a local audit snapshot every N epochs from snapshot_start_epoch; "
            "0 disables periodic snapshots. These files are not uploaded automatically."
        ),
    )
    resume_checkpoint: str | None = Field(
        default=None,
        description=(
            "Optional trusted full-state MLS recovery checkpoint. It restores model, "
            "optimizer, scheduler, AMP scaler, epoch counters, history, and RNG state."
        ),
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
    training_determinism: Literal["benchmark", "reproducible", "strict"] = Field(
        default="benchmark",
        description=(
            "CUDA/data-order reproducibility policy. benchmark preserves the historical "
            "fast cuDNN autotuning behavior; reproducible disables autotuning, seeds each "
            "epoch and DataLoader worker explicitly, and warns on unsupported deterministic "
            "CUDA operations; strict converts those warnings into fail-fast errors."
        ),
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
