"""Leakage-safe training of the direct 2.5D ICH segmentation/volume model."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)

from .cache import OUTPUT_LABELS
from .segmentation_data import (
    create_segmentation_loaders,
    load_hard_negative_slice_manifest,
    oof_hard_negative_row_mask,
    segmentation_classification_weights,
    segmentation_foreground_counts,
    segmentation_foreground_weights,
    subtype_aware_sampling_weights,
)
from .segmentation_evaluation import summarize_segmentation_predictions
from .segmentation_loss import ICH25DSegmentationLoss
from .segmentation_model import (
    DEFAULT_SEGMENTATION_ARCHITECTURE,
    DEFAULT_SEGMENTATION_ENCODER,
    FiveSliceContextInputAdapter,
    HorizontalSymmetryInputAdapter,
    SahBackgroundExpansionAdapter,
    base_segmentation_model,
    build_segmentation_model,
    input_adapter_residual,
    load_segmentation_weights,
)


CHECKPOINT_SELECTION_STRATEGIES = (
    "legacy",
    "fpr_penalized",
    "fpr_volume_penalized",
)
CHECKPOINT_SELECTION_METRICS = {
    "legacy": "ich_only_0.55_dice_0.30_any_auc_0.15_subtype_auc",
    "fpr_penalized": "ich_only_selection_minus_0.10_normal_fpr",
    "fpr_volume_penalized": (
        "ich_only_selection_minus_0.10_normal_fpr_minus_0.005_total_volume_mae_ml"
        "_minus_0.001_abs_total_volume_bias_ml"
    ),
}
SAFE_MLFLOW_ARTIFACT_NAMES = (
    "best.pth",
    "resolved_config.json",
    "history.csv",
    "best_calibration_summary.json",
    "outer_summary.json",
    "run_summary.json",
)


def log_safe_mlflow_artifacts(output: str | Path) -> tuple[str, ...]:
    """Log only aggregate/model artifacts, never row-level medical predictions."""
    root = Path(output)
    logged: list[str] = []
    for name in SAFE_MLFLOW_ARTIFACT_NAMES:
        artifact = root / name
        if not artifact.is_file():
            continue
        mlflow.log_artifact(
            str(artifact), artifact_path="ich_2p5d_segmentation_run"
        )
        logged.append(name)
    return tuple(logged)


def checkpoint_selection_score(
    summary: dict[str, Any], strategy: str
) -> float:
    selection = float(summary["selection_score"])
    if strategy == "legacy":
        return selection
    if strategy == "fpr_penalized":
        return selection - 0.10 * float(
            summary["normal_false_positive_rate_at_0_1ml"]
        )
    if strategy == "fpr_volume_penalized":
        return (
            selection
            - 0.10 * float(summary["normal_false_positive_rate_at_0_1ml"])
            - 0.005 * float(summary["total_volume_mae_ml"])
            - 0.001 * abs(float(summary["total_volume_bias_ml"]))
        )
    raise ValueError(
        "checkpoint_selection_strategy must be one of: "
        f"{', '.join(CHECKPOINT_SELECTION_STRATEGIES)}"
    )


@dataclass(frozen=True)
class ICH25DSegmentationTrainConfig:
    run_name: str
    output_dir: str
    manifest_path: str = "Data/processed/ich_2p5d/slice_manifest.csv"
    architecture: str = DEFAULT_SEGMENTATION_ARCHITECTURE
    encoder_name: str = DEFAULT_SEGMENTATION_ENCODER
    outer_fold: int = 0
    calibration_fold: int = 1
    epochs: int = 8
    batch_size: int = 8
    workers: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    classification_loss_weight: float = 0.25
    classification_focal_gamma: float = 1.0
    background_weight: float = 0.15
    empty_foreground_weight: float = 0.0
    empty_foreground_top_fraction: float = 1.0
    checkpoint_selection_strategy: str = "legacy"
    maximum_pos_weight: float = 20.0
    segmentation_class_weight_power: float = 0.0
    maximum_segmentation_class_weight: float = 8.0
    segmentation_class_weight_basis: str = "slice"
    sampler_study_balance_power: float = 0.0
    hard_negative_manifest: str | None = None
    hard_negative_multiplier: float = 1.0
    initial_checkpoint: str | None = None
    horizontal_symmetry_adapter: bool = False
    five_slice_context_adapter: bool = False
    sah_residual_adapter: bool = False
    sah_residual_hidden_channels: int = 16
    sah_maximum_logit_residual: float = 8.0
    sah_include_incumbent_iph: bool = False
    slice_context_radius: int = 1
    freeze_base_model: bool = False
    classification_head_only: bool = False
    ivh_center_loss_weight: float = 0.0
    ivh_center_square_size: int = 11
    physical_volume_loss_weight: float = 0.0
    diffuse_tversky_loss_weight: float = 0.0
    sah_tversky_loss_weight: float = 0.0
    sah_positive_pixel_loss_weight: float = 0.0
    segmentation_objective: str = "multiclass"
    foreground_dice_weight: float = 0.40
    foreground_focal_weight: float = 0.20
    conditional_subtype_weight: float = 0.30
    subtype_ovr_weight: float = 0.10
    conditional_subtype_mode: str = "cross_entropy"
    foreground_gradient_mode: str = "probability_weighted"
    pretrained: bool = True
    seed: int = 42
    patience: int = 3
    max_train_steps: int | None = None
    evaluate_outer: bool = True


def _should_evaluate_outer(config: ICH25DSegmentationTrainConfig) -> bool:
    """Reserve outer inference for explicitly authorized, non-smoke runs."""
    return config.evaluate_outer and config.max_train_steps is None


def _should_stop_after_epoch(config: ICH25DSegmentationTrainConfig) -> bool:
    """A bounded smoke run performs one partial epoch and one calibration pass."""
    return config.max_train_steps is not None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_initial_checkpoint_provenance(
    payload: dict[str, Any], config: ICH25DSegmentationTrainConfig
) -> None:
    """Reject warm starts that cross the model architecture or held-out folds."""
    source = payload.get("config")
    if not isinstance(source, dict):
        raise ValueError("Initial checkpoint is missing its training config")
    expected = {
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "outer_fold": config.outer_fold,
        "calibration_fold": config.calibration_fold,
    }
    mismatches = {
        key: {"checkpoint": source.get(key), "requested": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Initial checkpoint provenance does not match this split/model: {mismatches}"
        )
    if payload.get("output_labels") != OUTPUT_LABELS:
        raise ValueError("Initial checkpoint output labels do not match")
    if int(payload.get("segmentation_classes", -1)) != 6:
        raise ValueError("Initial checkpoint segmentation classes do not match")
    source_five_slice = bool(source.get("five_slice_context_adapter", False))
    source_sah_residual = bool(source.get("sah_residual_adapter", False))
    source_sah_include_iph = bool(source.get("sah_include_incumbent_iph", False))
    source_context_radius = int(source.get("slice_context_radius", 1))
    source_input_channels = 3 * (2 * source_context_radius + 1)
    if int(payload.get("input_channels", -1)) != source_input_channels:
        raise ValueError("Initial checkpoint input channels do not match")
    source_uses_adapter = bool(source.get("horizontal_symmetry_adapter", False))
    if source_uses_adapter and not config.horizontal_symmetry_adapter:
        raise ValueError(
            "A symmetry-adapter checkpoint cannot initialize a legacy model"
        )
    if source_five_slice and not config.five_slice_context_adapter:
        raise ValueError(
            "A five-slice-adapter checkpoint cannot initialize another model type"
        )
    if source_uses_adapter and config.five_slice_context_adapter:
        raise ValueError("Cannot initialize five-slice context from a symmetry adapter")
    if source_five_slice and config.horizontal_symmetry_adapter:
        raise ValueError("Cannot initialize symmetry adapter from five-slice context")
    if source_sah_residual and not config.sah_residual_adapter:
        raise ValueError("A SAH-residual checkpoint cannot initialize another model type")
    if source_sah_residual and (
        source_sah_include_iph != config.sah_include_incumbent_iph
    ):
        raise ValueError("Cannot change incumbent IPH support when resuming a SAH adapter")
    source_adapter_count = sum(
        bool(value)
        for value in (source_uses_adapter, source_five_slice, source_sah_residual)
    )
    target_adapter_count = sum(
        bool(value)
        for value in (
            config.horizontal_symmetry_adapter,
            config.five_slice_context_adapter,
            config.sah_residual_adapter,
        )
    )
    if source_adapter_count and source_adapter_count != target_adapter_count:
        raise ValueError("Cannot initialize one adapter type from another adapter type")


def load_initial_segmentation_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    config: ICH25DSegmentationTrainConfig,
) -> dict[str, Any]:
    """Load a same-split checkpoint, allowing legacy-to-adapter expansion."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("2.5D segmentation checkpoint must be a dictionary")
    validate_initial_checkpoint_provenance(payload, config)
    state = payload.get("state_dict", payload)
    source = payload.get("config", {})
    source_uses_adapter = bool(
        source.get("horizontal_symmetry_adapter", False)
        or source.get("five_slice_context_adapter", False)
        or source.get("sah_residual_adapter", False)
    )
    target_uses_adapter = bool(
        config.horizontal_symmetry_adapter
        or config.five_slice_context_adapter
        or config.sah_residual_adapter
    )
    target = (
        base_segmentation_model(model)
        if target_uses_adapter and not source_uses_adapter
        else model
    )
    target.load_state_dict(state, strict=True)
    return payload


def configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    freeze_base_model: bool,
    classification_head_only: bool = False,
) -> list[torch.nn.Parameter]:
    """Select a narrow trainable scope without perturbing incumbent masks."""
    if freeze_base_model and classification_head_only:
        raise ValueError("Only one frozen-base training scope can be enabled")
    if freeze_base_model:
        if not isinstance(
            model,
            (
                HorizontalSymmetryInputAdapter,
                FiveSliceContextInputAdapter,
                SahBackgroundExpansionAdapter,
            ),
        ):
            raise ValueError("freeze_base_model requires a supported adapter")
        model.base_model.requires_grad_(False)
        input_adapter_residual(model).requires_grad_(True)
    if classification_head_only:
        model.requires_grad_(False)
        head = getattr(base_segmentation_model(model), "classification_head", None)
        if not isinstance(head, torch.nn.Module):
            raise ValueError("Model does not expose an auxiliary classification head")
        head.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Training configuration has no trainable parameters")
    return parameters


def set_segmentation_training_mode(
    model: torch.nn.Module,
    *,
    freeze_base_model: bool,
    classification_head_only: bool = False,
) -> None:
    """Keep frozen BatchNorm/decoder state fixed for narrow-scope training."""
    if freeze_base_model and classification_head_only:
        raise ValueError("Only one frozen-base training scope can be enabled")
    model.train()
    if freeze_base_model:
        if not isinstance(
            model,
            (
                HorizontalSymmetryInputAdapter,
                FiveSliceContextInputAdapter,
                SahBackgroundExpansionAdapter,
            ),
        ):
            raise ValueError("freeze_base_model requires a supported adapter")
        model.base_model.eval()
        input_adapter_residual(model).train()
    if classification_head_only:
        model.eval()
        head = getattr(base_segmentation_model(model), "classification_head", None)
        if not isinstance(head, torch.nn.Module):
            raise ValueError("Model does not expose an auxiliary classification head")
        head.train()


def _notify_non_smoke(
    run_kind: str, event: str, message: str, **fields: object
) -> None:
    """Keep Telegram focused on substantive runs while retaining smoke failures."""
    if run_kind != "smoke":
        notify_campaign(event, message, **fields)


def _unpack_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise TypeError("Segmentation model must return mask and auxiliary logits")
    return outputs[0], outputs[1]


def _predict_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    horizontal_flip_tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probability-space predictions with an optional symmetry TTA.

    The spatial prediction from the flipped view is restored to the original
    coordinate system before averaging.  Probability averaging keeps the
    default path bit-for-bit unchanged while avoiding a scale assumption about
    logits from the two views.
    """
    mask_logits, class_logits = _unpack_outputs(model(images))
    mask_probabilities = torch.softmax(mask_logits.float(), dim=1)
    class_probabilities = torch.sigmoid(class_logits.float())
    if not horizontal_flip_tta:
        return mask_probabilities, class_probabilities

    flipped_images = torch.flip(images, dims=(-1,))
    flipped_mask_logits, flipped_class_logits = _unpack_outputs(
        model(flipped_images)
    )
    restored_mask_probabilities = torch.flip(
        torch.softmax(flipped_mask_logits.float(), dim=1), dims=(-1,)
    )
    mask_probabilities = 0.5 * (
        mask_probabilities + restored_mask_probabilities
    )
    class_probabilities = 0.5 * (
        class_probabilities + torch.sigmoid(flipped_class_logits.float())
    )
    return mask_probabilities, class_probabilities


def _predict_slices(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    horizontal_flip_tta: bool = False,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_probabilities, class_probabilities = _predict_probabilities(
                    model,
                    images,
                    horizontal_flip_tta=horizontal_flip_tta,
                )
            predicted_masks = mask_probabilities.argmax(dim=1).cpu()
            class_probabilities = class_probabilities.cpu().numpy()
            true_masks = batch["mask"]
            known = batch["known"].numpy()
            voxel_volumes = batch["voxel_volume_ml"].numpy()
            slice_indices = batch["slice_index"].numpy()
            for index, study_id in enumerate(batch["study_id"]):
                predicted = predicted_masks[index]
                observed = true_masks[index]
                is_known = bool(known[index] > 0.5)
                row: dict[str, object] = {
                    "study_id": str(study_id),
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_index": int(slice_indices[index]),
                    "known": int(is_known),
                    "voxel_volume_ml": float(voxel_volumes[index]),
                }
                for output_index, label in enumerate(OUTPUT_LABELS):
                    row[f"prob_{label}"] = float(class_probabilities[index, output_index])
                for class_id, label in enumerate(OUTPUT_LABELS[1:], start=1):
                    predicted_class = predicted == class_id
                    row[f"pred_pixels_{label}"] = int(predicted_class.sum())
                    if is_known:
                        observed_class = observed == class_id
                        row[f"intersection_{label}"] = int(
                            (predicted_class & observed_class).sum()
                        )
                        row[f"predicted_known_pixels_{label}"] = int(
                            predicted_class.sum()
                        )
                        row[f"observed_known_pixels_{label}"] = int(
                            observed_class.sum()
                        )
                    else:
                        row[f"intersection_{label}"] = 0
                        row[f"predicted_known_pixels_{label}"] = 0
                        row[f"observed_known_pixels_{label}"] = 0
                rows.append(row)
    return pd.DataFrame(rows)


def _flatten_summary_metrics(prefix: str, summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        f"{prefix}_selection_score": float(summary["selection_score"]),
        f"{prefix}_mean_foreground_dice": float(summary["mean_foreground_dice"]),
        f"{prefix}_any_ich_study_auc": float(summary["any_ich_study_auc"] or 0.0),
        f"{prefix}_macro_subtype_study_auc": float(summary["macro_subtype_study_auc"]),
        f"{prefix}_presence_f1_at_0_1ml": float(summary["presence_f1_at_0_1ml"]),
        f"{prefix}_normal_fpr_at_0_1ml": float(
            summary["normal_false_positive_rate_at_0_1ml"]
        ),
        f"{prefix}_total_volume_mae_ml": float(summary["total_volume_mae_ml"]),
        f"{prefix}_total_volume_bias_ml": float(summary["total_volume_bias_ml"]),
    }
    for label, subtype in summary["subtypes"].items():
        for name in ("dice_known_pixels", "study_auc", "mae_ml", "bias_ml"):
            value = subtype[name]
            if value is not None and np.isfinite(value):
                metrics[f"{prefix}_{label.lower()}_{name}"] = float(value)
        for stratum_name, stratum in subtype.get("volume_strata", {}).items():
            for name in (
                "positive_studies",
                "dice_known_pixels",
                "presence_sensitivity_at_0_1ml",
                "mae_ml",
                "median_absolute_error_ml",
                "median_relative_absolute_error",
            ):
                value = stratum[name]
                if value is not None and np.isfinite(value):
                    metrics[
                        f"{prefix}_{label.lower()}_{stratum_name}_{name}"
                    ] = float(value)
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    config: ICH25DSegmentationTrainConfig,
    epoch: int,
    calibration_summary: dict[str, Any],
    manifest_sha256: str,
    hard_negative_manifest_sha256: str | None,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 1,
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "output_labels": OUTPUT_LABELS,
        "segmentation_classes": 6,
        "input_channels": 3 * (2 * config.slice_context_radius + 1),
        "selection_metric": CHECKPOINT_SELECTION_METRICS[
            config.checkpoint_selection_strategy
        ],
        "calibration_summary": calibration_summary,
        "manifest_sha256": manifest_sha256,
        "hard_negative_manifest_sha256": hard_negative_manifest_sha256,
        "git_commit": git_commit(),
    }, temporary)
    os.replace(temporary, path)


def run_segmentation_training(
    config: ICH25DSegmentationTrainConfig,
) -> dict[str, Any]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("outer_fold and calibration_fold must differ")
    if config.checkpoint_selection_strategy not in CHECKPOINT_SELECTION_STRATEGIES:
        raise ValueError(
            "checkpoint_selection_strategy must be one of: "
            f"{', '.join(CHECKPOINT_SELECTION_STRATEGIES)}"
        )
    if not 0.0 <= config.sampler_study_balance_power <= 1.0:
        raise ValueError("sampler_study_balance_power must be in [0, 1]")
    if not 1.0 <= config.hard_negative_multiplier <= 10.0:
        raise ValueError("hard_negative_multiplier must be in [1, 10]")
    if config.hard_negative_multiplier > 1.0 and not config.hard_negative_manifest:
        raise ValueError(
            "hard_negative_multiplier > 1 requires hard_negative_manifest"
        )
    if config.ivh_center_loss_weight < 0:
        raise ValueError("ivh_center_loss_weight must be non-negative")
    if config.physical_volume_loss_weight < 0:
        raise ValueError("physical_volume_loss_weight must be non-negative")
    if config.diffuse_tversky_loss_weight < 0:
        raise ValueError("diffuse_tversky_loss_weight must be non-negative")
    if config.sah_tversky_loss_weight < 0:
        raise ValueError("sah_tversky_loss_weight must be non-negative")
    if config.sah_positive_pixel_loss_weight < 0:
        raise ValueError("sah_positive_pixel_loss_weight must be non-negative")
    if config.ivh_center_loss_weight > 0 and (
        config.ivh_center_square_size < 1
        or config.ivh_center_square_size % 2 == 0
    ):
        raise ValueError(
            "ivh_center_square_size must be a positive odd integer when enabled"
        )
    if config.max_train_steps is not None and config.max_train_steps < 1:
        raise ValueError("max_train_steps must be a positive integer when set")
    if config.slice_context_radius not in (1, 2):
        raise ValueError("slice_context_radius must be 1 or 2")
    adapter_count = sum(
        bool(value)
        for value in (
            config.horizontal_symmetry_adapter,
            config.five_slice_context_adapter,
            config.sah_residual_adapter,
        )
    )
    if adapter_count > 1:
        raise ValueError("Only one ICH adapter can be enabled")
    if config.horizontal_symmetry_adapter and config.slice_context_radius != 1:
        raise ValueError("horizontal symmetry adapter requires slice_context_radius=1")
    if config.five_slice_context_adapter and config.slice_context_radius != 2:
        raise ValueError("five-slice context adapter requires slice_context_radius=2")
    if config.slice_context_radius == 2 and not config.five_slice_context_adapter:
        raise ValueError("slice_context_radius=2 requires five_slice_context_adapter")
    if config.sah_residual_adapter and config.slice_context_radius != 1:
        raise ValueError("SAH residual adapter requires slice_context_radius=1")
    if config.sah_include_incumbent_iph and not config.sah_residual_adapter:
        raise ValueError("incumbent IPH support requires the SAH residual adapter")
    if config.sah_residual_hidden_channels < 1:
        raise ValueError("sah_residual_hidden_channels must be positive")
    if config.sah_maximum_logit_residual <= 0:
        raise ValueError("sah_maximum_logit_residual must be positive")
    if config.sah_residual_adapter and not config.freeze_base_model:
        raise ValueError("SAH residual adapter requires freeze_base_model")
    if config.freeze_base_model and adapter_count == 0:
        raise ValueError("freeze_base_model requires an adapter")
    if config.freeze_base_model and not config.initial_checkpoint:
        raise ValueError("freeze_base_model requires an initial_checkpoint")
    if config.classification_head_only and not config.initial_checkpoint:
        raise ValueError("classification_head_only requires an initial_checkpoint")
    if config.classification_head_only and (
        config.freeze_base_model
        or config.horizontal_symmetry_adapter
        or config.five_slice_context_adapter
        or config.sah_residual_adapter
    ):
        raise ValueError(
            "classification_head_only cannot be combined with an input adapter"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("2.5D ICH segmentation training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("2.5D ICH segmentation requires CUDA BF16 support")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pth"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {checkpoint_path}")
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    hard_negative_slices = (
        load_hard_negative_slice_manifest(config.hard_negative_manifest)
        if config.hard_negative_manifest
        else None
    )
    (
        train_loader,
        calibration_loader,
        outer_loader,
        train_frame,
        calibration_frame,
        outer_frame,
    ) = create_segmentation_loaders(
        config.manifest_path,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
        batch_size=config.batch_size,
        workers=config.workers,
        seed=config.seed,
        sampler_study_balance_power=config.sampler_study_balance_power,
        hard_negative_slices=hard_negative_slices,
        hard_negative_multiplier=config.hard_negative_multiplier,
        ivh_center_square_size=(
            config.ivh_center_square_size
            if config.ivh_center_loss_weight > 0
            else 0
        ),
        context_radius=config.slice_context_radius,
    )
    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    manifest_sha = file_sha256(config.manifest_path)
    hard_negative_manifest_sha = (
        file_sha256(config.hard_negative_manifest)
        if config.hard_negative_manifest
        else None
    )
    initial_checkpoint_sha = (
        file_sha256(config.initial_checkpoint) if config.initial_checkpoint else None
    )
    sampling_weights = subtype_aware_sampling_weights(
        train_frame,
        study_balance_power=config.sampler_study_balance_power,
        hard_negative_slices=hard_negative_slices,
        hard_negative_multiplier=config.hard_negative_multiplier,
    ).numpy()
    sampling_positive = train_frame[list(OUTPUT_LABELS[1:])].any(axis=1).to_numpy()
    sampling_hard_negative = oof_hard_negative_row_mask(
        train_frame, hard_negative_slices
    )
    sampling_weight_sum = float(sampling_weights.sum())
    sampler_diagnostics = {
        "weight_min": float(sampling_weights.min()),
        "weight_max": float(sampling_weights.max()),
        "weight_mean": float(sampling_weights.mean()),
        "effective_sample_size": float(
            sampling_weight_sum**2 / np.square(sampling_weights).sum()
        ),
        "positive_probability_mass": float(
            sampling_weights[sampling_positive].sum() / sampling_weight_sum
        ),
        "hard_negative_multiplier": float(config.hard_negative_multiplier),
        "hard_negative_slices": int(sampling_hard_negative.sum()),
        "hard_negative_studies": int(
            train_frame.loc[sampling_hard_negative, "study_id"].nunique()
        ),
        "hard_negative_probability_mass": float(
            sampling_weights[sampling_hard_negative].sum() / sampling_weight_sum
        ),
    }
    run_kind = (
        "smoke"
        if config.max_train_steps
        else "full_fold" if config.evaluate_outer else "calibration_screen"
    )
    start_message = (
        f"گیت فنی segmentation دوبعدونیم ICH آغاز شد. فرضیه: sampler hard-negative "
        f"OOF با ضریب={config.hard_negative_multiplier:.2f} و study-balance="
        f"{config.sampler_study_balance_power:.2f} باید بدون تغییر جرم کل مثبت/منفی "
        "از نظر حافظه، loss و ذخیرهٔ checkpoint سالم باشد. تحلیل کوتاه: "
        "این اجرا فقط train/calibration را می‌بیند و outer fold عمداً ارزیابی نمی‌شود. "
        "اقدام بعدی: در صورت سلامت فنی، اجرای کامل هم‌پارامتر روی calibration ازپیش‌تعیین‌شده."
        if run_kind == "smoke"
        else (
            f"غربال کامل calibration-only مدل segmentation دوبعدونیم ICH آغاز شد. "
            f"توان study-balance sampler={config.sampler_study_balance_power:.2f} و "
            f"ضریب hard-negative OOF={config.hard_negative_multiplier:.2f} "
            f"و loss مرکز IVH با وزن={config.ivh_center_loss_weight:.3f} و مربع "
            f"{config.ivh_center_square_size}×{config.ivh_center_square_size} و loss حجم "
            f"فیزیکی با وزن={config.physical_volume_loss_weight:.4f} تنظیم شده‌اند. "
            f"Tversky مثبت‌محور SDH/SAH وزن={config.diffuse_tversky_loss_weight:.4f} دارد. "
            f"NLL مثبت‌پیکسلی SAH وزن={config.sah_positive_pixel_loss_weight:.4f} دارد. "
            "تحلیل کوتاه: مدل تا پایان و با انتخاب checkpoint "
            "روی calibration آموزش می‌بیند، اما outer fold عمداً inference نمی‌شود تا "
            "در صورت شکست گیت، دادهٔ ارزیابی بیشتر مصرف نشود. اقدام بعدی: مقایسهٔ "
            "selection، FPR، MAE و معیارهای IVH با baseline هم‌split؛ فقط در صورت عبور "
            "از همهٔ گیت‌ها، پنج-fold OOF اجرا می‌شود."
            if run_kind == "calibration_screen"
            else f"آموزش مدل مستقیم segmentation دوبعدونیم ICH آغاز شد. فرضیه: وزن empty-foreground={config.empty_foreground_weight:.3f} روی سخت‌ترین سهم={config.empty_foreground_top_fraction:.4f} از پیکسل‌های ماسک سالم باید false-positive موضعی را کم کند؛ sampler hard-negative OOF با ضریب={config.hard_negative_multiplier:.2f} فقط برش‌های mimic کاذبِ foldهای آموزشی را بیشتر می‌بیند و جرم کل منفی را ثابت نگه می‌دارد. راهبرد انتخاب checkpoint={config.checkpoint_selection_strategy} است؛ معیار دقیق داخل metadata checkpoint ثبت می‌شود و MAE همیشه گیت مستقل promotion باقی می‌ماند. اقدام بعدی: ارزیابی یک‌بارهٔ outer و مقایسه با baseline دقیقاً هم‌fold."
        )
    )
    if config.classification_head_only:
        start_message = (
            "غربال classification-head-only مدل ICH آغاز شد. encoder، decoder، "
            "BatchNorm و تمام مسیر فضایی از checkpoint هم‌split فریز هستند و فقط "
            "auxiliary head بازآموزی می‌شود. تحلیل کوتاه: این آزمایش باید Dice، "
            "حجم، FPR، F1 و MAE را دقیقاً ثابت نگه دارد؛ تنها بهبود پایدار Any/subtype "
            "AUC ارزش ادامه دارد. اقدام بعدی: outer فقط پس از عبور گیت ranking و "
            "تأیید هویت فضایی calibration مجاز است."
        )
    if config.sah_residual_adapter:
        start_message = (
            "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
            "🔬 غربال residual انتخابی background/IPH→SAH آغاز شد. مدل پایه فریز "
            "است و head صفرمقدار فقط می‌تواند برندهٔ فعلی background یا IPH را به "
            "SAH تبدیل کند؛ SAH قبلی حذف نمی‌شود. تحلیل کوتاه: train-only selectivity "
            "پاس شده، اما precision دقیقاً روی مرز ۵۰٪ بود؛ بنابراین این تنها یک "
            "calibration قفل‌شده است و IPH safety هم‌زمان با SAH gain گیت می‌شود."
            if config.sah_include_incumbent_iph
            else "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
            "🔬 غربال residual اختصاصی SAH آغاز شد. مدل پایه و چهار زیرنوع غیرهدف "
            "فریز هستند؛ head صفرمقدار فقط می‌تواند پیکسل‌های پس‌زمینهٔ مدل پایه "
            "را به SAH تبدیل کند و حق حذف SAH قبلی یا تغییر IVH/IPH/SDH/EDH را ندارد."
        )
    _notify_non_smoke(
        run_kind,
        "start",
        start_message,
        run=config.run_name,
        kind=run_kind,
        architecture=f"{config.architecture}/{config.encoder_name}",
        fold=f"outer={config.outer_fold}, calibration={config.calibration_fold}",
        train_slices=len(train_frame),
        spatially_supervised_slices=int(train_frame["segmentation_known"].sum()),
        empty_foreground_weight=f"{config.empty_foreground_weight:.3f}",
        empty_top_fraction=f"{config.empty_foreground_top_fraction:.4f}",
        checkpoint_selection=config.checkpoint_selection_strategy,
        sampler_study_balance_power=f"{config.sampler_study_balance_power:.2f}",
        sampler_weight_max=f"{sampler_diagnostics['weight_max']:.2f}",
        hard_negative_multiplier=f"{config.hard_negative_multiplier:.2f}",
        hard_negative_slices=sampler_diagnostics["hard_negative_slices"],
        hard_negative_studies=sampler_diagnostics["hard_negative_studies"],
        initialization=(
            "warm_start_verified_same_split"
            if config.initial_checkpoint
            else "imagenet_encoder"
        ),
        symmetry_adapter=(
            "zero_initialized_frozen_base"
            if config.horizontal_symmetry_adapter and config.freeze_base_model
            else "enabled"
            if config.horizontal_symmetry_adapter
            else "disabled"
        ),
        five_slice_context_adapter=(
            "zero_initialized_frozen_base"
            if config.five_slice_context_adapter and config.freeze_base_model
            else "enabled"
            if config.five_slice_context_adapter
            else "disabled"
        ),
        sah_residual_adapter=(
            "background_to_sah_only_zero_initialized_frozen_base"
            if config.sah_residual_adapter
            else "disabled"
        ),
        sah_residual_hidden_channels=config.sah_residual_hidden_channels,
        sah_maximum_logit_residual=f"{config.sah_maximum_logit_residual:.2f}",
        sah_include_incumbent_iph=config.sah_include_incumbent_iph,
        slice_context_radius=config.slice_context_radius,
        classification_head_only=config.classification_head_only,
        ivh_center_loss_weight=f"{config.ivh_center_loss_weight:.3f}",
        ivh_center_square_size=f"{config.ivh_center_square_size}×{config.ivh_center_square_size}",
        physical_volume_loss_weight=f"{config.physical_volume_loss_weight:.4f}",
        diffuse_tversky_loss_weight=f"{config.diffuse_tversky_loss_weight:.4f}",
        sah_tversky_loss_weight=f"{config.sah_tversky_loss_weight:.4f}",
        sah_positive_pixel_loss_weight=f"{config.sah_positive_pixel_loss_weight:.4f}",
    )

    model = build_segmentation_model(
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
        horizontal_symmetry_adapter=config.horizontal_symmetry_adapter,
        five_slice_context_adapter=config.five_slice_context_adapter,
        sah_residual_adapter=config.sah_residual_adapter,
        sah_residual_hidden_channels=config.sah_residual_hidden_channels,
        sah_maximum_logit_residual=config.sah_maximum_logit_residual,
        sah_include_incumbent_iph=config.sah_include_incumbent_iph,
    ).to(device)
    initial_payload = None
    if config.initial_checkpoint:
        initial_payload = load_initial_segmentation_checkpoint(
            model, config.initial_checkpoint, config
        )
    trainable_parameters = configure_trainable_parameters(
        model,
        freeze_base_model=config.freeze_base_model,
        classification_head_only=config.classification_head_only,
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    pos_weight = segmentation_classification_weights(
        train_frame, maximum=config.maximum_pos_weight
    ).to(device)
    segmentation_class_weights = segmentation_foreground_weights(
        train_frame,
        power=config.segmentation_class_weight_power,
        maximum=config.maximum_segmentation_class_weight,
        basis=config.segmentation_class_weight_basis,
    ).to(device)
    segmentation_class_counts = segmentation_foreground_counts(
        train_frame,
        basis=config.segmentation_class_weight_basis,
    ).to(device)
    loss_fn = ICH25DSegmentationLoss(
        classification_pos_weight=pos_weight,
        segmentation_class_weights=segmentation_class_weights,
        segmentation_class_counts=segmentation_class_counts,
        classification_weight=config.classification_loss_weight,
        classification_focal_gamma=config.classification_focal_gamma,
        background_weight=config.background_weight,
        empty_foreground_weight=config.empty_foreground_weight,
        empty_foreground_top_fraction=config.empty_foreground_top_fraction,
        ivh_center_loss_weight=config.ivh_center_loss_weight,
        physical_volume_loss_weight=config.physical_volume_loss_weight,
        diffuse_tversky_loss_weight=config.diffuse_tversky_loss_weight,
        sah_tversky_loss_weight=config.sah_tversky_loss_weight,
        sah_positive_pixel_loss_weight=config.sah_positive_pixel_loss_weight,
        segmentation_objective=config.segmentation_objective,
        foreground_dice_weight=config.foreground_dice_weight,
        foreground_focal_weight=config.foreground_focal_weight,
        conditional_subtype_weight=config.conditional_subtype_weight,
        subtype_ovr_weight=config.subtype_ovr_weight,
        conditional_subtype_mode=config.conditional_subtype_mode,
        foreground_gradient_mode=config.foreground_gradient_mode,
    ).to(device)
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_dice = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "2p5d_multitask_segmentation",
                "run_kind": run_kind,
                "git_commit": git_commit(),
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
                "outer_fold_policy": "untouched_until_best_checkpoint",
                "initialization": (
                    "warm_start_verified_same_split"
                    if config.initial_checkpoint
                    else "imagenet_encoder"
                ),
            })
            mlflow.log_params({
                **asdict(config),
                "train_slices": len(train_frame),
                "train_segmentation_known_slices": int(
                    train_frame["segmentation_known"].sum()
                ),
                "train_classification_known_slices": int(
                    train_frame["classification_known"].sum()
                ),
                "calibration_slices": len(calibration_frame),
                "outer_slices": len(outer_frame),
                "train_studies": train_frame["study_id"].nunique(),
                "calibration_studies": calibration_frame["study_id"].nunique(),
                "outer_studies": outer_frame["study_id"].nunique(),
                "manifest_sha256": manifest_sha,
                "hard_negative_manifest_sha256": (
                    hard_negative_manifest_sha or "disabled"
                ),
                "initial_checkpoint_sha256": initial_checkpoint_sha or "disabled",
                "trainable_parameter_count": trainable_parameter_count,
                "metadata_source": str(metadata_source),
                "positive_class_weights": json.dumps(pos_weight.cpu().tolist()),
                "segmentation_class_weights": json.dumps(
                    segmentation_class_weights.cpu().tolist()
                ),
                "segmentation_class_counts": json.dumps(
                    segmentation_class_counts.cpu().tolist()
                ),
                **{
                    f"sampler_{key}": value
                    for key, value in sampler_diagnostics.items()
                },
            })

            if initial_payload is not None:
                calibration_slices = _predict_slices(
                    model, calibration_loader, device=device
                )
                calibration_studies, calibration_summary = (
                    summarize_segmentation_predictions(calibration_slices, truth)
                )
                best_score = checkpoint_selection_score(
                    calibration_summary, config.checkpoint_selection_strategy
                )
                best_dice = float(calibration_summary["mean_foreground_dice"])
                best_epoch = 0
                initial_metrics: dict[str, object] = {
                    "epoch": 0,
                    "learning_rate": float(config.learning_rate),
                    "calibration_checkpoint_score": best_score,
                    **_flatten_summary_metrics(
                        "calibration", calibration_summary
                    ),
                }
                history.append(initial_metrics)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in initial_metrics.items()
                        if key != "epoch"
                    },
                    step=0,
                )
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    config=config,
                    epoch=0,
                    calibration_summary=calibration_summary,
                    manifest_sha256=manifest_sha,
                    hard_negative_manifest_sha256=hard_negative_manifest_sha,
                )
                calibration_slices.to_csv(
                    output / "best_calibration_slice_predictions.csv", index=False
                )
                calibration_studies.to_csv(
                    output / "best_calibration_study_predictions.csv", index=False
                )
                (output / "best_calibration_summary.json").write_text(
                    json.dumps(calibration_summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(json.dumps(initial_metrics, sort_keys=True), flush=True)

            for epoch in range(1, config.epochs + 1):
                set_segmentation_training_mode(
                    model,
                    freeze_base_model=config.freeze_base_model,
                    classification_head_only=config.classification_head_only,
                )
                component_history: dict[str, list[float]] = {
                    name: [] for name in (
                        "loss",
                        "segmentation",
                        "dice",
                        "focal",
                        "empty_foreground",
                        "foreground_dice",
                        "foreground_focal",
                        "conditional_subtype",
                        "subtype_ovr",
                        "classification",
                        "ivh_center",
                        "physical_volume",
                        "physical_volume_subtype",
                        "physical_volume_total",
                        "diffuse_tversky",
                        "sah_tversky",
                        "sah_positive_pixel",
                    )
                }
                for step, batch in enumerate(train_loader, start=1):
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    segmentation_known = batch["segmentation_known"].to(
                        device, non_blocking=True
                    )
                    classification_known = batch["classification_known"].to(
                        device, non_blocking=True
                    )
                    ivh_center_targets = batch["ivh_center_target"].to(
                        device, non_blocking=True
                    )
                    voxel_volume_ml = batch["voxel_volume_ml"].to(
                        device, non_blocking=True
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        mask_logits, class_logits = _unpack_outputs(model(images))
                        components = loss_fn.components(
                            mask_logits,
                            class_logits,
                            masks,
                            targets,
                            segmentation_known,
                            classification_known,
                            ivh_center_targets=ivh_center_targets,
                            voxel_volume_ml=voxel_volume_ml,
                        )
                    components["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, max_norm=5.0
                    )
                    optimizer.step()
                    for name, value in components.items():
                        component_history[name].append(float(value.detach().cpu()))
                    if step % 50 == 0:
                        print(
                            f"epoch={epoch}/{config.epochs} step={step}/{len(train_loader)} "
                            f"loss={np.mean(component_history['loss'][-50:]):.5f}",
                            flush=True,
                        )
                    if config.max_train_steps and step >= config.max_train_steps:
                        break
                scheduler.step()

                calibration_slices = _predict_slices(
                    model, calibration_loader, device=device
                )
                calibration_studies, calibration_summary = summarize_segmentation_predictions(
                    calibration_slices, truth
                )
                score = checkpoint_selection_score(
                    calibration_summary, config.checkpoint_selection_strategy
                )
                dice = float(calibration_summary["mean_foreground_dice"])
                epoch_metrics: dict[str, object] = {
                    "epoch": epoch,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "calibration_checkpoint_score": score,
                    **{
                        f"train_{name}": float(np.mean(values))
                        for name, values in component_history.items()
                    },
                    **_flatten_summary_metrics("calibration", calibration_summary),
                }
                history.append(epoch_metrics)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics(
                    {
                        key: float(value) for key, value in epoch_metrics.items()
                        if key != "epoch"
                    },
                    step=epoch,
                )
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

                improved = score > best_score + 1e-6 or (
                    abs(score - best_score) <= 1e-6 and dice > best_dice + 1e-6
                )
                if improved:
                    best_score, best_dice, best_epoch = score, dice, epoch
                    stale_epochs = 0
                    _save_checkpoint(
                        checkpoint_path,
                        model,
                        config=config,
                        epoch=epoch,
                        calibration_summary=calibration_summary,
                        manifest_sha256=manifest_sha,
                        hard_negative_manifest_sha256=hard_negative_manifest_sha,
                    )
                    calibration_slices.to_csv(
                        output / "best_calibration_slice_predictions.csv", index=False
                    )
                    calibration_studies.to_csv(
                        output / "best_calibration_study_predictions.csv", index=False
                    )
                    (output / "best_calibration_summary.json").write_text(
                        json.dumps(calibration_summary, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    small_ivh = calibration_summary["subtypes"]["IVH"][
                        "volume_strata"
                    ]["small_le_2ml"]
                    if run_kind == "full_fold":
                        notify_campaign(
                            "checkpoint",
                            "checkpoint بهتر segmentation ICH ثبت شد. تحلیل کوتاه: انتخاب فقط براساس معیار ازپیش‌تعیین‌شده انجام شده و fold بیرونی هنوز دیده نشده است؛ Dice و حساسیت IVH کوچک صرفاً گیت توصیفی‌اند تا بهبود میانگین، افت ضایعات کم‌حجم را پنهان نکند. اقدام بعدی: ادامهٔ آموزش تا patience؛ outer فقط در اجرای کامل ارزیابی می‌شود.",
                            run=config.run_name,
                            epoch=epoch,
                            dice=f"{dice:.4f}",
                            any_auc=f"{float(calibration_summary['any_ich_study_auc'] or 0):.4f}",
                            volume_mae_ml=f"{float(calibration_summary['total_volume_mae_ml']):.3f}",
                            small_ivh_studies=small_ivh["positive_studies"],
                            small_ivh_dice=(
                                "n/a"
                                if small_ivh["dice_known_pixels"] is None
                                else f"{float(small_ivh['dice_known_pixels']):.4f}"
                            ),
                            small_ivh_sensitivity=(
                                "n/a"
                                if small_ivh[
                                    "presence_sensitivity_at_0_1ml"
                                ] is None
                                else f"{float(small_ivh['presence_sensitivity_at_0_1ml']):.4f}"
                            ),
                        )
                else:
                    stale_epochs += 1
                if config.patience > 0 and stale_epochs >= config.patience:
                    break
                if _should_stop_after_epoch(config):
                    break

            payload = load_segmentation_weights(model, checkpoint_path)
            outer_summary = None
            if _should_evaluate_outer(config):
                model.to(device).eval()
                outer_slices = _predict_slices(model, outer_loader, device=device)
                outer_studies, outer_summary = summarize_segmentation_predictions(
                    outer_slices, truth
                )
                outer_slices.to_csv(
                    output / "outer_slice_predictions.csv", index=False
                )
                outer_studies.to_csv(
                    output / "outer_study_predictions.csv", index=False
                )
                (output / "outer_summary.json").write_text(
                    json.dumps(outer_summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            duration = time.perf_counter() - started
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            summary = {
                "run_name": config.run_name,
                "run_id": run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
                "best_calibration_checkpoint_score": best_score,
                "best_calibration_selection_score": float(
                    payload["calibration_summary"]["selection_score"]
                ),
                "checkpoint_selection_strategy": config.checkpoint_selection_strategy,
                "sampler_diagnostics": sampler_diagnostics,
                "best_calibration_mean_foreground_dice": best_dice,
                "outer_evaluation_performed": outer_summary is not None,
                "outer_summary": outer_summary,
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "manifest_sha256": manifest_sha,
                "hard_negative_manifest_sha256": hard_negative_manifest_sha,
                "initial_checkpoint": config.initial_checkpoint,
                "initial_checkpoint_sha256": initial_checkpoint_sha,
                "initial_checkpoint_epoch": (
                    initial_payload.get("epoch")
                    if initial_payload is not None
                    else None
                ),
                "trainable_parameter_count": trainable_parameter_count,
                "git_commit": git_commit(),
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
            }
            final_metrics = {
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
            }
            if outer_summary is not None:
                final_metrics.update(
                    _flatten_summary_metrics("outer", outer_summary)
                )
            else:
                summary["calibration_summary"] = payload["calibration_summary"]
                if run_kind == "smoke":
                    summary["smoke_calibration_summary"] = payload[
                        "calibration_summary"
                    ]
            mlflow.log_metrics(final_metrics)
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            log_safe_mlflow_artifacts(output)

        if not _should_evaluate_outer(config):
            heldout_calibration = payload["calibration_summary"]
            small_ivh = heldout_calibration["subtypes"]["IVH"][
                "volume_strata"
            ]["small_le_2ml"]
            if run_kind == "calibration_screen":
                notify_campaign(
                    "success",
                    f"غربال کامل calibration-only مدل ICH تمام شد. selection={float(heldout_calibration['selection_score']):.4f}، Dice={float(heldout_calibration['mean_foreground_dice']):.4f}، Any-AUC={float(heldout_calibration['any_ich_study_auc'] or 0):.4f}، FPR نرمال={float(heldout_calibration['normal_false_positive_rate_at_0_1ml']):.4f} و MAE حجم={float(heldout_calibration['total_volume_mae_ml']):.3f}mL است. تحلیل کوتاه: outer fold خوانده نشده و این اعداد فقط گیت غربال‌اند؛ promotion فقط پس از مقایسهٔ هم‌split و عبور هم‌زمان معیارهای کلی و IVH ممکن است. اقدام بعدی: تولید گزارش اختلاف با baseline و تصمیم رد یا پنج-fold OOF.",
                    run=config.run_name,
                    kind=run_kind,
                    best_epoch=best_epoch,
                    selection=f"{float(heldout_calibration['selection_score']):.4f}",
                    dice=f"{float(heldout_calibration['mean_foreground_dice']):.4f}",
                    any_auc=f"{float(heldout_calibration['any_ich_study_auc'] or 0):.4f}",
                    presence_f1=f"{float(heldout_calibration['presence_f1_at_0_1ml']):.4f}",
                    normal_fpr=f"{float(heldout_calibration['normal_false_positive_rate_at_0_1ml']):.4f}",
                    volume_mae_ml=f"{float(heldout_calibration['total_volume_mae_ml']):.3f}",
                    small_ivh_studies=small_ivh["positive_studies"],
                    small_ivh_dice=(
                        "n/a"
                        if small_ivh["dice_known_pixels"] is None
                        else f"{float(small_ivh['dice_known_pixels']):.4f}"
                    ),
                    small_ivh_sensitivity=(
                        "n/a"
                        if small_ivh["presence_sensitivity_at_0_1ml"] is None
                        else f"{float(small_ivh['presence_sensitivity_at_0_1ml']):.4f}"
                    ),
                    peak_vram_gb=f"{peak_vram:.2f}",
                    duration_min=f"{duration / 60:.1f}",
                )
                return summary
            return summary

        outer_small_ivh = outer_summary["subtypes"]["IVH"]["volume_strata"][
            "small_le_2ml"
        ]
        notify_campaign(
            "success",
            f"آموزش segmentation دوبعدونیم ICH تمام شد. روی outer fold، selection={float(outer_summary['selection_score']):.4f}، Dice={float(outer_summary['mean_foreground_dice']):.4f}، Any-AUC={float(outer_summary['any_ich_study_auc'] or 0):.4f}، FPR نرمال={float(outer_summary['normal_false_positive_rate_at_0_1ml']):.4f} و MAE حجم={float(outer_summary['total_volume_mae_ml']):.3f}mL ثبت شد. تحلیل کوتاه: checkpoint فقط با calibration انتخاب و outer یک‌بار دیده شده است؛ بااین‌حال بالا بودن یک معیار به‌تنهایی مجوز promotion نیست و باید با reference همان split، FPR و خطای حجم هم‌زمان مقایسه شود. اقدام بعدی: ساخت جدول اختلاف با baseline هم‌fold و ادامه فقط در صورت بهبود قابل تکرار.",
            run=config.run_name,
            kind=run_kind,
            best_epoch=best_epoch,
            selection=f"{float(outer_summary['selection_score']):.4f}",
            dice=f"{float(outer_summary['mean_foreground_dice']):.4f}",
            any_auc=f"{float(outer_summary['any_ich_study_auc'] or 0):.4f}",
            presence_f1=f"{float(outer_summary['presence_f1_at_0_1ml']):.4f}",
            normal_fpr=f"{float(outer_summary['normal_false_positive_rate_at_0_1ml']):.4f}",
            volume_mae_ml=f"{float(outer_summary['total_volume_mae_ml']):.3f}",
            small_ivh_studies=outer_small_ivh["positive_studies"],
            small_ivh_dice=(
                "n/a"
                if outer_small_ivh["dice_known_pixels"] is None
                else f"{float(outer_small_ivh['dice_known_pixels']):.4f}"
            ),
            small_ivh_sensitivity=(
                "n/a"
                if outer_small_ivh["presence_sensitivity_at_0_1ml"] is None
                else f"{float(outer_small_ivh['presence_sensitivity_at_0_1ml']):.4f}"
            ),
            peak_vram_gb=f"{peak_vram:.2f}",
            duration_min=f"{duration / 60:.1f}",
        )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "آموزش segmentation دوبعدونیم ICH متوقف شد. تحلیل کوتاه: این رخداد تا بررسی traceback نتیجهٔ کیفیتی نیست و checkpoint ناقص پذیرفته نمی‌شود. اقدام بعدی: اصلاح کوچک‌ترین علت فنی و تکرار smoke test.",
            run=config.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
