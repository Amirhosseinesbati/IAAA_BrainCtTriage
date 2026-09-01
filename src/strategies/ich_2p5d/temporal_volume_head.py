"""Frozen-encoder temporal residual head for official ICH study volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.config import TRIAGE_THRESHOLDS
from src.strategies.ich_2p5d.cache import CLASS_IDS, OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
)
from src.strategies.ich_2p5d.segmentation_model import base_segmentation_model
from src.strategies.ich_v2.evaluation import VOLUME_KEYS


SUBTYPE_LABELS = OUTPUT_LABELS[1:]
TRIAGE_VOLUME_THRESHOLDS_ML = {
    "edh_critical": float(TRIAGE_THRESHOLDS["EDH_CRIT"]),
    "sdh_critical": float(TRIAGE_THRESHOLDS["SDH_CRIT"]),
    "iph_critical": float(TRIAGE_THRESHOLDS["IPH_CRIT"]),
    "total_fracture_combo": float(TRIAGE_THRESHOLDS["FRAC_VOL_CRIT"]),
    "total_mls_combo": float(TRIAGE_THRESHOLDS["COMBO_VOL"]),
    "total_critical": float(TRIAGE_THRESHOLDS["TOTAL_VOL_CRIT"]),
}


def forward_frozen_segmentation_components(
    base: torch.nn.Module, images: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Match the installed SMP forward contract while exposing encoder features."""
    features = base.encoder(images)
    if not isinstance(features, (list, tuple)) or not features:
        raise TypeError("Base segmentation encoder must return a feature sequence")
    feature_list = list(features)
    decoded = base.decoder(feature_list)
    mask_logits = base.segmentation_head(decoded)
    class_logits = base.classification_head(feature_list[-1])
    return feature_list, mask_logits, class_logits


class TemporalVolumeResidualHead(torch.nn.Module):
    """Predict a bounded log-volume residual while preserving the base at zero."""

    def __init__(
        self,
        feature_dim: int,
        *,
        projection_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        maximum_log_residual: float = 4.0,
    ) -> None:
        super().__init__()
        if min(feature_dim, projection_dim, hidden_dim) < 1:
            raise ValueError("Temporal volume dimensions must be positive")
        if maximum_log_residual <= 0:
            raise ValueError("maximum_log_residual must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_log_residual = float(maximum_log_residual)
        self.normalization = torch.nn.LayerNorm(self.feature_dim)
        self.projection = torch.nn.Linear(self.feature_dim, self.projection_dim)
        signal_dim = 2 * len(SUBTYPE_LABELS)
        self.recurrent = torch.nn.GRU(
            self.projection_dim + signal_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = torch.nn.Linear(
            2 * self.hidden_dim, len(SUBTYPE_LABELS)
        )
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        base_slice_volumes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Temporal volume features have an unexpected shape")
        if base_logits.shape != (*features.shape[:2], len(OUTPUT_LABELS)):
            raise ValueError("Temporal volume base logits have an unexpected shape")
        if base_slice_volumes.shape != (
            *features.shape[:2],
            len(SUBTYPE_LABELS),
        ):
            raise ValueError("Temporal base slice volumes have an unexpected shape")
        if lengths.ndim != 1 or len(lengths) != len(features):
            raise ValueError("Temporal volume lengths have an unexpected shape")
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("Temporal volume lengths must fit the padded sequence")
        if torch.any(base_slice_volumes < 0):
            raise ValueError("Base slice volumes must be non-negative")

        projected = F.gelu(self.projection(self.normalization(features.float())))
        base_signals = torch.cat(
            [
                torch.log1p(base_slice_volumes.float()),
                torch.sigmoid(base_logits[..., 1:].float()),
            ],
            dim=-1,
        )
        packed = pack_padded_sequence(
            torch.cat([projected, base_signals], dim=-1),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=features.shape[1],
        )
        raw_residual = self.residual(self.dropout(encoded))
        bounded_residual = self.maximum_log_residual * torch.tanh(
            raw_residual / self.maximum_log_residual
        )
        # This algebra makes residual==0 an exactly bit-identical identity while
        # still allowing a positive residual to recover a missed zero-volume slice.
        candidate = base_slice_volumes.float() + (
            base_slice_volumes.float() + 1.0
        ) * torch.expm1(bounded_residual)
        return candidate.clamp_min(0.0)


def temporal_volume_loss(
    candidate_slice_volumes: torch.Tensor,
    target_slice_volumes: torch.Tensor,
    spatial_known: torch.Tensor,
    study_target_volumes: torch.Tensor,
    lengths: torch.Tensor,
    *,
    slice_pos_weight: torch.Tensor,
    study_pos_weight: torch.Tensor,
    study_loss_weight: float = 0.75,
    total_loss_weight: float = 0.25,
    huber_beta: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Combine supervised per-slice area with study and total-volume losses."""
    if min(study_loss_weight, total_loss_weight) < 0 or huber_beta <= 0:
        raise ValueError("Temporal volume loss weights are invalid")
    if (
        candidate_slice_volumes.ndim != 3
        or candidate_slice_volumes.shape[-1] != len(SUBTYPE_LABELS)
    ):
        raise ValueError("Candidate slice volumes have an unexpected shape")
    if target_slice_volumes.shape != candidate_slice_volumes.shape:
        raise ValueError("Target slice volumes have an unexpected shape")
    if study_target_volumes.shape != (
        len(candidate_slice_volumes),
        len(SUBTYPE_LABELS),
    ):
        raise ValueError("Study target volumes have an unexpected shape")
    if slice_pos_weight.shape != (len(SUBTYPE_LABELS),):
        raise ValueError("Slice positive weights have an unexpected shape")
    if study_pos_weight.shape != (len(SUBTYPE_LABELS),):
        raise ValueError("Study positive weights have an unexpected shape")
    if spatial_known.shape != candidate_slice_volumes.shape[:2]:
        raise ValueError("Spatial-known mask has an unexpected shape")
    if lengths.shape != (len(candidate_slice_volumes),):
        raise ValueError("Temporal volume lengths have an unexpected shape")
    if (
        torch.any(lengths <= 0)
        or torch.any(lengths > candidate_slice_volumes.shape[1])
    ):
        raise ValueError("Temporal volume lengths must fit the padded sequence")
    volume_tensors = (
        candidate_slice_volumes,
        target_slice_volumes,
        study_target_volumes,
    )
    if any(not torch.isfinite(value).all() for value in volume_tensors):
        raise ValueError("Temporal volume tensors must be finite")
    if any(torch.any(value < 0) for value in volume_tensors):
        raise ValueError("Temporal volume tensors must be non-negative")
    if (
        not torch.isfinite(slice_pos_weight).all()
        or not torch.isfinite(study_pos_weight).all()
        or torch.any(slice_pos_weight < 1)
        or torch.any(study_pos_weight < 1)
    ):
        raise ValueError("Temporal volume positive weights must be finite and >= 1")

    steps = torch.arange(
        candidate_slice_volumes.shape[1], device=candidate_slice_volumes.device
    )[None, :]
    padding = steps < lengths[:, None]
    active = padding & (spatial_known > 0.5)
    if not torch.all(active.any(dim=1)):
        raise ValueError("Every sequence needs at least one spatially known slice")

    slice_error = F.smooth_l1_loss(
        torch.log1p(candidate_slice_volumes.float()),
        torch.log1p(target_slice_volumes.float()),
        beta=huber_beta,
        reduction="none",
    )
    slice_weight = 1.0 + (
        slice_pos_weight[None, None, :] - 1.0
    ) * (target_slice_volumes > 0).float()
    slice_active = active[:, :, None].float()
    slice_per_class = (
        slice_error * slice_weight * slice_active
    ).sum(dim=(0, 1)) / (slice_weight * slice_active).sum(dim=(0, 1)).clamp_min(1.0)
    slice_loss = slice_per_class.mean()

    candidate_study = (
        candidate_slice_volumes.float() * padding[:, :, None]
    ).sum(dim=1)
    study_error = F.smooth_l1_loss(
        torch.log1p(candidate_study),
        torch.log1p(study_target_volumes.float()),
        beta=huber_beta,
        reduction="none",
    )
    study_weight = 1.0 + (
        study_pos_weight[None, :] - 1.0
    ) * (study_target_volumes > 0).float()
    study_loss = ((study_error * study_weight).sum(dim=0) / study_weight.sum(
        dim=0
    ).clamp_min(1.0)).mean()

    total_loss = F.smooth_l1_loss(
        torch.log1p(candidate_study.sum(dim=1)),
        torch.log1p(study_target_volumes.float().sum(dim=1)),
        beta=huber_beta,
        reduction="mean",
    )
    return {
        "loss": slice_loss
        + study_loss_weight * study_loss
        + total_loss_weight * total_loss,
        "slice": slice_loss,
        "study": study_loss,
        "total": total_loss,
    }


def extract_frozen_encoder_volume_features(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    device: torch.device,
    batch_size: int = 16,
    workers: int = 4,
) -> dict[str, Any]:
    """Cache deterministic features and physical slice volumes outside Git."""
    base = base_segmentation_model(model)
    required_modules = {
        name: getattr(base, name, None)
        for name in (
            "encoder",
            "decoder",
            "segmentation_head",
            "classification_head",
        )
    }
    missing = [
        name for name, value in required_modules.items()
        if not isinstance(value, torch.nn.Module)
    ]
    if missing:
        raise ValueError(f"Base model lacks required modules: {missing}")

    ordered = frame.sort_values(["study_id", "slice_index"]).reset_index(drop=True)
    loader = DataLoader(
        ICHAdjacentSegmentationDataset(ordered, augment=False, context_radius=1),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    base_volumes: list[np.ndarray] = []
    target_volumes: list[np.ndarray] = []
    spatial_known: list[np.ndarray] = []
    study_ids: list[str] = []
    patient_ids: list[str] = []
    slice_indices: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features, mask_logits, class_logits = (
                    forward_frozen_segmentation_components(base, images)
                )
            predicted = mask_logits.float().argmax(dim=1).cpu()
            observed = batch["mask"]
            voxel_volume = batch["voxel_volume_ml"].float()[:, None]
            predicted_counts = torch.stack(
                [
                    (predicted == class_id).sum(dim=(-2, -1))
                    for class_id in CLASS_IDS
                ],
                dim=1,
            ).float()
            observed_counts = torch.stack(
                [
                    (observed == class_id).sum(dim=(-2, -1))
                    for class_id in CLASS_IDS
                ],
                dim=1,
            ).float()
            deepest = features[-1]
            pooled = F.adaptive_avg_pool2d(deepest.float(), 1).flatten(1)
            embeddings.append(pooled.cpu().numpy().astype(np.float16))
            logits.append(class_logits.float().cpu().numpy().astype(np.float32))
            base_volumes.append(
                (predicted_counts * voxel_volume).numpy().astype(np.float32)
            )
            target_volumes.append(
                (observed_counts * voxel_volume).numpy().astype(np.float32)
            )
            spatial_known.append(
                batch["segmentation_known"].numpy().astype(np.float32)
            )
            study_ids.extend(str(value) for value in batch["study_id"])
            patient_ids.extend(str(value) for value in batch["patient_id"])
            slice_indices.extend(int(value) for value in batch["slice_index"])

    expected_keys = list(
        zip(
            ordered["study_id"].astype(str),
            ordered["slice_index"].astype(int),
            strict=True,
        )
    )
    observed_keys = list(zip(study_ids, slice_indices, strict=True))
    if observed_keys != expected_keys:
        raise RuntimeError("Volume feature extraction changed slice order")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    embedding_array = np.concatenate(embeddings)
    np.savez_compressed(
        output,
        embeddings=embedding_array,
        base_logits=np.concatenate(logits),
        base_slice_volumes=np.concatenate(base_volumes),
        target_slice_volumes=np.concatenate(target_volumes),
        spatial_known=np.concatenate(spatial_known),
        study_id=np.asarray(study_ids),
        patient_id=np.asarray(patient_ids),
        slice_index=np.asarray(slice_indices, dtype=np.int32),
    )
    return {
        "path": str(output),
        "slices": int(len(study_ids)),
        "studies": int(len(set(study_ids))),
        "feature_dim": int(embedding_array.shape[1]),
        "dtype": str(embedding_array.dtype),
        "cache_schema": "ich_temporal_volume_v1",
    }


class ICHSequenceVolumeDataset(Dataset):
    """Variable-length volume sequences backed by a compact feature cache."""

    def __init__(self, cache_path: str | Path, truth: pd.DataFrame) -> None:
        with np.load(Path(cache_path), allow_pickle=False) as cache:
            self.embeddings = torch.from_numpy(cache["embeddings"].astype(np.float32))
            self.base_logits = torch.from_numpy(cache["base_logits"].astype(np.float32))
            self.base_slice_volumes = torch.from_numpy(
                cache["base_slice_volumes"].astype(np.float32)
            )
            self.target_slice_volumes = torch.from_numpy(
                cache["target_slice_volumes"].astype(np.float32)
            )
            self.spatial_known = torch.from_numpy(
                cache["spatial_known"].astype(np.float32)
            )
            study_ids = cache["study_id"].astype(str)
            slice_indices = cache["slice_index"].astype(np.int64)
        lengths = {
            len(self.embeddings),
            len(self.base_logits),
            len(self.base_slice_volumes),
            len(self.target_slice_volumes),
            len(self.spatial_known),
            len(study_ids),
        }
        if len(lengths) != 1:
            raise ValueError("Temporal volume cache arrays have different lengths")
        truth = truth.copy()
        truth["study_id"] = truth["study_id"].astype(str)
        if truth["study_id"].duplicated().any():
            raise ValueError("Temporal volume truth must have one row per study")
        truth_map = truth.set_index("study_id")
        self.study_ids: list[str] = []
        self.indices: list[torch.Tensor] = []
        self.study_target_volumes: list[torch.Tensor] = []
        for study_id in sorted(np.unique(study_ids)):
            indices = np.flatnonzero(study_ids == study_id)
            indices = indices[np.argsort(slice_indices[indices])]
            if study_id not in truth_map.index:
                raise ValueError(f"Temporal volume truth is missing study {study_id}")
            volumes = truth_map.loc[
                study_id, [f"gt_{key}" for key in VOLUME_KEYS]
            ]
            self.study_ids.append(study_id)
            self.indices.append(torch.as_tensor(indices, dtype=torch.long))
            self.study_target_volumes.append(
                torch.as_tensor(np.asarray(volumes, dtype=np.float32))
            )

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        indices = self.indices[index]
        return {
            "study_id": self.study_ids[index],
            "features": self.embeddings[indices],
            "base_logits": self.base_logits[indices],
            "base_slice_volumes": self.base_slice_volumes[indices],
            "target_slice_volumes": self.target_slice_volumes[indices],
            "spatial_known": self.spatial_known[indices],
            "study_target_volumes": self.study_target_volumes[index],
        }


def collate_ich_volume_sequences(
    items: list[dict[str, object]],
) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty volume sequence batch")
    lengths = torch.as_tensor(
        [len(item["features"]) for item in items], dtype=torch.long
    )
    maximum = int(lengths.max())
    batch = len(items)
    feature_dim = int(items[0]["features"].shape[-1])
    subtype_count = len(SUBTYPE_LABELS)
    features = torch.zeros((batch, maximum, feature_dim), dtype=torch.float32)
    base_logits = torch.zeros(
        (batch, maximum, len(OUTPUT_LABELS)), dtype=torch.float32
    )
    base_slice_volumes = torch.zeros(
        (batch, maximum, subtype_count), dtype=torch.float32
    )
    target_slice_volumes = torch.zeros_like(base_slice_volumes)
    spatial_known = torch.zeros((batch, maximum), dtype=torch.float32)
    for index, item in enumerate(items):
        length = int(lengths[index])
        features[index, :length] = item["features"]
        base_logits[index, :length] = item["base_logits"]
        base_slice_volumes[index, :length] = item["base_slice_volumes"]
        target_slice_volumes[index, :length] = item["target_slice_volumes"]
        spatial_known[index, :length] = item["spatial_known"]
    return {
        "study_id": [str(item["study_id"]) for item in items],
        "features": features,
        "base_logits": base_logits,
        "base_slice_volumes": base_slice_volumes,
        "target_slice_volumes": target_slice_volumes,
        "spatial_known": spatial_known,
        "study_target_volumes": torch.stack(
            [item["study_target_volumes"] for item in items]
        ),
        "lengths": lengths,
    }


def volume_summary(
    truth: np.ndarray, predicted: np.ndarray
) -> dict[str, object]:
    """Summarize official-volume quality without MLS or fracture dependencies."""
    if (
        truth.ndim != 2
        or predicted.ndim != 2
        or truth.shape != predicted.shape
        or truth.shape[1] != len(SUBTYPE_LABELS)
        or truth.shape[0] == 0
    ):
        raise ValueError("Temporal volume truth/predictions have unexpected shapes")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Temporal volume arrays must be finite")
    if np.any(truth < 0) or np.any(predicted < 0):
        raise ValueError("Temporal volumes must be non-negative")

    gt_total = truth.sum(axis=1)
    pred_total = predicted.sum(axis=1)
    gt_any = gt_total > 0.0
    pred_any = pred_total >= 0.1
    false_positive_rate = float(
        np.mean(pred_any[~gt_any]) if np.any(~gt_any) else 0.0
    )
    sensitivity = float(
        np.mean(pred_any[gt_any]) if np.any(gt_any) else 0.0
    )

    def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
        return (
            None
            if len(np.unique(target)) < 2
            else float(roc_auc_score(target.astype(np.uint8), score))
        )

    subtypes: dict[str, dict[str, float | int | None]] = {}
    for index, label in enumerate(SUBTYPE_LABELS):
        target = truth[:, index]
        score = predicted[:, index]
        target_present = target > 0.0
        predicted_present = score >= 0.1
        spearman = (
            float(pd.Series(score).corr(pd.Series(target), method="spearman"))
            if len(np.unique(target)) > 1 and len(np.unique(score)) > 1
            else None
        )
        subtypes[label] = {
            "positive_studies": int(np.count_nonzero(target_present)),
            "mae_ml": float(np.mean(np.abs(score - target))),
            "bias_ml": float(np.mean(score - target)),
            "presence_f1_at_0_1ml": float(
                f1_score(target_present, predicted_present, zero_division=0)
            ),
            "volume_auc": safe_auc(target_present, score),
            "spearman_volume": spearman,
        }

    trigger_pairs = {
        "edh_critical": (
            truth[:, SUBTYPE_LABELS.index("EDH")],
            predicted[:, SUBTYPE_LABELS.index("EDH")],
        ),
        "sdh_critical": (
            truth[:, SUBTYPE_LABELS.index("SDH")],
            predicted[:, SUBTYPE_LABELS.index("SDH")],
        ),
        "iph_critical": (
            truth[:, SUBTYPE_LABELS.index("IPH")],
            predicted[:, SUBTYPE_LABELS.index("IPH")],
        ),
        "total_fracture_combo": (gt_total, pred_total),
        "total_mls_combo": (gt_total, pred_total),
        "total_critical": (gt_total, pred_total),
    }
    trigger_f1: dict[str, float | None] = {}
    for name, (target_volume, predicted_volume) in trigger_pairs.items():
        threshold = TRIAGE_VOLUME_THRESHOLDS_ML[name]
        target_trigger = target_volume >= threshold
        predicted_trigger = predicted_volume >= threshold
        trigger_f1[name] = (
            float(f1_score(target_trigger, predicted_trigger, zero_division=0))
            if np.any(target_trigger)
            else None
        )
    available_trigger_f1 = [
        value for value in trigger_f1.values() if value is not None
    ]
    return {
        "studies": int(len(truth)),
        "total_volume_mae_ml": float(np.mean(np.abs(pred_total - gt_total))),
        "total_volume_bias_ml": float(np.mean(pred_total - gt_total)),
        "presence_f1_at_0_1ml": float(
            f1_score(gt_any, pred_any, zero_division=0)
        ),
        "normal_false_positive_rate_at_0_1ml": false_positive_rate,
        "presence_sensitivity_at_0_1ml": sensitivity,
        "critical_trigger_macro_f1": float(np.mean(available_trigger_f1))
        if available_trigger_f1
        else None,
        "triage_volume_trigger_f1": trigger_f1,
        "triage_volume_thresholds_ml": TRIAGE_VOLUME_THRESHOLDS_ML,
        "subtypes": subtypes,
    }
