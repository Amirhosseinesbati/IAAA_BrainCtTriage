"""Frozen-encoder temporal residual head for ICH study ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
)
from src.strategies.ich_2p5d.segmentation_model import base_segmentation_model
from src.strategies.ich_v2.evaluation import VOLUME_KEYS


class TemporalResidualHead(torch.nn.Module):
    """Add a zero-initialized bidirectional-GRU residual to legacy slice logits."""

    def __init__(
        self,
        feature_dim: int,
        *,
        projection_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(feature_dim, projection_dim, hidden_dim) < 1:
            raise ValueError("Temporal head dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.normalization = torch.nn.LayerNorm(self.feature_dim)
        self.projection = torch.nn.Linear(self.feature_dim, self.projection_dim)
        self.recurrent = torch.nn.GRU(
            self.projection_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = torch.nn.Linear(
            2 * self.hidden_dim, len(OUTPUT_LABELS)
        )
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Temporal features have an unexpected shape")
        if base_logits.shape != (*features.shape[:2], len(OUTPUT_LABELS)):
            raise ValueError("Temporal base logits have an unexpected shape")
        if lengths.ndim != 1 or len(lengths) != len(features):
            raise ValueError("Temporal lengths have an unexpected shape")
        projected = F.gelu(self.projection(self.normalization(features.float())))
        packed = pack_padded_sequence(
            projected,
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
        return base_logits.float() + self.residual(self.dropout(encoded))


def temporal_classification_loss(
    logits: torch.Tensor,
    slice_targets: torch.Tensor,
    classification_known: torch.Tensor,
    study_targets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    slice_pos_weight: torch.Tensor,
    study_pos_weight: torch.Tensor,
    study_loss_weight: float = 0.5,
    focal_gamma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Average slice supervision per study and add max-pooled study supervision."""
    if study_loss_weight < 0 or focal_gamma < 0:
        raise ValueError("Temporal loss weights must be non-negative")
    steps = torch.arange(logits.shape[1], device=logits.device)[None, :]
    padding_mask = steps < lengths[:, None]
    slice_mask = padding_mask & (classification_known > 0.5)
    if not torch.all(slice_mask.any(dim=1)):
        raise ValueError("Every sequence needs at least one known classification slice")

    bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        slice_targets.float(),
        pos_weight=slice_pos_weight,
        reduction="none",
    )
    probabilities = torch.sigmoid(logits.float())
    correct_probability = (
        slice_targets * probabilities
        + (1.0 - slice_targets) * (1.0 - probabilities)
    )
    focal = bce * (1.0 - correct_probability).pow(focal_gamma)
    active = slice_mask[:, :, None].expand_as(focal)
    per_study_slice = (focal * active).sum(dim=(1, 2)) / active.sum(
        dim=(1, 2)
    ).clamp_min(1)
    slice_loss = per_study_slice.mean()

    masked_logits = logits.float().masked_fill(~padding_mask[:, :, None], -1e9)
    study_logits = masked_logits.amax(dim=1)
    study_bce = F.binary_cross_entropy_with_logits(
        study_logits,
        study_targets.float(),
        pos_weight=study_pos_weight,
        reduction="none",
    )
    study_probability = torch.sigmoid(study_logits)
    study_correct = (
        study_targets * study_probability
        + (1.0 - study_targets) * (1.0 - study_probability)
    )
    study_loss = (
        study_bce * (1.0 - study_correct).pow(focal_gamma)
    ).mean()
    return {
        "loss": slice_loss + study_loss_weight * study_loss,
        "slice": slice_loss,
        "study": study_loss,
    }


def extract_frozen_encoder_features(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    device: torch.device,
    batch_size: int = 32,
    workers: int = 4,
) -> dict[str, Any]:
    """Persist deterministic non-augmented encoder features outside Git."""
    base = base_segmentation_model(model)
    encoder = getattr(base, "encoder", None)
    classification_head = getattr(base, "classification_head", None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(
        classification_head, torch.nn.Module
    ):
        raise ValueError("Base model does not expose encoder/classification_head")
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
    targets: list[np.ndarray] = []
    known: list[np.ndarray] = []
    study_ids: list[str] = []
    patient_ids: list[str] = []
    slice_indices: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                deepest = encoder(images)[-1]
                class_logits = classification_head(deepest)
            pooled = F.adaptive_avg_pool2d(deepest.float(), 1).flatten(1)
            embeddings.append(pooled.cpu().numpy().astype(np.float16))
            logits.append(class_logits.float().cpu().numpy().astype(np.float32))
            targets.append(batch["target"].numpy().astype(np.float32))
            known.append(batch["classification_known"].numpy().astype(np.float32))
            study_ids.extend(str(value) for value in batch["study_id"])
            patient_ids.extend(str(value) for value in batch["patient_id"])
            slice_indices.extend(int(value) for value in batch["slice_index"])
    if len(study_ids) != len(ordered):
        raise RuntimeError("Feature extraction did not cover every requested slice")
    expected_keys = list(
        zip(
            ordered["study_id"].astype(str),
            ordered["slice_index"].astype(int),
            strict=True,
        )
    )
    observed_keys = list(zip(study_ids, slice_indices, strict=True))
    if observed_keys != expected_keys:
        raise RuntimeError("Feature extraction changed slice order")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    embedding_array = np.concatenate(embeddings)
    np.savez_compressed(
        output,
        embeddings=embedding_array,
        base_logits=np.concatenate(logits),
        slice_targets=np.concatenate(targets),
        classification_known=np.concatenate(known),
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
    }


class ICHSequenceFeatureDataset(Dataset):
    """Variable-length study sequences backed by a compact feature cache."""

    def __init__(self, cache_path: str | Path, truth: pd.DataFrame) -> None:
        with np.load(Path(cache_path), allow_pickle=False) as cache:
            self.embeddings = torch.from_numpy(cache["embeddings"].astype(np.float32))
            self.base_logits = torch.from_numpy(cache["base_logits"].astype(np.float32))
            self.slice_targets = torch.from_numpy(
                cache["slice_targets"].astype(np.float32)
            )
            self.classification_known = torch.from_numpy(
                cache["classification_known"].astype(np.float32)
            )
            study_ids = cache["study_id"].astype(str)
            slice_indices = cache["slice_index"].astype(np.int64)
        if not (
            len(self.embeddings)
            == len(self.base_logits)
            == len(self.slice_targets)
            == len(self.classification_known)
            == len(study_ids)
        ):
            raise ValueError("Temporal feature cache arrays have different lengths")
        truth = truth.copy()
        truth["study_id"] = truth["study_id"].astype(str)
        truth_map = truth.set_index("study_id")
        self.study_ids: list[str] = []
        self.indices: list[torch.Tensor] = []
        self.study_targets: list[torch.Tensor] = []
        for study_id in sorted(np.unique(study_ids)):
            indices = np.flatnonzero(study_ids == study_id)
            indices = indices[np.argsort(slice_indices[indices])]
            if study_id not in truth_map.index:
                raise ValueError(f"Temporal truth is missing study {study_id}")
            volumes = truth_map.loc[study_id, [f"gt_{key}" for key in VOLUME_KEYS]]
            subtype = (np.asarray(volumes, dtype=np.float64) > 0).astype(np.float32)
            target = np.concatenate([[float(np.any(subtype))], subtype])
            self.study_ids.append(study_id)
            self.indices.append(torch.as_tensor(indices, dtype=torch.long))
            self.study_targets.append(torch.from_numpy(target))

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        indices = self.indices[index]
        return {
            "study_id": self.study_ids[index],
            "features": self.embeddings[indices],
            "base_logits": self.base_logits[indices],
            "slice_targets": self.slice_targets[indices],
            "classification_known": self.classification_known[indices],
            "study_target": self.study_targets[index],
        }


def collate_ich_sequences(items: list[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty sequence batch")
    lengths = torch.as_tensor(
        [len(item["features"]) for item in items], dtype=torch.long
    )
    maximum = int(lengths.max())
    batch = len(items)
    feature_dim = int(items[0]["features"].shape[-1])
    labels = len(OUTPUT_LABELS)
    features = torch.zeros((batch, maximum, feature_dim), dtype=torch.float32)
    base_logits = torch.zeros((batch, maximum, labels), dtype=torch.float32)
    slice_targets = torch.zeros((batch, maximum, labels), dtype=torch.float32)
    known = torch.zeros((batch, maximum), dtype=torch.float32)
    for index, item in enumerate(items):
        length = int(lengths[index])
        features[index, :length] = item["features"]
        base_logits[index, :length] = item["base_logits"]
        slice_targets[index, :length] = item["slice_targets"]
        known[index, :length] = item["classification_known"]
    return {
        "study_id": [str(item["study_id"]) for item in items],
        "features": features,
        "base_logits": base_logits,
        "slice_targets": slice_targets,
        "classification_known": known,
        "study_target": torch.stack([item["study_target"] for item in items]),
        "lengths": lengths,
    }


def auc_summary(truth: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    if truth.shape != scores.shape or truth.shape[1] != len(OUTPUT_LABELS):
        raise ValueError("Temporal truth/scores have unexpected shapes")

    def safe_auc(column: int) -> float | None:
        if len(np.unique(truth[:, column])) < 2:
            return None
        return float(roc_auc_score(truth[:, column], scores[:, column]))

    any_auc = safe_auc(0)
    subtype_auc = {
        label: safe_auc(index)
        for index, label in enumerate(OUTPUT_LABELS[1:], start=1)
    }
    available = [value for value in subtype_auc.values() if value is not None]
    return {
        "any_ich_auc": any_auc,
        "subtype_auc": subtype_auc,
        "macro_subtype_auc": float(np.mean(available)) if available else None,
    }
