"""Datasets for direct 2.5D ICH segmentation and physical-volume prediction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .cache import OUTPUT_LABELS
from .data import load_slice_manifest


SEGMENTATION_MANIFEST_COLUMNS = {
    "label_cache_path",
    "resized_voxel_volume_ml",
    "native_height",
    "native_width",
    "resized_affine_voxel_volume_ml",
    "slice_spacing_mm",
    "slice_thickness_mm",
    "spacing_to_thickness_ratio",
    "classification_known",
    "segmentation_known",
    "metadata_missing",
    "supervision_mismatch",
}


def load_segmentation_manifest(path: str | Path) -> pd.DataFrame:
    frame = load_slice_manifest(path)
    missing = SEGMENTATION_MANIFEST_COLUMNS - set(frame)
    if missing:
        raise ValueError(
            "2.5D segmentation cache is incomplete; rebuild it to add: "
            f"{sorted(missing)}"
        )
    if (pd.to_numeric(frame["resized_voxel_volume_ml"], errors="coerce") <= 0).any():
        raise ValueError("Every cached slice needs a positive physical voxel volume")
    return frame


def split_segmentation_slices(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    calibration_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep all evaluation and classification-known training slices.

    Classification-only rows stay in training while their spatial loss is
    masked by ``segmentation_known`` inside the multi-task objective.
    """
    if outer_fold == calibration_fold:
        raise ValueError("Outer and calibration folds must differ")
    training = frame.loc[
        (~frame["fold"].isin([outer_fold, calibration_fold]))
        & (frame["classification_known"] == 1)
    ].copy()
    calibration = frame.loc[frame["fold"] == calibration_fold].copy()
    outer = frame.loc[frame["fold"] == outer_fold].copy()
    patient_sets = [set(part["patient_id"]) for part in (training, calibration, outer)]
    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise ValueError("Patient leakage detected in 2.5D segmentation split")
    if not len(training) or not len(calibration) or not len(outer):
        raise ValueError("Every 2.5D segmentation split must be non-empty")
    return tuple(
        part.sort_values(["study_id", "slice_index"]).reset_index(drop=True)
        for part in (training, calibration, outer)
    )


class ICHAdjacentSegmentationDataset(Dataset):
    """Nine-channel adjacent input with a categorical center-slice mask."""

    def __init__(self, frame: pd.DataFrame, *, augment: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augment = bool(augment)
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _array(self, path: str) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r")
        return self._arrays[path]

    def _augment(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if random.random() < 0.70:
            height, width = image.shape[-2:]
            angle = random.uniform(-10.0, 10.0)
            translate = [
                random.randint(-max(1, int(width * 0.04)), max(1, int(width * 0.04))),
                random.randint(-max(1, int(height * 0.04)), max(1, int(height * 0.04))),
            ]
            scale = random.uniform(0.92, 1.08)
            image = vision_functional.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = vision_functional.affine(
                mask.unsqueeze(0).float(),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze(0).long()
        if random.random() < 0.25:
            image = (image + torch.randn_like(image) * 0.01).clamp_(0.0, 1.0)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_array = self._array(str(row["cache_path"]))
        label_array = self._array(str(row["label_cache_path"]))
        anchor = int(row["slice_index"])
        positions = [
            max(0, min(len(image_array) - 1, anchor + offset))
            for offset in (-1, 0, 1)
        ]
        image = np.concatenate([image_array[position] for position in positions], axis=0)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
        mask = torch.from_numpy(
            np.ascontiguousarray(label_array[anchor]).copy()
        ).long()
        if self.augment:
            image_tensor, mask = self._augment(image_tensor, mask)
        image_tensor = (image_tensor - 0.5) / 0.25
        target = torch.tensor(
            [float(row[name]) for name in OUTPUT_LABELS], dtype=torch.float32
        )
        return {
            "image": image_tensor,
            "mask": mask,
            "target": target,
            "known": torch.tensor(
                float(row["segmentation_known"]), dtype=torch.float32
            ),
            "segmentation_known": torch.tensor(
                float(row["segmentation_known"]), dtype=torch.float32
            ),
            "classification_known": torch.tensor(
                float(row["classification_known"]), dtype=torch.float32
            ),
            "voxel_volume_ml": torch.tensor(
                float(row["resized_voxel_volume_ml"]), dtype=torch.float32
            ),
            "study_id": str(row["study_id"]),
            "patient_id": str(row["patient_id"]),
            "slice_index": anchor,
        }


def subtype_aware_sampler(frame: pd.DataFrame, *, seed: int) -> WeightedRandomSampler:
    positives = frame.loc[:, OUTPUT_LABELS[1:]].to_numpy(dtype=np.float64)
    counts = positives.sum(axis=0)
    if np.any(counts <= 0):
        missing = [label for label, count in zip(OUTPUT_LABELS[1:], counts) if count <= 0]
        raise ValueError(f"Training split lacks positive slices for: {missing}")
    common_positive = float(counts.max())
    rare_weights = np.clip(np.sqrt(common_positive / counts), 1.0, 5.0)
    weights = np.ones(len(frame), dtype=np.float64)
    for index, row in enumerate(positives):
        present = rare_weights[row > 0]
        if len(present):
            weights[index] = 2.0 + float(present.max())
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
        generator=generator,
    )


def segmentation_classification_weights(
    frame: pd.DataFrame, *, maximum: float = 20.0
) -> torch.Tensor:
    positives = frame.loc[:, OUTPUT_LABELS].sum(axis=0).to_numpy(dtype=np.float64)
    negatives = len(frame) - positives
    if np.any(positives <= 0):
        raise ValueError("All auxiliary classification channels need positive slices")
    return torch.as_tensor(
        np.clip(negatives / positives, 1.0, maximum), dtype=torch.float32
    )


def create_segmentation_loaders(
    manifest_path: str | Path,
    *,
    outer_fold: int = 0,
    calibration_fold: int = 1,
    batch_size: int = 8,
    workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_segmentation_manifest(manifest_path)
    training, calibration, outer = split_segmentation_slices(
        frame, outer_fold=outer_fold, calibration_fold=calibration_fold
    )
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ICHAdjacentSegmentationDataset(training, augment=True),
        batch_size=batch_size,
        sampler=subtype_aware_sampler(training, seed=seed),
        **common,
    )
    evaluation_batch_size = max(batch_size, min(batch_size * 2, 16))
    calibration_loader = DataLoader(
        ICHAdjacentSegmentationDataset(calibration),
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    outer_loader = DataLoader(
        ICHAdjacentSegmentationDataset(outer),
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, calibration_loader, outer_loader, training, calibration, outer
