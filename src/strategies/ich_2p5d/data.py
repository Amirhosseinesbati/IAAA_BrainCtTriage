"""Slice datasets and leakage-safe outer/calibration splits for ICH 2.5D."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .cache import OUTPUT_LABELS


def load_slice_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
    required = {
        "study_id", "patient_id", "fold", "slice_index", "slice_count",
        "known", "cache_path", *OUTPUT_LABELS,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"2.5D manifest is missing columns: {sorted(missing)}")
    if "classification_known" not in frame:
        frame["classification_known"] = frame["known"]
    if "segmentation_known" not in frame:
        frame["segmentation_known"] = frame["known"]
    return frame


def split_known_slices(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    calibration_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if outer_fold == calibration_fold:
        raise ValueError("Outer and calibration folds must differ")
    if "classification_known" not in frame:
        if "known" not in frame:
            raise ValueError("2.5D split needs classification_known or legacy known")
        frame = frame.copy()
        frame["classification_known"] = frame["known"]
    known = frame.loc[frame["classification_known"] == 1].copy()
    outer = known.loc[known["fold"] == outer_fold].copy()
    calibration = known.loc[known["fold"] == calibration_fold].copy()
    training = known.loc[~known["fold"].isin([outer_fold, calibration_fold])].copy()
    patient_sets = [set(part["patient_id"]) for part in (training, calibration, outer)]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise ValueError("Patient leakage detected in 2.5D train/calibration/outer split")
    return tuple(
        part.sort_values(["study_id", "slice_index"]).reset_index(drop=True)
        for part in (training, calibration, outer)
    )


class ICHAdjacentSliceDataset(Dataset):
    """Three adjacent slices, each represented by three registered CT windows."""

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

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        array = self._array(str(row["cache_path"]))
        anchor = int(row["slice_index"])
        positions = [max(0, min(len(array) - 1, anchor + offset)) for offset in (-1, 0, 1)]
        image = np.concatenate([array[position] for position in positions], axis=0)
        image = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=(-1,))
            if random.random() < 0.5:
                size = image.shape[-1]
                crop_size = random.randint(int(size * 0.9), size)
                top = random.randint(0, size - crop_size)
                left = random.randint(0, size - crop_size)
                image = F.interpolate(
                    image[:, top:top + crop_size, left:left + crop_size].unsqueeze(0),
                    size=(size, size), mode="bilinear", align_corners=False,
                ).squeeze(0)
            if random.random() < 0.25:
                image = (image + torch.randn_like(image) * 0.01).clamp_(0.0, 1.0)
        image = (image - 0.5) / 0.25
        target = torch.tensor(
            [float(row[name]) for name in OUTPUT_LABELS], dtype=torch.float32
        )
        return {
            "image": image,
            "target": target,
            "study_id": str(row["study_id"]),
            "slice_index": anchor,
        }


def balanced_sampler(frame: pd.DataFrame, *, seed: int) -> WeightedRandomSampler:
    positive = frame["any_ich"].to_numpy(dtype=np.int64) > 0
    counts = np.bincount(positive.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training split must contain positive and negative ICH slices")
    weights = np.where(positive, 0.5 / counts[1], 0.5 / counts[0])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
        generator=generator,
    )


def create_loaders(
    manifest_path: str | Path,
    *,
    outer_fold: int = 0,
    calibration_fold: int = 1,
    batch_size: int = 32,
    workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_slice_manifest(manifest_path)
    training, calibration, outer = split_known_slices(
        frame, outer_fold=outer_fold, calibration_fold=calibration_fold
    )
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        ICHAdjacentSliceDataset(training, augment=True),
        batch_size=batch_size,
        sampler=balanced_sampler(training, seed=seed),
        **common,
    )
    calibration_loader = DataLoader(
        ICHAdjacentSliceDataset(calibration),
        batch_size=batch_size * 2,
        shuffle=False,
        **common,
    )
    outer_loader = DataLoader(
        ICHAdjacentSliceDataset(outer),
        batch_size=batch_size * 2,
        shuffle=False,
        **common,
    )
    return train_loader, calibration_loader, outer_loader, training, calibration, outer


def positive_class_weights(frame: pd.DataFrame, maximum: float = 20.0) -> torch.Tensor:
    positives = frame.loc[:, OUTPUT_LABELS].sum(axis=0).to_numpy(dtype=np.float64)
    negatives = len(frame) - positives
    if np.any(positives <= 0):
        raise ValueError("Every 2.5D output needs a positive training slice")
    return torch.as_tensor(np.clip(negatives / positives, 1.0, maximum), dtype=torch.float32)
