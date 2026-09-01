"""Datasets for direct 2.5D ICH segmentation and physical-volume prediction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as connected_components
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .cache import CLASS_IDS, OUTPUT_LABELS
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
HARD_NEGATIVE_MANIFEST_COLUMNS = {
    "study_id",
    "patient_id",
    "slice_index",
    "source_outer_fold",
    "ground_truth_any_ich",
    "predicted_foreground_pixels",
}

IVH_CLASS_ID = OUTPUT_LABELS.index("IVH")
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def ivh_center_target(mask: torch.Tensor, *, square_size: int) -> torch.Tensor:
    """Create equal-area center squares for every 8-connected IVH component."""
    if square_size == 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    if square_size < 1 or square_size % 2 == 0:
        raise ValueError("IVH center square size must be zero or a positive odd integer")
    binary = mask.detach().cpu().numpy() == IVH_CLASS_ID
    output = torch.zeros_like(mask, dtype=torch.float32)
    if not np.any(binary):
        return output
    labels, count = connected_components(binary, structure=CONNECTIVITY_8)
    half = square_size // 2
    height, width = binary.shape
    if square_size > min(height, width):
        raise ValueError("IVH center square cannot exceed the mask dimensions")
    for component_id in range(1, count + 1):
        coordinates = np.argwhere(labels == component_id)
        center_y, center_x = np.rint(coordinates.mean(axis=0)).astype(np.int64)
        y_start = min(max(0, center_y - half), height - square_size)
        x_start = min(max(0, center_x - half), width - square_size)
        y_stop, x_stop = y_start + square_size, x_start + square_size
        output[y_start:y_stop, x_start:x_stop] = 1.0
    return output


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


def load_hard_negative_slice_manifest(path: str | Path) -> pd.DataFrame:
    """Load leakage-auditable slice keys mined from patient-disjoint OOF predictions."""
    frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
    missing = HARD_NEGATIVE_MANIFEST_COLUMNS - set(frame)
    if missing:
        raise ValueError(
            "Hard-negative manifest is incomplete; missing: "
            f"{sorted(missing)}"
        )
    frame["slice_index"] = pd.to_numeric(frame["slice_index"], errors="raise").astype(int)
    frame["source_outer_fold"] = pd.to_numeric(
        frame["source_outer_fold"], errors="raise"
    ).astype(int)
    if not frame["source_outer_fold"].isin(range(5)).all():
        raise ValueError("Hard-negative source_outer_fold must be in [0, 4]")
    if not (pd.to_numeric(frame["ground_truth_any_ich"], errors="raise") == 0).all():
        raise ValueError("Hard-negative manifest contains an AnyICH-positive row")
    if not (pd.to_numeric(frame["predicted_foreground_pixels"], errors="raise") > 0).all():
        raise ValueError("Every hard-negative row must contain predicted foreground")
    if frame.duplicated(["study_id", "slice_index"]).any():
        raise ValueError("Hard-negative manifest contains duplicate slice keys")
    return frame.sort_values(
        ["source_outer_fold", "study_id", "slice_index"]
    ).reset_index(drop=True)


def oof_hard_negative_row_mask(
    frame: pd.DataFrame,
    hard_negative_slices: pd.DataFrame | None,
) -> np.ndarray:
    """Match OOF hard-negative keys and verify their fold/label provenance."""
    if hard_negative_slices is None:
        return np.zeros(len(frame), dtype=bool)
    required_frame = {
        "study_id",
        "patient_id",
        "slice_index",
        "fold",
        *OUTPUT_LABELS[1:],
    }
    missing = required_frame - set(frame)
    if missing:
        raise ValueError(
            "Training frame cannot validate hard negatives; missing: "
            f"{sorted(missing)}"
        )
    hard = hard_negative_slices.copy()
    missing_hard = HARD_NEGATIVE_MANIFEST_COLUMNS - set(hard)
    if missing_hard:
        raise ValueError(
            "Hard-negative manifest is incomplete; missing: "
            f"{sorted(missing_hard)}"
        )
    hard["study_id"] = hard["study_id"].astype(str)
    if hard.duplicated(["study_id", "slice_index"]).any():
        raise ValueError("Hard-negative manifest contains duplicate slice keys")
    if not (pd.to_numeric(hard["ground_truth_any_ich"], errors="raise") == 0).all():
        raise ValueError("Hard-negative manifest contains an AnyICH-positive row")
    if not (
        pd.to_numeric(hard["predicted_foreground_pixels"], errors="raise") > 0
    ).all():
        raise ValueError("Every hard-negative row must contain predicted foreground")
    source_fold = {
        (str(row.study_id), int(row.slice_index)): int(row.source_outer_fold)
        for row in hard.itertuples(index=False)
    }
    source_patient = {
        (str(row.study_id), int(row.slice_index)): str(row.patient_id)
        for row in hard.itertuples(index=False)
    }
    keys = [
        (str(study_id), int(slice_index))
        for study_id, slice_index in zip(
            frame["study_id"], frame["slice_index"], strict=True
        )
    ]
    matched = np.asarray([key in source_fold for key in keys], dtype=bool)
    if not np.any(matched):
        return matched
    positives = frame.loc[:, OUTPUT_LABELS[1:]].to_numpy(dtype=np.float64).any(axis=1)
    if np.any(positives & matched):
        raise ValueError("An OOF hard-negative key maps to a positive training slice")
    folds = frame["fold"].to_numpy(dtype=int)
    patients = frame["patient_id"].astype(str).to_numpy()
    for index in np.flatnonzero(matched):
        if folds[index] != source_fold[keys[index]]:
            raise ValueError("Hard-negative source fold disagrees with training manifest")
        if patients[index] != source_patient[keys[index]]:
            raise ValueError("Hard-negative patient disagrees with training manifest")
    return matched


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

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        augment: bool = False,
        ivh_center_square_size: int = 0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augment = bool(augment)
        if ivh_center_square_size < 0 or (
            ivh_center_square_size % 2 == 0 and ivh_center_square_size != 0
        ):
            raise ValueError(
                "IVH center square size must be zero or a positive odd integer"
            )
        self.ivh_center_square_size = int(ivh_center_square_size)
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
        center_target = ivh_center_target(
            mask,
            square_size=self.ivh_center_square_size,
        )
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
            "ivh_center_target": center_target,
            "voxel_volume_ml": torch.tensor(
                float(row["resized_voxel_volume_ml"]), dtype=torch.float32
            ),
            "study_id": str(row["study_id"]),
            "patient_id": str(row["patient_id"]),
            "slice_index": anchor,
        }


def subtype_aware_sampling_weights(
    frame: pd.DataFrame,
    *,
    study_balance_power: float = 0.0,
    hard_negative_slices: pd.DataFrame | None = None,
    hard_negative_multiplier: float = 1.0,
) -> torch.Tensor:
    """Return subtype-aware slice weights with optional study equalization.

    ``study_balance_power=0`` exactly preserves the original slice-balanced
    sampler. Positive weights at larger powers are divided by the number of
    positive slices for the active subtype within that study, then normalized
    so total positive sampling mass stays unchanged.
    """
    if not 0.0 <= study_balance_power <= 1.0:
        raise ValueError("sampler study-balance power must be in [0, 1]")
    if not 1.0 <= hard_negative_multiplier <= 10.0:
        raise ValueError("hard-negative multiplier must be in [1, 10]")
    if hard_negative_multiplier > 1.0 and hard_negative_slices is None:
        raise ValueError("hard-negative multiplier requires an OOF slice manifest")
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
    positive_rows = positives.any(axis=1)
    if study_balance_power > 0.0:
        if "study_id" not in frame:
            raise ValueError("Study-balanced sampling requires study_id")
        study_counts = (
            pd.DataFrame(positives, columns=OUTPUT_LABELS[1:])
            .assign(study_id=frame["study_id"].astype(str).to_numpy())
            .groupby("study_id")[list(OUTPUT_LABELS[1:])]
            .transform("sum")
            .to_numpy(dtype=np.float64)
        )
        candidates = np.zeros_like(positives, dtype=np.float64)
        for subtype_index, rare_weight in enumerate(rare_weights):
            active = positives[:, subtype_index] > 0
            candidates[active, subtype_index] = (
                2.0 + rare_weight
            ) / np.power(study_counts[active, subtype_index], study_balance_power)
        balanced_positive = candidates.max(axis=1)
        original_positive_mass = float(weights[positive_rows].sum())
        balanced_positive_mass = float(balanced_positive[positive_rows].sum())
        if balanced_positive_mass <= 0:
            raise ValueError("Study-balanced sampling produced no positive mass")
        weights[positive_rows] = (
            balanced_positive[positive_rows]
            * original_positive_mass
            / balanced_positive_mass
        )
    hard_negative_rows = oof_hard_negative_row_mask(frame, hard_negative_slices)
    if hard_negative_multiplier > 1.0:
        if not np.any(hard_negative_rows):
            raise ValueError("No OOF hard-negative slices occur in this training split")
        negative_rows = ~positive_rows
        original_negative_mass = float(weights[negative_rows].sum())
        weights[hard_negative_rows] *= hard_negative_multiplier
        reweighted_negative_mass = float(weights[negative_rows].sum())
        if reweighted_negative_mass <= 0:
            raise ValueError("Hard-negative sampling produced no negative mass")
        weights[negative_rows] *= original_negative_mass / reweighted_negative_mass
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Subtype-aware sampling weights must be finite and positive")
    return torch.as_tensor(weights, dtype=torch.double)


def subtype_aware_sampler(
    frame: pd.DataFrame,
    *,
    seed: int,
    study_balance_power: float = 0.0,
    hard_negative_slices: pd.DataFrame | None = None,
    hard_negative_multiplier: float = 1.0,
) -> WeightedRandomSampler:
    weights = subtype_aware_sampling_weights(
        frame,
        study_balance_power=study_balance_power,
        hard_negative_slices=hard_negative_slices,
        hard_negative_multiplier=hard_negative_multiplier,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
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


def segmentation_foreground_weights(
    frame: pd.DataFrame,
    *,
    power: float = 0.0,
    maximum: float = 8.0,
    basis: str = "slice",
) -> torch.Tensor:
    """Return rare-class weights from slice presence or supervised mask pixels."""
    if power < 0:
        raise ValueError("segmentation foreground-weight power cannot be negative")
    if maximum < 1:
        raise ValueError("maximum segmentation class weight must be at least one")
    if basis == "slice":
        counts = frame.loc[:, OUTPUT_LABELS[1:]].sum(axis=0).to_numpy(
            dtype=np.float64
        )
    elif basis == "pixel":
        required = {"label_cache_path", "slice_index", "segmentation_known"}
        missing = required - set(frame)
        if missing:
            raise ValueError(
                "Pixel-frequency weights require manifest columns: "
                f"{sorted(missing)}"
            )
        counts = np.zeros(len(CLASS_IDS), dtype=np.float64)
        supervised = frame.loc[frame["segmentation_known"] > 0.5]
        for label_cache_path, group in supervised.groupby(
            "label_cache_path", sort=False
        ):
            labels = np.load(str(label_cache_path), mmap_mode="r")
            for slice_index in group["slice_index"].to_numpy(dtype=np.int64):
                mask = np.asarray(labels[int(slice_index)])
                counts += np.bincount(
                    mask.reshape(-1), minlength=max(CLASS_IDS) + 1
                )[list(CLASS_IDS)]
    else:
        raise ValueError(
            "segmentation foreground-weight basis must be 'slice' or 'pixel'"
        )
    if np.any(counts <= 0):
        raise ValueError("Every foreground class needs positive training slices")
    weights = np.power(counts.max() / counts, power)
    return torch.as_tensor(np.clip(weights, 1.0, maximum), dtype=torch.float32)


def create_segmentation_loaders(
    manifest_path: str | Path,
    *,
    outer_fold: int = 0,
    calibration_fold: int = 1,
    batch_size: int = 8,
    workers: int = 2,
    seed: int = 42,
    sampler_study_balance_power: float = 0.0,
    hard_negative_slices: pd.DataFrame | None = None,
    hard_negative_multiplier: float = 1.0,
    ivh_center_square_size: int = 0,
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
        ICHAdjacentSegmentationDataset(
            training,
            augment=True,
            ivh_center_square_size=ivh_center_square_size,
        ),
        batch_size=batch_size,
        sampler=subtype_aware_sampler(
            training,
            seed=seed,
            study_balance_power=sampler_study_balance_power,
            hard_negative_slices=hard_negative_slices,
            hard_negative_multiplier=hard_negative_multiplier,
        ),
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
