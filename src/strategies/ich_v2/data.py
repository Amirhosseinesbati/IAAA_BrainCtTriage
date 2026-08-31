"""Fold-safe MONAI loaders for partial-label ICH training."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


DATA_KEYS = ("image", "label", "supervision")


def _balanced_limit(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.sort_values("study_id").reset_index(drop=True)
    if limit <= 0:
        raise ValueError("Study limit must be positive")
    groups = ["triage_class", "supervision_type"]
    shuffled = frame.sample(frac=1.0, random_state=seed)
    per_group = max(1, math.ceil(limit / shuffled.groupby(groups).ngroups))
    selected = shuffled.groupby(groups, group_keys=False).head(per_group)
    if len(selected) < limit:
        remaining = shuffled.loc[~shuffled.index.isin(selected.index)]
        selected = pd.concat([selected, remaining.head(limit - len(selected))])
    return selected.head(limit).sort_values("study_id").reset_index(drop=True)


def load_fold_frames(
    dataset_dir: str | Path,
    *,
    fold: int,
    max_train_studies: int | None = None,
    max_val_studies: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(
        Path(dataset_dir) / "manifest.csv",
        dtype={"study_id": str, "patient_id": str},
    )
    required = {
        "study_id", "patient_id", "fold", "triage_class", "supervision_type",
        *DATA_KEYS,
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"ICH-v2 manifest is missing columns: {sorted(missing)}")
    validation = manifest.loc[manifest["fold"] == fold].copy()
    training = manifest.loc[manifest["fold"] != fold].copy()
    train_patients = set(training["patient_id"])
    val_patients = set(validation["patient_id"])
    if train_patients & val_patients:
        raise ValueError("Patient leakage detected in ICH-v2 fold split")
    return (
        _balanced_limit(training, max_train_studies, seed),
        _balanced_limit(validation, max_val_studies, seed + 1),
    )


def _base_transforms() -> list[object]:
    return [
        LoadImaged(keys=DATA_KEYS, image_only=False),
        EnsureChannelFirstd(keys=DATA_KEYS),
        Orientationd(keys=DATA_KEYS, axcodes="RAS"),
        Spacingd(
            keys=DATA_KEYS,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image", a_min=-200, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=DATA_KEYS, source_key="image"),
    ]


def build_train_transform(
    *,
    roi_size: tuple[int, int, int],
    samples_per_volume: int,
    class_crop_ratios: tuple[float, float, float, float, float, float],
) -> Compose:
    transforms = _base_transforms()
    transforms.extend([
        SpatialPadd(keys=DATA_KEYS, spatial_size=roi_size, method="symmetric"),
        RandCropByLabelClassesd(
            keys=DATA_KEYS,
            label_key="label",
            spatial_size=roi_size,
            ratios=list(class_crop_ratios),
            num_classes=6,
            num_samples=samples_per_volume,
            image_key="image",
            image_threshold=0.0,
            allow_smaller=False,
            warn=False,
        ),
        RandFlipd(keys=DATA_KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=DATA_KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=DATA_KEYS, prob=0.2, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        RandGaussianNoised(keys="image", prob=0.15, std=0.01),
        ToTensord(keys=DATA_KEYS),
    ])
    return Compose(transforms)


def build_val_transform() -> Compose:
    return Compose([*_base_transforms(), ToTensord(keys=DATA_KEYS)])


def _items(frame: pd.DataFrame) -> list[dict[str, object]]:
    items = frame.loc[:, [
        "study_id", "patient_id", "fold", "triage_class", "supervision_type", *DATA_KEYS,
    ]].rename(columns={"supervision_type": "label_scope"})
    # MONAI Spacingd treats every ``<image-key>_*`` entry as legacy image
    # metadata.  Keeping the manifest name ``supervision_type`` here would
    # therefore collide with the actual ``supervision`` volume.
    return items.to_dict("records")


def create_loaders(
    dataset_dir: str | Path,
    *,
    fold: int,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    samples_per_volume: int = 2,
    background_crop_ratio: float = 4.0,
    foreground_crop_ratio: float = 2.0,
    workers: int = 0,
    max_train_studies: int | None = None,
    max_val_studies: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    training, validation = load_fold_frames(
        dataset_dir,
        fold=fold,
        max_train_studies=max_train_studies,
        max_val_studies=max_val_studies,
        seed=seed,
    )
    train_dataset = Dataset(
        data=_items(training),
        transform=build_train_transform(
            roi_size=roi_size,
            samples_per_volume=samples_per_volume,
            class_crop_ratios=(
                background_crop_ratio,
                foreground_crop_ratio,
                foreground_crop_ratio,
                foreground_crop_ratio,
                foreground_crop_ratio,
                foreground_crop_ratio,
            ),
        ),
    )
    val_dataset = Dataset(data=_items(validation), transform=build_val_transform())
    common = {
        "num_workers": workers,
        # On the rented container, MONAI MetaTensor batches plus multiple
        # workers can make PyTorch's pin-memory IPC thread lose its file
        # descriptor ("received 0 items of ancdata").  Direct pageable copies
        # are slightly slower but deterministic and avoid losing long runs.
        "pin_memory": False,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **common)
    return train_loader, val_loader, training, validation
