"""
monai/dataset.py — MONAI 3D Dataset for ICH segmentation.

Loads NIfTI volumes from nnUNet_raw format and serves 3D patches
for MONAI network training with configurable transforms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from monai.data import Dataset as MonaiDataset
from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from src.config import NNUNET_RAW_DIR

logger = logging.getLogger(__name__)


def _find_dataset_folder(raw_dir: Path) -> Optional[Path]:
    """Locate the Dataset{id}_{name} folder inside nnUNet_raw."""
    candidates = sorted(raw_dir.glob("Dataset*_*"))
    return candidates[0] if candidates else None


def _collect_data_dicts(images_dir: Path, labels_dir: Path) -> list[dict]:
    """Build MONAI-compatible data dicts from NIfTI files."""
    image_paths = sorted(images_dir.glob("*_0000.nii.gz"))
    data_dicts = []
    for img_path in image_paths:
        label_path = labels_dir / img_path.name.replace("_0000.nii.gz", ".nii.gz")
        if label_path.exists():
            data_dicts.append({
                "image": str(img_path),
                "label": str(label_path),
            })
    return data_dicts


def create_monai_dataloaders(
    config,
    split: str = "train",
) -> DataLoader:
    """
    Create a MONAI DataLoader for ICH NIfTI data.

    Parameters
    ----------
    config : MONAIConfig
        Strategy configuration (roi_size, batch_size, augmentation).
    split : str
        "train" or "val".
    """
    raw_dir = Path(NNUNET_RAW_DIR)
    dataset_folder = _find_dataset_folder(raw_dir)
    if dataset_folder is None:
        raise FileNotFoundError(f"No Dataset* folder in {raw_dir}")

    images_dir = dataset_folder / "imagesTr"
    labels_dir = dataset_folder / "labelsTr"

    data_dicts = _collect_data_dicts(images_dir, labels_dir)
    if not data_dicts:
        raise FileNotFoundError("No valid image/label pairs found")

    # Train / val split (volume-level)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(data_dicts))
    val_count = max(1, int(len(data_dicts) * config.val_split))

    if split == "val":
        indices = indices[:val_count]
    else:
        indices = indices[val_count:]

    split_dicts = [data_dicts[i] for i in indices]

    roi = (config.roi_size, config.roi_size, config.roi_size)

    # ── Base transforms (applied to both train and val) ───────────
    base_transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    # ── Train-specific transforms ─────────────────────────────────
    train_transforms = base_transforms.copy()
    if split == "train":
        train_transforms += [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1, neg=1, num_samples=4,
                allow_smaller=True,        # † handles volumes < roi_size gracefully
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
            ToTensord(keys=["image", "label"]),
        ]
        if not config.augmentation:
            # Remove augmentation transforms when augmentation=False
            train_transforms = [
                t for t in train_transforms
                if not any(x in str(type(t)).lower() for x in ["randflip", "randscale", "randshift", "randgaussian"])
            ]
        train_pipeline = Compose(train_transforms)

    else:
        # ── Validation transforms ───────────────────────────────────
        # Use RandSpatialCropd(random_size=False) as a deterministic
        # central crop that handles volumes smaller than roi_size
        # (unlike CenterSpatialCropd which crashes on small volumes).
        from monai.transforms import RandSpatialCropd
        val_transforms = base_transforms + [
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=roi,
                random_size=False,
                allow_smaller=True,
            ),
            ToTensord(keys=["image", "label"]),
        ]
        train_pipeline = Compose(val_transforms)

    dataset = MonaiDataset(data=split_dicts, transform=train_pipeline)
    shuffle = split == "train"
    # batch_size=1 is required because allow_smaller=True causes crops
    # from different volumes to have different spatial sizes, and
    # PyTorch cannot stack tensors of unequal shape in a batch.
    # With num_samples=4 each step processes 4 crops from 1 volume,
    # which provides sufficient gradient diversity.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(
        "MONAI %s DataLoader: %d volumes, batch_size=1, roi=%s, num_samples=4",
        split, len(dataset), roi,
    )
    return loader
