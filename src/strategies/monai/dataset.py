"""
monai/dataset.py — MONAI 3D Dataset for ICH segmentation.

Loads NIfTI volumes from the generic NIfTI directory (*ICH_NIFTI_DIR*)
and serves 3D patches for MONAI network training with configurable
transforms.

The data is expected to have been produced by :class:`NiftiDatasetBuilder`
(``src/preprocessing/builders/nifti_builder.py``), **not** by the nnUNet
builder — this avoids any coupling to nnUNet naming conventions.
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
    Compose,
    CropForegroundd,
    DivisiblePadd,
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

from src.config import ICH_NIFTI_DIR
from src.evaluation.splits import split_items_by_fold, study_id_from_path
from src.strategies.augmentation_config import MONAIAugmentationConfig

logger = logging.getLogger(__name__)


def _find_dataset_folder(data_dir: Path) -> Optional[Path]:
    """Locate the dataset folder inside *ICH_NIFTI_DIR*.

    Looks for subdirectories that contain ``images/`` and ``labels/``
    (the generic NIfTI layout produced by :class:`NiftiDatasetBuilder`).
    """
    for candidate in sorted(data_dir.iterdir()):
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    return None


def _collect_data_dicts(images_dir: Path, labels_dir: Path) -> list[dict]:
    """Build MONAI-compatible data dicts from generic NIfTI files.

    Expects matching stems in *images_dir* and *labels_dir* (assumes the
    generic naming produced by :class:`NiftiDatasetBuilder`, e.g.
    ``BRN_{pid}.nii.gz`` in both directories).
    """
    image_paths = sorted(images_dir.glob("*.nii.gz"))
    data_dicts = []
    for img_path in image_paths:
        label_path = labels_dir / img_path.name
        if label_path.exists():
            data_dicts.append({
                "image": str(img_path),
                "label": str(label_path),
            })
    return data_dicts


def _build_monai_train_transforms(
    aug_config: MONAIAugmentationConfig,
    base_transforms: list,
    roi,
) -> list:
    """Build MONAI train transforms from ``MONAIAugmentationConfig``."""
    transforms = base_transforms.copy()
    transforms.append(
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi,
            pos=1, neg=1, num_samples=4,
            allow_smaller=True,
        ),
    )

    if aug_config.flip_axis_0.enabled:
        transforms.append(RandFlipd(keys=["image", "label"], prob=aug_config.flip_axis_0.prob, spatial_axis=0))
    if aug_config.flip_axis_1.enabled:
        transforms.append(RandFlipd(keys=["image", "label"], prob=aug_config.flip_axis_1.prob, spatial_axis=1))
    if aug_config.flip_axis_2.enabled:
        transforms.append(RandFlipd(keys=["image", "label"], prob=aug_config.flip_axis_2.prob, spatial_axis=2))
    if aug_config.scale_intensity.enabled:
        transforms.append(RandScaleIntensityd(keys=["image"], factors=0.1, prob=aug_config.scale_intensity.prob))
    if aug_config.shift_intensity.enabled:
        transforms.append(RandShiftIntensityd(keys=["image"], offsets=0.1, prob=aug_config.shift_intensity.prob))
    if aug_config.gaussian_noise.enabled:
        transforms.append(RandGaussianNoised(keys=["image"], prob=aug_config.gaussian_noise.prob, std=0.01))

    transforms.append(ToTensord(keys=["image", "label"]))
    return transforms


def create_monai_dataloaders(
    config,
    split: str = "train",
) -> DataLoader:
    """
    Create a MONAI DataLoader for ICH NIfTI data.

    Parameters
    ----------
    config : MONAIConfig
        Strategy configuration (roi_size, batch_size, augmentation,
        model_dimension, slices_per_stack).
    split : str
        "train" or "val".
    """
    nifti_dir = Path(ICH_NIFTI_DIR)
    dataset_folder = _find_dataset_folder(nifti_dir)
    if dataset_folder is None:
        raise FileNotFoundError(
            f"No dataset folder found in {nifti_dir}. "
            "Run NiftiDatasetBuilder().build() first."
        )

    images_dir = dataset_folder / "images"
    labels_dir = dataset_folder / "labels"

    data_dicts = _collect_data_dicts(images_dir, labels_dir)
    if not data_dicts:
        raise FileNotFoundError("No valid image/label pairs found")

    # All serious experiments share the immutable study/patient-grouped fold
    # manifest. The random fallback exists only for explicitly requested smoke
    # tests and must not be used for OOF model selection.
    if getattr(config, "use_competition_folds", True):
        training, validation = split_items_by_fold(
            data_dicts,
            int(config.fold),
            id_getter=lambda item: study_id_from_path(item["image"]),
        )
        split_dicts = validation if split == "val" else training
    else:
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(data_dicts))
        val_count = max(1, int(len(data_dicts) * config.val_split))
        selected = indices[:val_count] if split == "val" else indices[val_count:]
        split_dicts = [data_dicts[i] for i in selected]

    # Adjust patch depth based on model dimension
    if getattr(config, "model_dimension", "3D") == "2.5D":
        depth = getattr(config, "slices_per_stack", 3) or 3
        roi = (config.roi_size, config.roi_size, depth)
    else:
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
        # SwinUNETR requires spatial dims divisible by patch_size**5
        # (e.g. patch_size=(2,2,2) → divisible by 32 in every dim).
        # DivisiblePadd ensures this without manual roi_size constraints.
        DivisiblePadd(keys=["image", "label"], k=32, method="end"),
    ]

    # ── Train-specific transforms ─────────────────────────────────
    if split == "train":
        aug_config = getattr(config, "augmentation_config", None)
        if aug_config is not None and isinstance(aug_config, MONAIAugmentationConfig):
            train_transforms = _build_monai_train_transforms(aug_config, base_transforms, roi)
        elif not config.augmentation:
            # Augmentation disabled via legacy field — keep only base + crop + totensor
            train_transforms = base_transforms.copy()
            train_transforms += [
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi,
                    pos=1, neg=1, num_samples=4,
                    allow_smaller=True,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        else:
            # Legacy fallback: hardcoded defaults
            train_transforms = base_transforms.copy()
            train_transforms += [
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi,
                    pos=1, neg=1, num_samples=4,
                    allow_smaller=True,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
                ToTensord(keys=["image", "label"]),
            ]
        train_pipeline = Compose(train_transforms)

    else:
        # ── Validation transforms ───────────────────────────────────
        # No random/center crop — the model processes the full volume
        # (after CropForegroundd).  This avoids the `allow_smaller`
        # headache: every existing volume is already foreground-cropped
        # and models like SwinUNETR / UNETR are fully convolutional, so
        # they gracefully handle any spatial shape.
        val_transforms = base_transforms + [
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

    dim_label = getattr(config, "model_dimension", "3D")
    logger.info(
        "MONAI %s DataLoader: %d volumes, batch_size=1, roi=%s, "
        "num_samples=4, dimension=%s",
        split, len(dataset), roi, dim_label,
    )
    return loader
