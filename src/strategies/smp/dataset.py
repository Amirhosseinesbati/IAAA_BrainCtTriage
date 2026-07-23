"""
smp/dataset.py — PyTorch Dataset for SMP 2D ICH segmentation.

Reads the NIfTI volumes produced by NNUnetDatasetBuilder and yields
2D slices with corresponding label masks. Supports on-the-fly
augmentation via Albumentations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import albumentations as A
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import ICH_LABELS, NNUNET_RAW_DIR

logger = logging.getLogger(__name__)


def _find_dataset_folder(raw_dir: Path) -> Optional[Path]:
    """Locate the Dataset{id}_{name} folder inside nnUNet_raw."""
    candidates = sorted(raw_dir.glob("Dataset*_*"))
    if not candidates:
        logger.error("No Dataset folder found in %s", raw_dir)
        return None
    return candidates[0]


class ICHEmbeddingDataset(Dataset):
    """
    2D slice-level dataset for ICH segmentation.

    Iterates over all NIfTI volumes in `imagesTr/`, extracts 2D axial
    slices, and returns (image, mask) pairs normalized to [0, 1].

    Parameters
    ----------
    raw_dir : Path, optional
        Path to nnUNet_raw directory.
    image_size : int, default 512
        Resize image and mask to (image_size, image_size) pixels.
    augmentation : bool, default True
        Apply Albumentations transforms during training.
    split : {"train", "val"}, default "train"
        Whether to return training or validation slices.
    val_ratio : float, default 0.2
        Fraction of volumes reserved for validation.
    """

    NUM_CLASSES = len(ICH_LABELS)  # 6 (0-background + 5 ICH types)

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        image_size: int = 512,
        augmentation: bool = True,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir or NNUNET_RAW_DIR)
        self.image_size = image_size
        self.augmentation = augmentation
        self.split = split
        self.val_ratio = val_ratio

        # Locate NIfTI files
        dataset_folder = _find_dataset_folder(self.raw_dir)
        if dataset_folder is None:
            raise FileNotFoundError(f"No Dataset* folder in {self.raw_dir}")

        self.images_dir = dataset_folder / "imagesTr"
        self.labels_dir = dataset_folder / "labelsTr"

        self.dataset_json = dataset_folder / "dataset.json"
        if not self.dataset_json.exists():
            raise FileNotFoundError(f"dataset.json not found in {dataset_folder}")

        # Collect all volume paths
        self.image_paths = sorted(self.images_dir.glob("*_0000.nii.gz"))
        if not self.image_paths:
            raise FileNotFoundError(f"No NIfTI images in {self.images_dir}")

        # Train / val split at volume level
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.image_paths))
        val_count = max(1, int(len(self.image_paths) * val_ratio))

        if split == "val":
            volume_indices = indices[:val_count]
        else:
            volume_indices = indices[val_count:]

        # Build slice index: (volume_idx, slice_idx)
        self._slices: list[tuple[int, int]] = []
        for vol_idx in sorted(volume_indices):
            try:
                img = nib.load(str(self.image_paths[vol_idx]))
                num_slices = img.shape[2]  # axial dim
                for s in range(num_slices):
                    self._slices.append((vol_idx, s))
            except Exception as e:
                logger.warning("Skipping %s: %s", self.image_paths[vol_idx].name, e)

        # Augmentation pipeline
        self.transform = None
        if augmentation and split == "train":
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05,
                    rotate_limit=10, p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3,
                ),
                A.GaussNoise(var_limit=(0, 0.01), p=0.2),
            ])

        logger.info(
            "ICHEmbeddingDataset [%s]: %d volumes → %d slices (img_size=%d)",
            split, len(set(s[0] for s in self._slices)),
            len(self._slices), image_size,
        )

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, idx: int) -> dict:
        vol_idx, slice_idx = self._slices[idx]

        # Load image slice
        img_path = self.image_paths[vol_idx]
        label_path = self.labels_dir / img_path.name.replace("_0000.nii.gz", ".nii.gz")

        img_nii = nib.load(str(img_path))
        image = img_nii.get_fdata()[:, :, slice_idx].astype(np.float32)

        # Load mask slice (clamp to valid labels)
        if label_path.exists():
            mask_nii = nib.load(str(label_path))
            mask = mask_nii.get_fdata()[:, :, slice_idx].astype(np.int64)
            mask = np.clip(mask, 0, self.NUM_CLASSES - 1)
        else:
            mask = np.zeros(image.shape, dtype=np.int64)

        # Normalize image to [0, 1] using 0.5 / 99.5 percentiles
        p_low, p_high = np.percentile(image[image > -900], [0.5, 99.5])
        image = np.clip((image - p_low) / max(p_high - p_low, 1e-6), 0.0, 1.0)

        # Resize to target size
        from PIL import Image as PILImage
        image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        mask_pil = PILImage.fromarray(mask.astype(np.uint8))

        image_resized = np.array(
            image_pil.resize((self.image_size, self.image_size), PILImage.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        mask_resized = np.array(
            mask_pil.resize((self.image_size, self.image_size), PILImage.NEAREST),
            dtype=np.int64,
        )

        # Augmentation
        if self.transform is not None:
            augmented = self.transform(image=image_resized, mask=mask_resized)
            image_resized = augmented["image"]
            mask_resized = augmented["mask"]

        # To tensor: (1, H, W) and (H, W)
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_resized).long()

        # Skip empty slices if they happen (but return them for consistency)
        return {"image": image_tensor, "mask": mask_tensor, "index": idx}
