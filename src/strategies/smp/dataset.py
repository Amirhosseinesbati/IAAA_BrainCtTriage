"""
smp/dataset.py — PyTorch Dataset for SMP 2D ICH segmentation.

Reads NIfTI volumes from the generic NIfTI directory (*ICH_NIFTI_DIR*)
and yields 2D slices with corresponding label masks.  The data is
expected to have been produced by :class:`NiftiDatasetBuilder`, not
by the nnUNet builder — this avoids any coupling to nnUNet naming.
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

from src.config import ICH_LABELS, ICH_NIFTI_DIR
from src.evaluation.splits import split_items_by_fold, study_id_from_path
from src.strategies.augmentation_config import SMPAugmentationConfig

logger = logging.getLogger(__name__)


def _find_dataset_folder(data_dir: Path) -> Optional[Path]:
    """Locate the dataset folder inside *ICH_NIFTI_DIR*.

    Looks for subdirectories that contain both ``images/`` and
    ``labels/`` (the generic NIfTI layout).
    """
    for candidate in sorted(data_dir.iterdir()):
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    logger.error("No dataset folder found in %s", data_dir)
    return None


def build_smp_augmentation(aug_config: SMPAugmentationConfig) -> Optional[A.Compose]:
    """
    Build an Albumentations ``Compose`` pipeline from an ``SMPAugmentationConfig``.

    Returns ``None`` when ``aug_config.enabled`` is ``False``.
    """
    if not aug_config.enabled:
        return None

    transforms: list[A.BasicTransform] = []

    if aug_config.top_bottom_flip.enabled:
        transforms.append(A.VerticalFlip(p=aug_config.top_bottom_flip.prob))

    if aug_config.left_right_flip.enabled:
        transforms.append(A.HorizontalFlip(p=aug_config.left_right_flip.prob))

    if aug_config.rotate90.enabled:
        transforms.append(A.RandomRotate90(p=aug_config.rotate90.prob))

    ssr = aug_config.shift_scale_rotate
    if ssr.enabled:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=ssr.shift_limit,
            scale_limit=ssr.scale_limit,
            rotate_limit=ssr.rotate_limit,
            p=ssr.prob,
        ))

    if aug_config.brightness_contrast.enabled:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=aug_config.brightness_contrast.prob,
        ))

    if aug_config.gauss_noise.enabled:
        transforms.append(A.GaussNoise(
            var_limit=aug_config.gauss_noise.var_limit,
            p=aug_config.gauss_noise.prob,
        ))

    if aug_config.scale_intensity.enabled:
        # ScaleIntensity: random scaling of pixel values by (1 ± factor)
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=aug_config.scale_intensity.prob * 0.2,
            contrast_limit=0.0,
            p=aug_config.scale_intensity.prob,
        ))

    if aug_config.adjust_contrast.enabled:
        transforms.append(A.RandomGamma(
            gamma_limit=(80, 120), p=aug_config.adjust_contrast.prob,
        ))

    if not transforms:
        return None

    return A.Compose(transforms)


class ICHEmbeddingDataset(Dataset):
    """
    2D slice-level dataset for ICH segmentation.

    Iterates over all NIfTI volumes in `imagesTr/`, extracts 2D axial
    slices, and returns (image, mask) pairs normalized to [0, 1].

    Parameters
    ----------
    data_dir : Path, optional
        Path to *ICH_NIFTI_DIR* (generic NIfTI directory).
    image_size : int, default 512
        Resize image and mask to (image_size, image_size) pixels.
    augmentation : bool, default True
        Apply Albumentations transforms during training.
    split : {"train", "val"}, default "train"
        Whether to return training or validation slices.
    val_ratio : float, default 0.2
        Fraction of volumes reserved for validation.
    model_dimension : str, default "2D"
        ``"2D"`` for single-slice input, ``"2.5D"`` for stacked slices.
    slices_per_stack : int | None, default None
        Number of consecutive slices stacked as input channels.
        Only used when ``model_dimension="2.5D"``.
    """

    NUM_CLASSES = len(ICH_LABELS)  # 6 (0-background + 5 ICH types)

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        image_size: int = 512,
        augmentation: bool = True,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        fold: int = 0,
        use_competition_folds: bool = True,
        model_dimension: str = "2D",
        slices_per_stack: Optional[int] = None,
        augmentation_config: Optional[SMPAugmentationConfig] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir or ICH_NIFTI_DIR)
        self.image_size = image_size
        self.augmentation = augmentation
        self.split = split
        self.val_ratio = val_ratio
        self.model_dimension = model_dimension
        self.slices_per_stack = slices_per_stack if model_dimension == "2.5D" else 1
        self.augmentation_config = augmentation_config

        # Locate NIfTI files
        dataset_folder = _find_dataset_folder(self.data_dir)
        if dataset_folder is None:
            raise FileNotFoundError(
                f"No dataset folder found in {self.data_dir}. "
                "Run NiftiDatasetBuilder().build() first."
            )

        self.images_dir = dataset_folder / "images"
        self.labels_dir = dataset_folder / "labels"

        # Collect all volume paths
        self.image_paths = sorted(self.images_dir.glob("*.nii.gz"))
        if not self.image_paths:
            raise FileNotFoundError(f"No NIfTI images in {self.images_dir}")

        # Resolve the same study-level validation fold used by every strategy.
        if use_competition_folds:
            training, validation = split_items_by_fold(
                self.image_paths, fold, id_getter=study_id_from_path,
            )
            selected_paths = validation if split == "val" else training
            selected = set(selected_paths)
            volume_indices = [
                index for index, path in enumerate(self.image_paths) if path in selected
            ]
        else:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.image_paths))
            val_count = max(1, int(len(self.image_paths) * val_ratio))
            volume_indices = indices[:val_count] if split == "val" else indices[val_count:]

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

        # Augmentation pipeline (dynamic from config, or fallback to hardcoded)
        self.transform = None
        if split == "train" and self.augmentation_config is not None:
            self.transform = build_smp_augmentation(self.augmentation_config)
        elif augmentation and split == "train":
            # Legacy fallback: build from hardcoded defaults
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

    def _load_2d_slice(self, img_nii, mask_nii, slice_idx: int):
        """Load, normalize, and resize a single 2D slice."""
        image = img_nii.get_fdata()[:, :, slice_idx].astype(np.float32)

        if mask_nii is not None:
            mask = mask_nii.get_fdata()[:, :, slice_idx].astype(np.int64)
            mask = np.clip(mask, 0, self.NUM_CLASSES - 1)
        else:
            mask = np.zeros(image.shape, dtype=np.int64)

        # Normalize
        p_low, p_high = np.percentile(image[image > -900], [0.5, 99.5])
        image = np.clip((image - p_low) / max(p_high - p_low, 1e-6), 0.0, 1.0)

        # Resize
        from PIL import Image as PILImage
        image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        image_resized = np.array(
            image_pil.resize((self.image_size, self.image_size), PILImage.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        return image_resized, mask

    def __getitem__(self, idx: int) -> dict:
        vol_idx, slice_idx = self._slices[idx]

        # Load NIfTI volumes
        img_path = self.image_paths[vol_idx]
        label_path = self.labels_dir / img_path.name

        img_nii = nib.load(str(img_path))
        num_slices = img_nii.shape[2]

        mask_nii = nib.load(str(label_path)) if label_path.exists() else None

        if self.model_dimension == "2.5D" and self.slices_per_stack > 1:
            # ── 2.5D: stack consecutive slices as input channels ─────
            half = self.slices_per_stack // 2
            stack_images = []

            for offset in range(-half, half + 1):
                src_idx = slice_idx + offset
                # Replicate border slices
                src_idx = max(0, min(src_idx, num_slices - 1))
                img_slice, _ = self._load_2d_slice(img_nii, None, src_idx)
                stack_images.append(img_slice)

            # Stack along first axis → (slices_per_stack, H, W)
            stacked = np.stack(stack_images, axis=0)

            # Load mask for the center (target) slice only
            _, mask_orig = self._load_2d_slice(img_nii, mask_nii, slice_idx)

            # Resize mask
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask_orig.astype(np.uint8))
            mask_resized = np.array(
                mask_pil.resize((self.image_size, self.image_size), PILImage.NEAREST),
                dtype=np.int64,
            )

            image_array = stacked.astype(np.float32)
        else:
            # ── 2D: single slice ─────────────────────────────────────
            image_resized, mask_orig = self._load_2d_slice(img_nii, mask_nii, slice_idx)

            # Resize mask
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask_orig.astype(np.uint8))
            mask_resized = np.array(
                mask_pil.resize((self.image_size, self.image_size), PILImage.NEAREST),
                dtype=np.int64,
            )

            image_array = image_resized  # (H, W)

        # ── Augmentation (applied per-slice on 2D; 2.5D stacks skip) ──
        # For 2.5D, augmentation is applied per-slice *before* stacking
        # in a future enhancement.  For now, 2.5D stacks skip Albumentations
        # (the individual slices have already been normalized).
        if self.transform is not None and self.model_dimension == "2D":
            augmented = self.transform(image=image_array, mask=mask_resized)
            image_array = augmented["image"]
            mask_resized = augmented["mask"]

        # To tensor
        if self.model_dimension == "2.5D":
            image_tensor = torch.from_numpy(image_array).float()  # (C, H, W)
        else:
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()  # (1, H, W)

        mask_tensor = torch.from_numpy(mask_resized).long()

        return {"image": image_tensor, "mask": mask_tensor, "index": idx}
