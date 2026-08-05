"""
dataset.py — MLS Heatmap Dataset with on-the-fly heatmap generation and augmentation.

Loads pre-built 3-channel PNG images + keypoint CSV from the existing
MlsDatasetBuilder output, generates Gaussian heatmaps on-the-fly, and
applies geometric + intensity augmentations that consistently transform
both the image and its keypoint coordinates.

Augmentations (applied with configurable probability):
- Rotation (±deg) around image center
- Translation (horizontal/vertical)
- Intensity jitter (brightness + contrast)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import MLS_DIR
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Augmentation Transforms
# ═════════════════════════════════════════════════════════════════════════

def rotate_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    angle_deg: float,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image and keypoints around the center.

    Args:
        image: (H, W, C) numpy array.
        keypoints: (3, 2) numpy array of (x, y) pixel coordinates.
        angle_deg: Rotation angle in degrees (positive = counter-clockwise).
        img_size: Image size in pixels (square).

    Returns:
        Rotated image and transformed keypoints.
    """
    H, W = image.shape[:2]
    center = (W / 2, H / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Transform keypoints
    ones = np.ones((keypoints.shape[0], 1))
    kp_homo = np.concatenate([keypoints, ones], axis=1)  # (K, 3)
    new_kps = (rot_mat @ kp_homo.T).T  # (K, 2)

    return rotated, new_kps


def translate_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    tx: float,
    ty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate image and keypoints.

    Args:
        image: (H, W, C) numpy array.
        keypoints: (3, 2) numpy array of (x, y) pixel coordinates.
        tx, ty: Translation in pixels (x and y).

    Returns:
        Translated image and transformed keypoints.
    """
    H, W = image.shape[:2]
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, trans_mat, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    new_kps = keypoints.copy()
    new_kps[:, 0] += tx
    new_kps[:, 1] += ty

    return translated, new_kps


def intensity_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """
    Apply brightness and contrast jitter to image.

    Args:
        image: (H, W, C) numpy array in [0, 1] range.
        brightness: Brightness offset (added to all pixels).
        contrast: Contrast multiplier.

    Returns:
        Augmented image.
    """
    result = image * contrast + brightness
    return np.clip(result, 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════

class MLSHeatmapDataset(Dataset):
    """
    Dataset for heatmap-based MLS keypoint regression.

    Reads pre-built 3-channel PNG images and keypoint coordinates from
    the existing MlsDatasetBuilder output. Generates Gaussian heatmaps
    on-the-fly and applies optional augmentation.

    Args:
        csv_path: Path to the MLS labels CSV file.
        img_dir: Path to the directory containing PNG images.
        img_size: Input image size (square).
        heatmap_size: Output heatmap size (square). Typically img_size // 4.
        heatmap_sigma: Standard deviation of Gaussian heatmap (heatmap pixels).
        augment: Whether to apply data augmentation.
        rotation_deg: Max rotation ±degrees.
        translation: Max translation as fraction of img_size.
        intensity_jitter_scale: Max intensity jitter (brightness/contrast).
        augment_prob: Probability of applying augmentation.
    """

    KEYPOINT_COLS = ["x1", "y1", "x2", "y2", "x3", "y3"]

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        img_size: int = 512,
        heatmap_size: int = 128,
        heatmap_sigma: float = 2.0,
        augment: bool = False,
        rotation_deg: float = 10.0,
        translation: float = 0.05,
        intensity_jitter_scale: float = 0.05,
        augment_prob: float = 0.5,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.augment = augment
        self.augment_prob = augment_prob
        self.rotation_deg = rotation_deg
        self.translation = translation
        self.intensity_jitter_scale = intensity_jitter_scale

        # Load CSV and filter to positive samples (all 3 keypoints present)
        df = pd.read_csv(csv_path)
        df = df[df["is_target"] == 1].reset_index(drop=True)
        self.data = df

        if len(self.data) == 0:
            logger.warning(f"No positive samples found in {csv_path}")

        logger.info(
            f"MLSHeatmapDataset: {len(self.data)} samples, "
            f"img_size={img_size}, heatmap_size={heatmap_size}, "
            f"sigma={heatmap_sigma}, augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def _get_keypoints(self, row: pd.Series) -> np.ndarray:
        """Extract (3, 2) keypoint array from a CSV row."""
        kps = np.array([
            [row["x1"], row["y1"]],
            [row["x2"], row["y2"]],
            [row["x3"], row["y3"]],
        ], dtype=np.float32)
        return kps

    def _load_image(self, img_path: str) -> np.ndarray:
        """Load a 3-channel PNG and normalize to [0, 1]."""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        return img_rgb

    def _apply_augmentation(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation with configured probability."""
        if not self.augment or np.random.random() > self.augment_prob:
            return image, keypoints

        # Rotation
        if self.rotation_deg > 0:
            angle = np.random.uniform(-self.rotation_deg, self.rotation_deg)
            image, keypoints = rotate_image_and_keypoints(
                image, keypoints, angle, self.img_size
            )

        # Translation
        if self.translation > 0:
            max_tx = self.translation * self.img_size
            max_ty = self.translation * self.img_size
            tx = np.random.uniform(-max_tx, max_tx)
            ty = np.random.uniform(-max_ty, max_ty)
            image, keypoints = translate_image_and_keypoints(
                image, keypoints, tx, ty
            )

        # Intensity jitter
        if self.intensity_jitter_scale > 0:
            brightness = np.random.uniform(-self.intensity_jitter_scale, self.intensity_jitter_scale)
            contrast = np.random.uniform(1.0 - self.intensity_jitter_scale, 1.0 + self.intensity_jitter_scale)
            image = intensity_jitter(image, brightness, contrast)

        return image, keypoints

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            image: Tensor (C, H, W) normalized to [0, 1].
            heatmap_target: Tensor (K, heatmap_size, heatmap_size) of Gaussians.
            mask: Tensor (K,) where mask[i] = 1 if keypoint i is present.
            keypoints: Tensor (K, 2) of (x, y) coordinates in **image pixels**
                after augmentation. Used to compute keypoint MAE and the true
                MLS value during validation (instead of re-decoding heatmaps).
        """
        row = self.data.iloc[idx]

        # Load image
        img_name = row["image_name"]
        img_path = str(self.img_dir / img_name)
        image = self._load_image(img_path)

        # Get keypoints
        keypoints = self._get_keypoints(row)

        # Apply augmentation (consistent for image + keypoints)
        if self.augment:
            image, keypoints = self._apply_augmentation(image, keypoints)

        # Convert to tensor: (C, H, W) from (H, W, C)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Generate Gaussian heatmaps from (possibly augmented) keypoints
        kp_list = [(float(kp[0]), float(kp[1])) for kp in keypoints]
        heatmap_target, mask = generate_gaussian_heatmap(
            kp_list,
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        # True keypoints in image pixel space (after augmentation)
        keypoints_tensor = torch.from_numpy(keypoints.copy()).float()  # (K, 2)

        return image_tensor, heatmap_target, mask, keypoints_tensor


# ═════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═════════════════════════════════════════════════════════════════════════

def create_mls_dataloaders(
    csv_path: str,
    img_dir: str,
    img_size: int = 512,
    heatmap_size: int = 128,
    heatmap_sigma: float = 2.0,
    batch_size: int = 8,
    val_split: float = 0.2,
    augment: bool = True,
    rotation_deg: float = 10.0,
    translation: float = 0.05,
    intensity_jitter_scale: float = 0.05,
    augment_prob: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders for MLS heatmap training.

    Args:
        csv_path: Path to MLS labels CSV.
        img_dir: Path to image directory.
        img_size: Input image size.
        heatmap_size: Output heatmap size.
        heatmap_sigma: Gaussian sigma.
        batch_size: Batch size.
        val_split: Fraction of data for validation.
        augment: Enable augmentation for training set.
        ...augmentation params...
        num_workers: DataLoader workers.
        seed: Random seed for split.

    Returns:
        (train_loader, val_loader)
    """
    # Full dataset (no augmentation for splitting)
    full_dataset = MLSHeatmapDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=img_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        augment=False,
    )

    # Split indices
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    # Create subsets with augmentation only for training
    train_dataset = MLSHeatmapDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=img_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        augment=augment,
        rotation_deg=rotation_deg,
        translation=translation,
        intensity_jitter_scale=intensity_jitter_scale,
        augment_prob=augment_prob,
    )
    # Override the internal data to use only train indices
    train_dataset.data = full_dataset.data.iloc[train_indices].reset_index(drop=True)

    val_dataset = MLSHeatmapDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=img_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        augment=False,
    )
    val_dataset.data = full_dataset.data.iloc[val_indices].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"batch_size={batch_size}"
    )

    return train_loader, val_loader
