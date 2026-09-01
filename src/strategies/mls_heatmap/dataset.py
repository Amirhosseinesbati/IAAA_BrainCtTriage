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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import MLS_DIR, TRAINING_CSV_PATH
from src.evaluation.splits import normalize_study_id, split_study_ids
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
        include_negatives: bool = False,
        return_selector: bool = False,
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
        self.include_negatives = include_negatives
        self.return_selector = return_selector

        # The historical heatmap path uses positives only. Multitask training
        # explicitly keeps negatives so selector confidence is supervised.
        df = pd.read_csv(csv_path)
        if not include_negatives:
            df = df[df["is_target"] == 1]
        df = df.reset_index(drop=True)
        df["patient_id"] = df["patient_id"].map(normalize_study_id)
        df = self._attach_spacing(df)
        self.data = df

        if len(self.data) == 0:
            logger.warning(f"No positive samples found in {csv_path}")

        logger.info(
            f"MLSHeatmapDataset: {len(self.data)} samples, "
            f"img_size={img_size}, heatmap_size={heatmap_size}, "
            f"sigma={heatmap_sigma}, augment={augment}"
        )

    @staticmethod
    def _attach_spacing(df: pd.DataFrame) -> pd.DataFrame:
        """Attach DICOM x spacing and the official study-level maximum MLS.

        Newly built MLS CSVs contain ``spacing_x`` directly. Older datasets
        are upgraded in memory from the competition metadata, avoiding the
        previous hard-coded 0.5 mm/px validation approximation.
        """
        result = df.copy()
        if "spacing_x" not in result:
            result["spacing_x"] = np.nan
        if "study_mls_mm" not in result:
            result["study_mls_mm"] = np.nan
        numeric_spacing = pd.to_numeric(result["spacing_x"], errors="coerce")
        needs_spacing = ~np.isfinite(numeric_spacing) | (numeric_spacing <= 0)
        numeric_study_mls = pd.to_numeric(result["study_mls_mm"], errors="coerce")
        needs_study_mls = ~np.isfinite(numeric_study_mls) | (numeric_study_mls < 0)
        if needs_spacing.any() or needs_study_mls.any():
            columns = [
                "dicom_series.id", "dicom_series.PixelSpacing1", "MidlineShiftMM",
            ]
            if not TRAINING_CSV_PATH.is_file():
                raise ValueError(
                    "MLS labels require spacing/study truth and training metadata "
                    f"is unavailable at {TRAINING_CSV_PATH}. Rebuild the MLS dataset."
                )
            metadata = pd.read_csv(TRAINING_CSV_PATH, usecols=columns)
            metadata["dicom_series.id"] = metadata["dicom_series.id"].map(normalize_study_id)
            grouped = metadata.groupby("dicom_series.id")
            if needs_spacing.any():
                spacing_map = grouped["dicom_series.PixelSpacing1"].median()
                result.loc[needs_spacing, "spacing_x"] = result.loc[
                    needs_spacing, "patient_id"
                ].map(spacing_map)
            if needs_study_mls.any():
                study_mls_map = grouped["MidlineShiftMM"].max()
                result.loc[needs_study_mls, "study_mls_mm"] = result.loc[
                    needs_study_mls, "patient_id"
                ].map(study_mls_map)
        result["spacing_x"] = pd.to_numeric(result["spacing_x"], errors="coerce")
        invalid = ~np.isfinite(result["spacing_x"]) | (result["spacing_x"] <= 0)
        if invalid.any():
            studies = result.loc[invalid, "patient_id"].drop_duplicates().tolist()
            raise ValueError(f"Invalid or missing MLS pixel spacing for studies: {studies[:10]}")
        result["study_mls_mm"] = pd.to_numeric(result["study_mls_mm"], errors="coerce")
        invalid_study_mls = ~np.isfinite(result["study_mls_mm"]) | (result["study_mls_mm"] < 0)
        if invalid_study_mls.any():
            studies = result.loc[invalid_study_mls, "patient_id"].drop_duplicates().tolist()
            raise ValueError(f"Invalid or missing study MLS truth for studies: {studies[:10]}")
        return result

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

    def __getitem__(self, idx: int):
        """
        Get a training sample.

        Returns:
            image: Tensor (C, H, W) normalized to [0, 1].
            heatmap_target: Tensor (K, heatmap_size, heatmap_size) of Gaussians.
            mask: Tensor (K,) where mask[i] = 1 if keypoint i is present.
            keypoints: Tensor (K, 2) of (x, y) coordinates in **image pixels**
                after augmentation. Used to compute keypoint MAE and the true
                MLS value during validation (instead of re-decoding heatmaps).
            spacing_x: Scalar tensor with the study's DICOM column spacing.
        """
        row = self.data.iloc[idx]
        is_target = float(row["is_target"])

        # Load image
        raw_image_path = row.get("image_path", "")
        image_path = "" if pd.isna(raw_image_path) else str(raw_image_path).strip()
        img_path = image_path if image_path else str(self.img_dir / row["image_name"])
        image = self._load_image(img_path)

        # Get keypoints
        keypoints = self._get_keypoints(row)

        # Apply augmentation (consistent for image + keypoints)
        if self.augment:
            image, keypoints = self._apply_augmentation(image, keypoints)

        # Convert to tensor: (C, H, W) from (H, W, C)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Generate Gaussian heatmaps from (possibly augmented) keypoints
        kp_list = (
            [(float(kp[0]), float(kp[1])) for kp in keypoints]
            if is_target > 0.5 else [None, None, None]
        )
        heatmap_target, mask = generate_gaussian_heatmap(
            kp_list,
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        # True keypoints in image pixel space (after augmentation)
        keypoints_tensor = torch.from_numpy(keypoints.copy()).float()  # (K, 2)

        spacing_tensor = torch.tensor(float(row["spacing_x"]), dtype=torch.float32)
        base = (image_tensor, heatmap_target, mask, keypoints_tensor, spacing_tensor)
        if self.return_selector:
            return (
                *base,
                torch.tensor(is_target, dtype=torch.float32),
                torch.tensor(float(row["study_mls_mm"]), dtype=torch.float32),
                str(row["patient_id"]),
            )
        return base


# ═════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═════════════════════════════════════════════════════════════════════════

def build_mls_sampling_weights(data: pd.DataFrame, mode: str) -> torch.Tensor:
    """Return row weights for a reproducible MLS sampler policy.

    All modes allocate equal total mass to target and nontarget rows. Legacy
    slice balancing makes study mass proportional to its row count. Full study
    balancing makes study mass uniform. The hybrid policy uses the geometric
    midpoint: study mass is proportional to sqrt(row count), retaining useful
    exposure from long studies without allowing linear slice-count dominance.
    """
    required = {"patient_id", "is_target"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Sampling data is missing columns: {sorted(missing)}")
    labels = data["is_target"].to_numpy(dtype=int)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("MLS sampler expects binary is_target labels")
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Balanced selector sampling needs both classes, got {counts.tolist()}")

    if mode == "slice_class_balanced":
        weights = np.where(labels == 1, 1.0 / counts[1], 1.0 / counts[0])
    elif mode == "hybrid_study_class_balanced":
        working = pd.DataFrame({
            "patient_id": data["patient_id"].astype(str).to_numpy(),
            "is_target": labels,
        })
        rows_per_study_class = working.groupby(
            ["is_target", "patient_id"], sort=False
        )["patient_id"].transform("size").to_numpy(dtype=float)
        unnormalized = 1.0 / np.sqrt(rows_per_study_class)
        normalizer = pd.Series(unnormalized).groupby(working["is_target"]).transform(
            "sum"
        ).to_numpy(dtype=float)
        weights = unnormalized / normalizer
    elif mode == "study_class_balanced":
        working = pd.DataFrame({
            "patient_id": data["patient_id"].astype(str).to_numpy(),
            "is_target": labels,
        })
        rows_per_study_class = working.groupby(
            ["is_target", "patient_id"], sort=False
        )["patient_id"].transform("size").to_numpy(dtype=float)
        studies_per_class = working.groupby("is_target")["patient_id"].nunique()
        class_study_counts = working["is_target"].map(studies_per_class).to_numpy(dtype=float)
        weights = 1.0 / (class_study_counts * rows_per_study_class)
    else:
        raise ValueError(f"Unknown MLS sampling mode: {mode}")

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("MLS sampler produced invalid row weights")
    return torch.as_tensor(weights, dtype=torch.double)

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
    fold: int = 0,
    use_competition_folds: bool = True,
    include_negatives: bool = False,
    return_selector: bool = False,
    balanced_sampling: bool = False,
    sampling_mode: str = "slice_class_balanced",
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

    Notes:
        The split is performed at the **patient level** (all slices of a
        patient go to either train or val) to avoid data leakage between
        correlated slices of the same study.

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
        include_negatives=include_negatives,
        return_selector=return_selector,
    )

    # Study-level split from the patient-grouped immutable competition folds.
    df = full_dataset.data
    studies = df["patient_id"].unique()
    if use_competition_folds:
        train_patients, val_patients = split_study_ids(studies, fold)
    else:
        rng = np.random.default_rng(seed)
        shuffled_patients = rng.permutation(studies)
        n_val_patients = max(1, int(round(len(shuffled_patients) * val_split)))
        val_patients = set(shuffled_patients[:n_val_patients].tolist())
        train_patients = set(shuffled_patients[n_val_patients:].tolist())

    pos = np.arange(len(df))
    train_indices = pos[df["patient_id"].isin(train_patients)]
    val_indices = pos[df["patient_id"].isin(val_patients)]

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
        include_negatives=include_negatives,
        return_selector=return_selector,
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
        include_negatives=include_negatives,
        return_selector=return_selector,
    )
    val_dataset.data = full_dataset.data.iloc[val_indices].reset_index(drop=True)

    sampler = None
    if balanced_sampling:
        weights = build_mls_sampling_weights(train_dataset.data, sampling_mode)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights), replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
        f"batch_size={batch_size}, sampling_mode={sampling_mode if sampler else 'shuffle'}"
    )

    return train_loader, val_loader
