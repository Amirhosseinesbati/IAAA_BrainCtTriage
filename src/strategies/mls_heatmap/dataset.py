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

import logging
import os
import random
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import MLS_DIR, PROJECT_ROOT, TRAINING_CSV_PATH, TRAINING_PKL_PATH
from src.evaluation.splits import normalize_study_id, split_study_ids
from src.strategies.mls_heatmap.context_cache import load_mls_2p5d_cache_manifest
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap

logger = logging.getLogger(__name__)


def seed_mls_loader_worker(_worker_id: int) -> None:
    """Seed Python/NumPy from PyTorch's per-worker seed.

    PyTorch derives ``initial_seed`` from the main-process RNG when a loader
    iterator is created.  The trainer reseeds that RNG at each epoch in the
    opt-in reproducible modes, making augmentation streams stable across
    hosts, worker PIDs, and exact-resume boundaries.
    """
    worker_seed = int(torch.initial_seed() % 2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def scheduled_heatmap_sigma(
    reference_sigma: float,
    anneal_end: float | None,
    epoch: int,
    total_epochs: int,
) -> float:
    """Return a symmetric coarse-to-fine training-target width.

    ``reference_sigma`` stays the fixed validation target and is also the
    schedule midpoint.  This keeps the mean target width equal to the frozen
    baseline while testing whether broad early supervision followed by a
    sharper target improves localization.  Epochs are one-based so exact
    resume recovers the same target without serializing mutable dataset state.
    """
    reference = float(reference_sigma)
    if anneal_end is None:
        return reference
    end = float(anneal_end)
    if end > reference:
        raise ValueError("anneal_end must not exceed reference_sigma")
    start = 2.0 * reference - end
    if start > 8.0 or end < 0.5:
        raise ValueError("scheduled heatmap sigma must remain within [0.5, 8.0]")
    if total_epochs <= 1:
        return end
    clipped_epoch = min(max(int(epoch), 1), int(total_epochs))
    progress = (clipped_epoch - 1) / (int(total_epochs) - 1)
    return start + (end - start) * progress


def resolve_mls_image_path(
    raw_path: object,
    image_name: object,
    image_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve portable MLS image paths from historical absolute CSV values."""
    raw = "" if pd.isna(raw_path) else str(raw_path).strip()
    name = str(image_name).strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
        normalized = raw.replace("\\", "/")
        marker = "/data/processed/"
        location = normalized.lower().find(marker)
        if location >= 0:
            relative = normalized[location + len(marker):]
            candidates.append(project_root / "Data" / "processed" / Path(relative))
    candidates.append(image_dir / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot resolve MLS image {name!r}; attempted: {attempted}")


# ═════════════════════════════════════════════════════════════════════════
# Augmentation Transforms
# ═════════════════════════════════════════════════════════════════════════

def rotate_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    angle_deg: float,
    img_size: int,
    *,
    force_per_channel: bool = False,
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
    rotated = _warp_all_channels(
        image, rot_mat, W, H, force_per_channel=force_per_channel,
    )

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
    *,
    force_per_channel: bool = False,
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
    translated = _warp_all_channels(
        image, trans_mat, W, H, force_per_channel=force_per_channel,
    )

    new_kps = keypoints.copy()
    new_kps[:, 0] += tx
    new_kps[:, 1] += ty

    return translated, new_kps


def _warp_all_channels(
    image: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
    *,
    force_per_channel: bool = False,
) -> np.ndarray:
    """Apply one affine to all channels without trusting OpenCV's >4-channel path.

    The legacy 1/3-channel route is kept byte-for-byte on OpenCV's native
    multi-channel implementation.  A 2.5D sample has nine channels; warping
    each plane with the same matrix prevents neighbour-specific transforms.
    """
    if (
        not force_per_channel
        and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] <= 4))
    ):
        return cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    if image.ndim != 3:
        raise ValueError(f"Expected [H, W] or [H, W, C] image, got {image.shape}")
    return np.stack(
        [
            cv2.warpAffine(
                image[:, :, channel], matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            for channel in range(image.shape[2])
        ],
        axis=-1,
    )


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
        input_channels: int = 3,
        context_cache_root: str | Path | None = None,
        context_cache_manifest_sha256: str | None = None,
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
        self.input_channels = int(input_channels)
        self.context_cache_root = (
            Path(context_cache_root).resolve() if context_cache_root is not None else None
        )
        self.context_cache_manifest_sha256 = context_cache_manifest_sha256
        self._cache_manifest_verified = False
        self._cache_manifest_verified_pid: int | None = None
        self._study_cache_dir: Path | None = None
        self._cached_studies: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_open_studies = 8
        if self.context_cache_root is not None:
            manifest, _ = load_mls_2p5d_cache_manifest(
                self.context_cache_root,
                expected_sha256=self.context_cache_manifest_sha256,
            )
            if int(manifest["image_size"]) != int(img_size):
                raise ValueError(
                    "MLS 2.5D cache image_size does not match requested dataset image_size: "
                    f"{manifest['image_size']} != {img_size}"
                )
            if self.input_channels not in {3, 9}:
                raise ValueError("MLS 2.5D cache supports only 3 or 9 input channels")
            self._study_cache_dir = self.context_cache_root / str(manifest["study_cache_dir"])
            self._cache_manifest_verified = True
            self._cache_manifest_verified_pid = os.getpid()
        elif self.input_channels not in {1, 3}:
            raise ValueError(
                "Nine-channel MLS training requires the immutable 2.5D cache; "
                "a PNG-only fallback is forbidden"
            )

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
            if TRAINING_CSV_PATH.is_file():
                metadata = pd.read_csv(TRAINING_CSV_PATH, usecols=columns)
            elif TRAINING_PKL_PATH.is_file():
                metadata = pd.read_pickle(TRAINING_PKL_PATH).loc[:, columns].copy()
            else:
                raise ValueError(
                    "MLS labels require spacing/study truth and training metadata "
                    f"is unavailable at {TRAINING_CSV_PATH} or {TRAINING_PKL_PATH}. "
                    "Restore DVC raw metadata or rebuild the MLS dataset."
                )
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

    def _load_cached_study(self, study_id: str) -> np.ndarray:
        """Open one immutable per-study float32 cache through a bounded LRU."""
        if self._study_cache_dir is None:
            raise RuntimeError("No MLS 2.5D study cache configured")
        if (
            not self._cache_manifest_verified
            or self._cache_manifest_verified_pid != os.getpid()
        ):
            # Worker processes may deserialize a Dataset after the parent
            # validated it. Recheck the pinned manifest at first use so an
            # atomic replacement cannot silently change their image contract.
            load_mls_2p5d_cache_manifest(
                self.context_cache_root,
                expected_sha256=self.context_cache_manifest_sha256,
            )
            self._cache_manifest_verified = True
            self._cache_manifest_verified_pid = os.getpid()
        key = str(study_id)
        current = self._cached_studies.pop(key, None)
        if current is not None:
            self._cached_studies[key] = current
            return current
        path = self._study_cache_dir / f"{key}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"MLS 2.5D cache study is missing: {path}")
        current = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (3, self.img_size, self.img_size)
        if current.ndim != 4 or tuple(current.shape[1:]) != expected:
            raise ValueError(
                f"Invalid MLS 2.5D study cache shape {tuple(current.shape)} for {study_id}; "
                f"expected [D, {expected[0]}, {expected[1]}, {expected[2]}]"
            )
        if current.dtype != np.float32:
            raise ValueError(f"MLS 2.5D study cache must be float32, got {current.dtype}")
        self._cached_studies[key] = current
        while len(self._cached_studies) > self._max_open_studies:
            self._cached_studies.popitem(last=False)
        return current

    def _load_context_image(self, row: pd.Series) -> np.ndarray:
        """Assemble central or adjacent cache channels in canonical z order."""
        raw_index = pd.to_numeric(row.get("slice_index"), errors="coerce")
        if not np.isfinite(raw_index) or int(raw_index) != float(raw_index):
            raise ValueError("MLS 2.5D cache labels require an integer slice_index")
        cache = self._load_cached_study(str(row["patient_id"]))
        index = int(raw_index)
        if index < 0 or index >= cache.shape[0]:
            raise IndexError(
                f"MLS cache slice_index={index} outside study depth {cache.shape[0]} "
                f"for study {row['patient_id']}"
            )
        positions = [index] if self.input_channels == 3 else [
            max(0, index - 1), index, min(cache.shape[0] - 1, index + 1),
        ]
        channels = np.concatenate([np.asarray(cache[position]) for position in positions], axis=0)
        if channels.shape[0] != self.input_channels:
            raise RuntimeError(
                f"MLS context channel mismatch: assembled {channels.shape[0]}, "
                f"configured {self.input_channels}"
            )
        return np.moveaxis(channels, 0, -1).astype(np.float32, copy=False)

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
                image, keypoints, angle, self.img_size,
                force_per_channel=self.context_cache_root is not None,
            )

        # Translation
        if self.translation > 0:
            max_tx = self.translation * self.img_size
            max_ty = self.translation * self.img_size
            tx = np.random.uniform(-max_tx, max_tx)
            ty = np.random.uniform(-max_ty, max_ty)
            image, keypoints = translate_image_and_keypoints(
                image, keypoints, tx, ty,
                force_per_channel=self.context_cache_root is not None,
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

        # Load either the legacy central PNG or the immutable context cache.
        if self.context_cache_root is None:
            img_path = resolve_mls_image_path(
                row.get("image_path", ""), row["image_name"], self.img_dir,
            )
            image = self._load_image(str(img_path))
        else:
            image = self._load_context_image(row)

        # Get keypoints
        keypoints = self._get_keypoints(row)

        # Apply augmentation (consistent for image + keypoints)
        if self.augment:
            image, keypoints = self._apply_augmentation(image, keypoints)

        # Convert to a writable contiguous CHW buffer before Torch sees it.
        # Context images originate in a read-only np.memmap; a transposed view
        # would otherwise make ``torch.from_numpy`` accept undefined write
        # behavior on validation samples without augmentation.
        image_chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        if not image_chw.flags.writeable:
            image_chw = image_chw.copy()
        image_tensor = torch.from_numpy(image_chw)

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


class MLSPositiveStudyBagDataset(Dataset):
    """Return all annotated target slices of one positive study as a bag.

    This is deliberately an *auxiliary* view of the ordinary training dataset:
    the main slice-class-balanced loader remains authoritative.  Restricting a
    bag to annotated target slices keeps the study target semantically valid
    and bounds a bag by the 5--27 supplied target annotations rather than by
    the full DICOM series length.
    """

    def __init__(self, base_dataset: MLSHeatmapDataset):
        if not base_dataset.return_selector:
            raise ValueError("Study-bag dataset requires return_selector=true")
        positive = base_dataset.data.loc[
            pd.to_numeric(base_dataset.data["is_target"], errors="raise") > 0.5
        ].copy()
        self._base_dataset = base_dataset
        self._bags: list[list[int]] = [
            group.index.to_list()
            for _study_id, group in positive.groupby("patient_id", sort=True)
        ]
        if not self._bags:
            raise ValueError("Study-bag dataset requires at least one positive study")
        if min(len(bag) for bag in self._bags) < 1:
            raise ValueError("Study-bag dataset contains an empty positive bag")

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, index: int) -> list[tuple]:
        return [self._base_dataset[row_index] for row_index in self._bags[index]]


def collate_mls_study_bag(batch: list[list[tuple]]) -> tuple:
    """Collate exactly one variable-length study bag into ordinary tensors."""
    if len(batch) != 1:
        raise ValueError("Study-bag loader must use batch_size=1")
    samples = batch[0]
    if not samples:
        raise ValueError("Study-bag loader received an empty bag")
    field_count = len(samples[0])
    if field_count != 8 or any(len(sample) != field_count for sample in samples):
        raise ValueError("Study-bag samples must use the multitask selector schema")
    images, targets, masks, keypoints, spacing, is_target, study_mls, study_ids = zip(*samples)
    if len(set(str(study_id) for study_id in study_ids)) != 1:
        raise ValueError("Study-bag rows span more than one study")
    return (
        torch.stack(images),
        torch.stack(targets),
        torch.stack(masks),
        torch.stack(keypoints),
        torch.stack(spacing),
        torch.stack(is_target),
        torch.stack(study_mls),
        tuple(str(study_id) for study_id in study_ids),
    )


def create_mls_positive_study_bag_loader(
    base_dataset: MLSHeatmapDataset,
    *,
    num_workers: int,
    deterministic_workers: bool,
) -> DataLoader:
    """Create a shuffled positive-study auxiliary loader without resampling rows."""
    bag_dataset = MLSPositiveStudyBagDataset(base_dataset)
    return DataLoader(
        bag_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mls_study_bag,
        worker_init_fn=seed_mls_loader_worker if deterministic_workers else None,
    )


class MLSPositiveStudyPairDataset(Dataset):
    """Return two annotated target slices from exactly one positive study.

    The ranking auxiliary deliberately receives only pairs, not a whole bag:
    it constrains selector order while preserving the existing independent
    slice geometry objective and its deployed p90 aggregation contract.
    """

    def __init__(self, base_dataset: MLSHeatmapDataset):
        if not base_dataset.return_selector:
            raise ValueError("Study-pair dataset requires return_selector=true")
        positive = base_dataset.data.loc[
            pd.to_numeric(base_dataset.data["is_target"], errors="raise") > 0.5
        ].copy()
        self._base_dataset = base_dataset
        self._pairs: list[tuple[int, int]] = []
        for _study_id, group in positive.groupby("patient_id", sort=True):
            indices = group.index.to_list()
            self._pairs.extend(combinations(indices, 2))
        if not self._pairs:
            raise ValueError(
                "Study-pair dataset requires a positive study with at least two target slices"
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[tuple, tuple]:
        first, second = self._pairs[index]
        return self._base_dataset[first], self._base_dataset[second]


def collate_mls_study_pair(batch: list[tuple[tuple, tuple]]) -> tuple:
    """Collate exactly one same-study annotated pair into ordinary tensors."""
    if len(batch) != 1:
        raise ValueError("Study-pair loader must use batch_size=1")
    first, second = batch[0]
    if len(first) != 8 or len(second) != 8:
        raise ValueError("Study-pair samples must use the multitask selector schema")
    if str(first[-1]) != str(second[-1]):
        raise ValueError("Study-pair rows span more than one study")
    fields = tuple(torch.stack(items) for items in zip(first[:-1], second[:-1]))
    return (*fields, (str(first[-1]), str(second[-1])))


def create_mls_positive_study_pair_loader(
    base_dataset: MLSHeatmapDataset,
    *,
    num_workers: int,
    deterministic_workers: bool,
) -> DataLoader:
    """Create a shuffled same-study pair loader without altering the main sampler."""
    pair_dataset = MLSPositiveStudyPairDataset(base_dataset)
    return DataLoader(
        pair_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mls_study_pair,
        worker_init_fn=seed_mls_loader_worker if deterministic_workers else None,
    )

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
    deterministic_workers: bool = False,
    input_channels: int = 3,
    context_cache_root: str | Path | None = None,
    context_cache_manifest_sha256: str | None = None,
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
        input_channels=input_channels,
        context_cache_root=context_cache_root,
        context_cache_manifest_sha256=context_cache_manifest_sha256,
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
        input_channels=input_channels,
        context_cache_root=context_cache_root,
        context_cache_manifest_sha256=context_cache_manifest_sha256,
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
        input_channels=input_channels,
        context_cache_root=context_cache_root,
        context_cache_manifest_sha256=context_cache_manifest_sha256,
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
        worker_init_fn=seed_mls_loader_worker if deterministic_workers else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_mls_loader_worker if deterministic_workers else None,
    )

    logger.info(
        f"DataLoaders: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"batch_size={batch_size}, sampling_mode={sampling_mode if sampler else 'shuffle'}"
    )

    return train_loader, val_loader
