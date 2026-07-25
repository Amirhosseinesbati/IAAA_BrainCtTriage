"""
yolo_seg/data_prep.py — Convert NIfTI ICH masks to YOLO instance segmentation format.

Reads NIfTI volumes from *ICH_NIFTI_DIR* (generic NIfTI storage, no nnUNet
naming) and produces:
- images/train/, images/val/  —  2D axial slice PNGs
- labels/train/, labels/val/  —  YOLO segmentation per-instance labels
- dataset.yaml               —  dataset configuration for YOLO
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import nibabel as nib
import numpy as np
import yaml

from src.config import ICH_LABEL_NAMES, ICH_NIFTI_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Mapping: nnU-Net label index → YOLO class index (skip 0=background)
# ICH_LABELS: {"background": 0, "IVH": 1, "IPH": 2, "SDH": 3, "EDH": 4, "SAH": 5}
# YOLO needs 0-based contiguous IDs for foreground classes:
YOLO_CLASS_ID = {
    1: 0,  # IVH
    2: 1,  # IPH
    3: 2,  # SDH
    4: 3,  # EDH
    5: 4,  # SAH
}

YOLO_CLASS_NAMES = ["IVH", "IPH", "SDH", "EDH", "SAH"]


def _find_dataset_folder(data_dir: Path) -> Optional[Path]:
    """Locate the dataset folder inside *ICH_NIFTI_DIR* (generic layout)."""
    for candidate in sorted(data_dir.iterdir()):
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    return None


def _mask_to_yolo_polygons(mask_slice: np.ndarray) -> list[str]:
    """
    Convert a 2D segmentation mask to YOLO segmentation format.

    Returns a list of YOLO label strings: "class_id x1 y1 x2 y2 ..."
    One string per connected component (instance).

    YOLO format: normalized coordinates (0-1).
    """
    lines = []
    h, w = mask_slice.shape

    for nnunet_label, yolo_cls in YOLO_CLASS_ID.items():
        binary = (mask_slice == nnunet_label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:  # Need at least 3 points for a polygon
                continue

            # Simplify contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Normalize coordinates
            points = []
            for pt in approx:
                x, y = pt[0]
                points.extend([x / w, y / h])

            if points:
                lines.append(f"{yolo_cls} " + " ".join(f"{p:.6f}" for p in points))

    return lines


def prepare_yolo_seg_data(
    data_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Convert generic NIfTI dataset (from *ICH_NIFTI_DIR*) to YOLO instance
    segmentation format.

    Parameters
    ----------
    data_dir : Path, optional
        *ICH_NIFTI_DIR* (generic NIfTI directory).
    out_dir : Path, optional
        Output root for YOLO dataset.
    val_ratio : float
        Fraction of volumes for validation.
    seed : int
        Random seed for reproducible train/val split.
    """
    data_dir = Path(data_dir or ICH_NIFTI_DIR)
    out_dir = Path(out_dir or (PROCESSED_DIR / "yolo_ich_seg"))

    dataset_folder = _find_dataset_folder(data_dir)
    if dataset_folder is None:
        raise FileNotFoundError(
            f"No dataset folder found in {data_dir}. "
            "Run NiftiDatasetBuilder().build() first."
        )

    images_dir = dataset_folder / "images"
    labels_dir = dataset_folder / "labels"

    image_paths = sorted(images_dir.glob("*.nii.gz"))
    if not image_paths:
        raise FileNotFoundError(f"No NIfTI images in {images_tr}")

    logger.info("Found %d volumes in %s", len(image_paths), dataset_folder)

    # Train / val split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(image_paths))
    val_count = max(1, int(len(image_paths) * val_ratio))
    val_indices = set(indices[:val_count].tolist())

    # Prepare output directories
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count_final = 0

    for vol_idx, img_path in enumerate(image_paths):
        is_val = vol_idx in val_indices
        split = "val" if is_val else "train"

        # Load volume
        img_nii = nib.load(str(img_path))
        image_3d = img_nii.get_fdata().astype(np.float32)  # (H, W, D)

        label_path = labels_dir / img_path.name
        mask_3d = None
        if label_path.exists():
            mask_nii = nib.load(str(label_path))
            mask_3d = mask_nii.get_fdata().astype(np.int64)  # (H, W, D)

        depth = image_3d.shape[2]
        base_name = img_path.name.replace(".nii.gz", "")

        for s in range(depth):
            image_slice = image_3d[:, :, s]

            # Normalize to [0, 255] for YOLO
            p_low, p_high = np.percentile(image_slice[image_slice > -900], [0.5, 99.5])
            img_uint8 = np.clip(
                (image_slice - p_low) / max(p_high - p_low, 1e-6) * 255, 0, 255,
            ).astype(np.uint8)

            # Convert to 3-channel RGB (YOLO expects 3 channels)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

            slice_name = f"{base_name}_slice{s:04d}"
            img_out = out_dir / "images" / split / f"{slice_name}.png"
            cv2.imwrite(str(img_out), img_rgb)

            # Generate YOLO segmentation labels
            if mask_3d is not None:
                mask_slice = mask_3d[:, :, s]
                yolo_lines = _mask_to_yolo_polygons(mask_slice)
                if yolo_lines:
                    label_out = out_dir / "labels" / split / f"{slice_name}.txt"
                    with open(label_out, "w") as f:
                        f.write("\n".join(yolo_lines))

        if is_val:
            val_count_final += 1
        else:
            train_count += 1

    # ── Generate dataset.yaml ─────────────────────────────────────
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(YOLO_CLASS_NAMES)},
        "nc": len(YOLO_CLASS_NAMES),
    }

    with open(out_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logger.info(
        "YOLO Seg dataset ready: %d train / %d val volumes → %s",
        train_count, val_count_final, out_dir,
    )
