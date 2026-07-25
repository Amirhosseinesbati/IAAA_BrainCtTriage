"""
nifti_builder.py — Strategy-agnostic NIfTI dataset builder for ICH.

Reads raw DICOM + JSON annotations and produces NIfTI volumes (HU and
mask) in a **generic directory structure** that any strategy (MONAI,
SMP, YOLO, nnU-Net) can consume.

Directory layout::

    {ICH_NIFTI_DIR}/BrainICH/
    ├── images/            # HU volume NIfTI  (H, W, D), float32
    │   └── BRN_{pid}.nii.gz
    ├── labels/            # uint8 mask NIfTI  (H, W, D)
    │   └── BRN_{pid}.nii.gz
    └── dataset.json       # label mapping + metadata

The naming convention **does not** use nnU-Net suffixes (``_0000``)
or subdirectories (``imagesTr`` / ``labelsTr``) — this is the
fundamental decoupling from nnUNet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.config import (
    ICH_LABELS,
    ICH_NIFTI_DIR,
    RAW_TRAINING_DIR,
    RAW_ANNOTATIONS_DIR,
    get_annotated_patient_ids,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


class NiftiDatasetBuilder:
    """
    Build a generic NIfTI dataset from raw DICOM + JSON annotations.

    The output is stored under *ICH_NIFTI_DIR* and is **not** tied to
    any particular segmentation framework.  Every ICH strategy (MONAI,
    SMP, YOLO, nnU-Net) can load from this location.

    Parameters
    ----------
    raw_dicom_dir : Path or str, optional
        Directory containing per-patient DICOM subdirectories.
    raw_json_dir : Path or str, optional
        Directory containing per-patient JSON annotation subdirectories.
    output_dir : Path or str, optional
        Root output directory (default ``ICH_NIFTI_DIR``).
    dataset_name : str
        Name subdirectory under *output_dir* (default ``"BrainICH"``).
    image_prefix : str
        Prefix for NIfTI filenames (default ``"BRN"``).
    """

    def __init__(
        self,
        raw_dicom_dir: Optional[str] = None,
        raw_json_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        dataset_name: str = "BrainICH",
        image_prefix: str = "BRN",
    ):
        self.raw_dicom_dir = Path(raw_dicom_dir or RAW_TRAINING_DIR)
        self.raw_json_dir = Path(raw_json_dir or RAW_ANNOTATIONS_DIR)
        self.image_prefix = image_prefix

        self.dataset_name = dataset_name
        self.out_dir = Path(output_dir or ICH_NIFTI_DIR) / dataset_name
        self.images_dir = self.out_dir / "images"
        self.labels_dir = self.out_dir / "labels"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> int:
        """Execute the full dataset build pipeline.

        Returns
        -------
        int
            Number of successfully processed patients.
        """
        logger.info("Building NIfTI dataset: %s", self.dataset_name)
        logger.info("  Output: %s", self.out_dir)

        patient_ids = get_annotated_patient_ids(self.raw_json_dir)
        logger.info("  Found %d patients with annotations", len(patient_ids))

        processed = 0
        for pid in tqdm(patient_ids, desc="NIfTI Build"):
            dicom_dir = self.raw_dicom_dir / pid
            json_dir = self.raw_json_dir / pid
            if not dicom_dir.exists() or not json_dir.exists():
                continue
            try:
                if self._process_patient(str(pid), str(dicom_dir), str(json_dir)):
                    processed += 1
            except Exception as exc:
                logger.warning("  ⚠️  %s: %s", pid, exc)
                continue

        self._write_dataset_json()
        logger.info("✅ NIfTI dataset: %d patients built → %s", processed, self.out_dir)
        return processed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_patient(
        self,
        pid: str,
        dicom_dir: str,
        json_dir: str,
    ) -> bool:
        """Convert one patient's DICOM + JSON to paired NIfTI files.

        Returns ``True`` if the patient had valid annotations and was
        saved, ``False`` otherwise.
        """
        reader = BrainDicomReader(dicom_dir).load_and_sort()
        parser = AnnotationParser(json_dir)
        ann = parser.parse_all_slices(reader.slices)

        if not ann["has_annotation"] or ann["masks_3d"] is None:
            return False

        mask_3d = ann["masks_3d"]               # (D, H, W)
        image_3d = reader.get_3d_volume_hu()     # (H, W, D)
        mask_hwd = mask_3d.transpose(1, 2, 0)    # → (H, W, D)

        meta = reader.metadata
        affine = np.diag([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"], 1.0])

        # Trim to matching depth
        min_d = min(image_3d.shape[2], mask_hwd.shape[2])
        image_3d = image_3d[:, :, :min_d]
        mask_hwd = mask_hwd[:, :, :min_d]

        # Save
        nib.save(
            nib.Nifti1Image(image_3d.astype(np.float32), affine),
            str(self.images_dir / f"{self.image_prefix}_{pid}.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(mask_hwd.astype(np.uint8), affine),
            str(self.labels_dir / f"{self.image_prefix}_{pid}.nii.gz"),
        )
        return True

    def _write_dataset_json(self) -> None:
        """Write dataset.json with label mapping."""
        info = {
            "dataset_name": self.dataset_name,
            "labels": {name: val for name, val in ICH_LABELS.items()},
        }
        with open(self.out_dir / "dataset.json", "w") as f:
            json.dump(info, f, indent=4)
        logger.info("  dataset.json written → %s", self.out_dir / "dataset.json")

    def __repr__(self) -> str:
        return f"NiftiDatasetBuilder({self.dataset_name})"
