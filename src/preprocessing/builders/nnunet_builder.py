"""
nnunet_builder.py — nnU-Net-specific dataset builder.

Reads raw DICOM + JSON annotations and produces NIfTI volumes in the
**nnU-Net raw format** (Dataset{id}_{name}/ with imagesTr/, labelsTr/,
and the ``_0000`` channel suffix).

For a **generic, strategy-agnostic** NIfTI builder (no nnUNet naming
conventions) see :class:`NiftiDatasetBuilder` in
``src/preprocessing/builders/nifti_builder.py``.

Usage note
----------
The core DICOM→NIfTI logic is identical to ``NiftiDatasetBuilder``.
If you do **not** plan to train nnU-Net, use ``NiftiDatasetBuilder``
instead — your data will be stored without nnUNet-specific naming.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.config import ICH_LABELS, NNUNET_RAW_DIR, RAW_TRAINING_DIR, RAW_ANNOTATIONS_DIR, get_annotated_patient_ids
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


class NNUnetDatasetBuilder:
    """
    nnU-Net-specific dataset builder.

    Writes NIfTI files to an nnU-Net-compatible directory structure:

        {NNUNET_RAW_DIR}/Dataset{id}_{name}/
        ├── imagesTr/   →  BRN_{pid}_0000.nii.gz   (HU volume)
        ├── labelsTr/   →  BRN_{pid}.nii.gz         (mask)
        └── dataset.json

    For non-nnUNet strategies (MONAI, SMP, YOLO), prefer the generic
    :class:`~src.preprocessing.builders.nifti_builder.NiftiDatasetBuilder`.
    """

    def __init__(
        self,
        raw_dicom_dir: Optional[str] = None,
        raw_json_dir: Optional[str] = None,
        nnunet_raw_dir: Optional[str] = None,
        dataset_id: int = 501,
        dataset_name: str = "BrainICH",
        image_prefix: str = "BRN",
    ):
        self.raw_dicom_dir = Path(raw_dicom_dir or RAW_TRAINING_DIR)
        self.raw_json_dir = Path(raw_json_dir or RAW_ANNOTATIONS_DIR)
        self.image_prefix = image_prefix

        self.dataset_name = f"Dataset{dataset_id}_{dataset_name}"
        self.out_dir = Path(nnunet_raw_dir or NNUNET_RAW_DIR) / self.dataset_name
        self.imagesTr = self.out_dir / "imagesTr"
        self.labelsTr = self.out_dir / "labelsTr"
        self.imagesTr.mkdir(parents=True, exist_ok=True)
        self.labelsTr.mkdir(parents=True, exist_ok=True)

    def build(self):
        """Execute the nnU-Net dataset build pipeline.

        .. note::
            For a generic builder without nnUNet naming, use
            ``NiftiDatasetBuilder`` instead.
        """
        logger.info(f"Building nnU-Net dataset: {self.dataset_name}")
        logger.info(f"  Output: {self.out_dir}")

        patient_ids = get_annotated_patient_ids(self.raw_json_dir)
        logger.info(f"  Found {len(patient_ids)} patients with annotations")

        processed = 0
        for pid in tqdm(patient_ids, desc="nnU-Net Build"):
            dicom_dir = self.raw_dicom_dir / pid
            json_dir = self.raw_json_dir / pid
            if not dicom_dir.exists() or not json_dir.exists():
                continue
            try:
                if self._process_patient(pid, str(dicom_dir), str(json_dir)):
                    processed += 1
            except Exception as e:
                logger.warning(f"  ⚠️  {pid}: {e}")
                continue

        self._generate_dataset_json(processed)
        logger.info(f"✅ nnU-Net: {processed} patients built")

    def _process_patient(self, pid, dicom_dir, json_dir):
        reader = BrainDicomReader(dicom_dir).load_and_sort()
        parser = AnnotationParser(json_dir)
        ann = parser.parse_all_slices(reader.slices)

        if not ann["has_annotation"] or ann["masks_3d"] is None:
            return False

        mask_3d = ann["masks_3d"]              # (D, H, W)
        image_3d = reader.get_3d_volume_hu()    # (H, W, D)
        mask_hwd = mask_3d.transpose(1, 2, 0)   # -> (H, W, D)

        meta = reader.metadata
        affine = np.diag([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"], 1.0])

        # Shape safety check
        min_d = min(image_3d.shape[2], mask_hwd.shape[2])
        image_3d = image_3d[:, :, :min_d]
        mask_hwd = mask_hwd[:, :, :min_d]

        # Save
        nib.save(nib.Nifti1Image(image_3d.astype(np.float32), affine),
                 str(self.imagesTr / f"{self.image_prefix}_{pid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(mask_hwd.astype(np.uint8), affine),
                 str(self.labelsTr / f"{self.image_prefix}_{pid}.nii.gz"))
        return True

    def _generate_dataset_json(self, num_training):
        info = {"channel_names": {"0": "CT"},
                "labels": {name: val for name, val in ICH_LABELS.items()},
                "numTraining": num_training,
                "file_ending": ".nii.gz"}
        with open(self.out_dir / "dataset.json", "w") as f:
            json.dump(info, f, indent=4)
        logger.info(f"  dataset.json written ({num_training} cases)")

    def __repr__(self):
        return f"NNUnetDatasetBuilder({self.dataset_name})"
