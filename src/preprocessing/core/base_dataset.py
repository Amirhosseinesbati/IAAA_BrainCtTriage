"""
base_dataset.py — Shared PyTorch Dataset base class for Brain CT studies.

Provides BrainCTDataset that all task-specific datasets (nnUNet, YOLO, MLS)
can inherit from or compose with. Handles:
- DICOM loading via BrainDicomReader (lazy)
- Annotation parsing via AnnotationParser
- Standard windowing transforms
- Consistent metadata extraction
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    RAW_TRAINING_DIR, RAW_ANNOTATIONS_DIR,
    WINDOWS, IMG_SIZE, ICH_LABEL_NAMES,
    get_patient_ids, get_annotated_patient_ids,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


class BrainCTDataset(Dataset):
    """
    PyTorch Dataset that provides lazy access to Brain CT studies.

    Each item returns the 3D HU volume + metadata for one patient/series.
    Task-specific subclasses should override __getitem__ for their needs.

    Args:
        patient_ids: List of patient directory IDs. If None, auto-detected.
        raw_dir: Root directory containing patient DICOM folders.
        annotation_dir: Root directory containing annotation JSON folders.
        transform: Optional callable applied to the HU volume.
        require_annotation: If True, only include patients with annotations.
        load_annotations: If True, also load and return annotation data.
    """

    def __init__(
        self,
        patient_ids: Optional[List[str]] = None,
        raw_dir: Union[str, Path] = RAW_TRAINING_DIR,
        annotation_dir: Optional[Union[str, Path]] = RAW_ANNOTATIONS_DIR,
        transform: Optional[Callable] = None,
        require_annotation: bool = False,
        load_annotations: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.transform = transform
        self.load_annotations = load_annotations

        # Determine patient list
        if patient_ids is None:
            all_patients = get_patient_ids(self.raw_dir)
            if require_annotation and self.annotation_dir:
                anno_patients = set(get_annotated_patient_ids(self.annotation_dir))
                self.patient_ids = [p for p in all_patients if p in anno_patients]
            else:
                self.patient_ids = all_patients
        else:
            self.patient_ids = patient_ids

        # Caches
        self._readers: dict = {}  # patient_id -> BrainDicomReader
        self._parsers: dict = {}  # patient_id -> AnnotationParser

        logger.info(
            f"BrainCTDataset: {len(self.patient_ids)} patients "
            f"(require_annotation={require_annotation}, "
            f"load_annotations={load_annotations})"
        )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _get_reader(self, idx: int) -> BrainDicomReader:
        """Lazy-load and cache BrainDicomReader for a patient."""
        pid = self.patient_ids[idx]
        if pid not in self._readers:
            patient_dir = str(self.raw_dir / pid)
            reader = BrainDicomReader(patient_dir)
            reader.load_and_sort()
            self._readers[pid] = reader
        return self._readers[pid]

    def _get_parser(self, idx: int) -> Optional[AnnotationParser]:
        """Lazy-load and cache AnnotationParser for a patient."""
        if not self.annotation_dir:
            return None
        pid = self.patient_ids[idx]
        if pid not in self._parsers:
            json_dir = str(self.annotation_dir / pid)
            if not os.path.isdir(json_dir):
                return None
            self._parsers[pid] = AnnotationParser(json_dir)
        return self._parsers[pid]

    def __getitem__(self, idx: int) -> dict:
        """
        Get data for one patient/series.

        Returns dict with:
            - patient_id (str)
            - hu_volume (np.ndarray): (H, W, D) Hounsfield Units
            - metadata (dict): DICOM metadata
            - annotations (dict, optional): if load_annotations=True
        """
        pid = self.patient_ids[idx]
        reader = self._get_reader(idx)

        result = {
            "patient_id": pid,
            "hu_volume": reader.get_3d_volume_hu(),
            "metadata": reader.metadata.copy(),
        }

        if self.load_annotations:
            parser = self._get_parser(idx)
            if parser is not None:
                result["annotations"] = parser.parse_all_slices(reader.slices)
            else:
                result["annotations"] = {"has_annotation": False}
                logger.debug(f"No annotations for patient {pid}")

        if self.transform:
            result["hu_volume"] = self.transform(result["hu_volume"])

        return result

    def get_metadata_df_row(self, idx: int) -> dict:
        """
        Get a row of summary metadata (useful for building training CSVs).

        Returns dict with patient-level info.
        """
        item = self[idx]
        meta = item["metadata"]
        row = {
            "patient_id": meta["patient_id"],
            "num_slices": meta["original_z_slices"],
            "spacing_x": meta["spacing_x"],
            "spacing_y": meta["spacing_y"],
            "spacing_z": meta["spacing_z"],
        }

        if self.load_annotations and item.get("annotations", {}).get("has_annotation"):
            ann = item["annotations"]
            if ann["masks_3d"] is not None:
                masks = ann["masks_3d"]
                vol_per_slice = []
                vol_px = masks * meta["spacing_x"] * meta["spacing_y"] * meta["spacing_z"] / 1000.0
                for label_val, name in ICH_LABEL_NAMES.items():
                    if label_val == 0:
                        continue
                    vol = vol_px[masks == label_val].sum()
                    row[f"V_{name}"] = float(vol)
                # MLS - from keypoints
                row["n_slices_with_kp"] = sum(
                    1 for kp in ann["keypoints_per_slice"] if kp
                )
                row["n_slices_with_boxes"] = sum(
                    1 for bb in ann["bboxes_per_slice"] if bb
                )

        return row


class WindowTransform:
    """
    Apply CT windowing to a 3D HU volume.

    Each slice in the volume is independently windowed.

    Args:
        window_name: Key in config.WINDOWS (e.g., "brain", "bone").
    """

    def __init__(self, window_name: str = "brain"):
        if window_name not in WINDOWS:
            raise KeyError(f"Unknown window '{window_name}'. Options: {list(WINDOWS.keys())}")
        self.window_name = window_name

    def __call__(self, hu_volume: np.ndarray) -> np.ndarray:
        """Apply windowing to each slice (H, W, D) -> (H, W, D) normalized [0,1]."""
        result = np.zeros_like(hu_volume, dtype=np.float32)
        for z in range(hu_volume.shape[2]):
            result[:, :, z] = BrainDicomReader.apply_windowing(
                hu_volume[:, :, z], self.window_name
            )
        return result


class MultiWindowTransform:
    """
    Create multi-channel windowed representation (like RGB).

    Stacks multiple windowed versions along a new first axis.

    Example:
        transform = MultiWindowTransform(["brain", "subdural", "bone"])
        # output shape: (3, H, W, D)
    """

    def __init__(self, window_names: List[str] = None):
        if window_names is None:
            window_names = ["brain", "subdural", "bone"]
        for wn in window_names:
            if wn not in WINDOWS:
                raise KeyError(f"Unknown window '{wn}'")
        self.window_names = window_names

    def __call__(self, hu_volume: np.ndarray) -> np.ndarray:
        """Stack windowed slices -> (n_windows, H, W, D)."""
        channels = []
        for wn in self.window_names:
            win = WindowTransform(wn)
            channels.append(win(hu_volume))
        return np.stack(channels, axis=0)
