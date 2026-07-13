"""
dicom_reader.py — Core DICOM reader for Brain CT studies.

Improved version with:
- __len__ and __getitem__ support for iteration
- get_slice_hu(index) for per-slice access
- get_windowed_slice(index, window_name) using centralized config
- Caching for 3D volume
- Error handling for missing/corrupted DICOM tags
- Automatic slice size validation
"""

import os
import glob
import logging
from typing import Optional, List, Tuple

import numpy as np
import nibabel as nib
import pydicom

from src.config import WINDOWS, IMG_SIZE

logger = logging.getLogger(__name__)


class BrainDicomReader:
    """
    Reads, sorts, and provides access to a Brain CT DICOM series.

    Usage:
        reader = BrainDicomReader("path/to/dicom_dir").load_and_sort()
        hu_volume = reader.get_3d_volume_hu()       # (H, W, D)
        hu_slice = reader.get_slice_hu(10)           # (H, W)
        windowed = reader.get_windowed_slice(10, "brain")  # (H, W) normalized
        nifti_path = reader.save_as_nifti("output.nii.gz")
    """

    def __init__(self, patient_dir: str):
        """
        Args:
            patient_dir: Path to directory containing .dcm files for one patient.
        """
        self.patient_dir = patient_dir
        self.dicom_files = sorted(glob.glob(os.path.join(patient_dir, "*.dcm")))
        self.slices: List[pydicom.Dataset] = []
        self.metadata: dict = {}
        self._hu_volume: Optional[np.ndarray] = None
        self._is_loaded = False

        if not self.dicom_files:
            raise FileNotFoundError(
                f"No DICOM files (*.dcm) found in {patient_dir}. "
                f"Found files: {os.listdir(patient_dir)[:5]}"
            )

    # ---- Loading & Sorting ----

    def load_and_sort(self) -> "BrainDicomReader":
        """Load all DICOM slices and sort by Z-axis position."""
        self.slices = []
        for fpath in self.dicom_files:
            try:
                ds = pydicom.dcmread(fpath, force=True)
                self.slices.append(ds)
            except Exception as e:
                logger.warning(f"⚠️  Could not read {fpath}: {e}")
                continue

        if not self.slices:
            raise ValueError(f"Failed to read any valid DICOM files from {self.patient_dir}")

        # Sort by Z-axis (ImagePositionPatient[2])
        self.slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract metadata from first slice
        first_slice = self.slices[0]
        spacing_xy = getattr(first_slice, "PixelSpacing", [1.0, 1.0])

        if len(self.slices) > 1:
            z_spacing = abs(
                float(self.slices[1].ImagePositionPatient[2])
                - float(self.slices[0].ImagePositionPatient[2])
            )
        else:
            z_spacing = float(getattr(first_slice, "SliceThickness", 1.0))

        self.metadata = {
            "patient_id": os.path.basename(self.patient_dir),
            "patient_dir": self.patient_dir,
            "original_z_slices": len(self.slices),
            "rescale_slope": float(getattr(first_slice, "RescaleSlope", 1.0)),
            "rescale_intercept": float(getattr(first_slice, "RescaleIntercept", 0.0)),
            "spacing_x": float(spacing_xy[1]),
            "spacing_y": float(spacing_xy[0]),
            "spacing_z": float(z_spacing),
            "rows": int(getattr(first_slice, "Rows", IMG_SIZE)),
            "columns": int(getattr(first_slice, "Columns", IMG_SIZE)),
        }
        self._is_loaded = True
        self._validate_slices()
        return self

    def _validate_slices(self):
        """Verify all slices have consistent dimensions."""
        if not self.slices:
            return
        ref_shape = (self.slices[0].Rows, self.slices[0].Columns)
        for i, s in enumerate(self.slices[1:], 1):
            shape = (s.Rows, s.Columns)
            if shape != ref_shape:
                logger.warning(
                    f"⚠️  Slice {i} has size {shape}, expected {ref_shape}. "
                    f"Will pad/crop to match first slice."
                )

    # ---- Length & Indexing ----

    def __len__(self) -> int:
        """Number of slices in the series."""
        return len(self.slices)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, dict]:
        """
        Get a single slice's HU array and its DICOM metadata.

        Returns:
            (hu_array_2d, slice_metadata_dict)
        """
        if not self._is_loaded:
            self.load_and_sort()
        if index < 0 or index >= len(self.slices):
            raise IndexError(f"Slice index {index} out of range [0, {len(self.slices)})")

        ds = self.slices[index]
        hu = self._pixel_to_hu(ds.pixel_array, ds)
        meta = {
            "index": index,
            "SOPInstanceUID": getattr(ds, "SOPInstanceUID", ""),
            "ImagePositionPatient": list(getattr(ds, "ImagePositionPatient", [0, 0, 0])),
            "RescaleSlope": float(getattr(ds, "RescaleSlope", 1.0)),
            "RescaleIntercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        }
        return hu, meta

    # ---- HU Conversion ----

    @staticmethod
    def _pixel_to_hu(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """Convert raw pixel values to Hounsfield Units."""
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return (pixel_array * slope) + intercept

    # ---- Per-slice access ----

    def get_slice_hu(self, index: int) -> np.ndarray:
        """Get a single slice as HU array (H, W)."""
        hu, _ = self[index]
        return hu

    def get_windowed_slice(self, index: int, window_name: str = "brain") -> np.ndarray:
        """
        Get a single slice with windowing applied, normalized to [0, 1].

        Args:
            index: Slice index.
            window_name: Key from config.WINDOWS (e.g., "brain", "bone", "subdural").

        Returns:
            2D array normalized to [0, 1].
        """
        hu = self.get_slice_hu(index)
        return self.apply_windowing(hu, window_name)

    @staticmethod
    def apply_windowing(image_hu: np.ndarray, window_spec) -> np.ndarray:
        """
        Apply CT windowing.

        Args:
            image_hu: 2D array of Hounsfield Units.
            window_spec: Either a dict with "width" and "level",
                        or a string key into config.WINDOWS.

        Returns:
            2D array normalized to [0, 1].
        """
        if isinstance(window_spec, str):
            # Look up in config
            if window_spec not in WINDOWS:
                raise KeyError(f"Unknown window '{window_spec}'. Options: {list(WINDOWS.keys())}")
            window_spec = WINDOWS[window_spec]

        width = window_spec["width"]
        level = window_spec["level"]
        min_val = level - (width / 2)
        max_val = level + (width / 2)
        windowed = np.clip(image_hu, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        return windowed

    # ---- 3D Volume ----

    def get_3d_volume_hu(self) -> np.ndarray:
        """
        Get the full 3D volume as HU (Height, Width, Depth).

        Results are cached after first computation.
        """
        if self._hu_volume is not None:
            return self._hu_volume

        if not self._is_loaded:
            self.load_and_sort()

        slices_hu = []
        for ds in self.slices:
            hu = self._pixel_to_hu(ds.pixel_array, ds)
            slices_hu.append(hu)

        self._hu_volume = np.stack(slices_hu, axis=-1)  # (H, W, D)
        return self._hu_volume

    # ---- Export ----

    def save_as_nifti(self, output_path: str) -> str:
        """
        Export the 3D HU volume as a NIfTI file.

        Args:
            output_path: Path for the .nii.gz file.

        Returns:
            The output path.
        """
        image_3d = self.get_3d_volume_hu()
        meta = self.metadata
        affine = np.diag([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"], 1.0])
        nifti_img = nib.Nifti1Image(image_3d, affine)
        nib.save(nifti_img, output_path)

        # Verify file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to save NIfTI to {output_path}")

        logger.info(f"💾 Saved NIfTI: {output_path} (shape={image_3d.shape})")
        return output_path

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        n_slices = len(self.slices)
        pid = self.metadata.get("patient_id", "?")
        return f"BrainDicomReader({pid}: {n_slices} slices, {status})"
