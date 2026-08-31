"""DICOM-to-NIfTI geometry and physically meaningful ICH volumes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LABEL_TO_VOLUME_KEY: Mapping[int, str] = {
    1: "V_IVH",
    2: "V_IPH",
    3: "V_SDH",
    4: "V_EDH",
    5: "V_SAH",
}


def _as_float_array(value: Any, *, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values, got {value!r}")
    return result


def dicom_affine_ras(slices: Sequence[Any]) -> np.ndarray:
    """Build an affine for an array indexed as ``(row, column, slice)``.

    DICOM patient coordinates are LPS while NIfTI convention is RAS.  The
    first ImageOrientationPatient vector follows increasing columns and the
    second follows increasing rows, so the two in-plane affine columns are
    deliberately swapped relative to their order in the DICOM tag.
    """
    if not slices:
        raise ValueError("At least one DICOM slice is required")

    first = slices[0]
    orientation = _as_float_array(
        getattr(first, "ImageOrientationPatient", None),
        length=6,
        name="ImageOrientationPatient",
    )
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    row_spacing, column_spacing = _as_float_array(
        getattr(first, "PixelSpacing", None), length=2, name="PixelSpacing"
    )
    origin_lps = _as_float_array(
        getattr(first, "ImagePositionPatient", None),
        length=3,
        name="ImagePositionPatient",
    )

    if len(slices) > 1:
        positions = np.stack([
            _as_float_array(
                getattr(item, "ImagePositionPatient", None),
                length=3,
                name="ImagePositionPatient",
            )
            for item in slices
        ])
        slice_step_lps = np.median(np.diff(positions, axis=0), axis=0)
    else:
        normal = np.cross(row_direction, column_direction)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            raise ValueError("Degenerate ImageOrientationPatient vectors")
        thickness = float(getattr(first, "SpacingBetweenSlices", 0.0) or 0.0)
        if thickness <= 0:
            thickness = float(getattr(first, "SliceThickness", 1.0))
        slice_step_lps = normal / normal_norm * thickness

    basis_lps = np.column_stack(
        (
            column_direction * row_spacing,
            row_direction * column_spacing,
            slice_step_lps,
        )
    )
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = lps_to_ras @ basis_lps
    affine[:3, 3] = lps_to_ras @ origin_lps
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) <= 1e-8:
        raise ValueError("DICOM geometry produced a singular or non-finite affine")
    return affine


def voxel_volume_ml(affine: np.ndarray) -> float:
    """Return physical voxel volume in mL from a NIfTI affine."""
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Affine must be a finite 4x4 matrix")
    volume = abs(float(np.linalg.det(matrix[:3, :3]))) / 1000.0
    if volume <= 0:
        raise ValueError("Affine has non-positive voxel volume")
    return volume


def volumes_from_labelmap(
    labelmap: np.ndarray,
    affine: np.ndarray,
    *,
    label_to_key: Mapping[int, str] = LABEL_TO_VOLUME_KEY,
) -> dict[str, float]:
    """Measure each ICH subtype without resizing the prediction."""
    labels = np.asarray(labelmap)
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got shape {labels.shape}")
    per_voxel = voxel_volume_ml(affine)
    return {
        key: float(np.count_nonzero(labels == label) * per_voxel)
        for label, key in label_to_key.items()
    }
