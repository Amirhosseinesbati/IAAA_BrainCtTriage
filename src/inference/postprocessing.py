"""Physical-space cleanup for raw task predictions before calibration."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from src.config import config_section
from src.evaluation.triage import VOLUME_KEYS, validate_intermediates


def remove_small_components(
    label_map: np.ndarray,
    *,
    voxel_volume_ml: float,
    min_component_ml: Mapping[int, float],
) -> np.ndarray:
    """Remove 3D connected components smaller than per-label physical limits."""
    from scipy import ndimage

    if label_map.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got shape {label_map.shape}")
    if voxel_volume_ml <= 0 or not math.isfinite(voxel_volume_ml):
        raise ValueError("voxel_volume_ml must be positive and finite")
    cleaned = np.asarray(label_map).copy()
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    for label_id, minimum_ml in min_component_ml.items():
        if minimum_ml <= 0:
            continue
        components, count = ndimage.label(cleaned == int(label_id), structure=structure)
        if not count:
            continue
        voxel_counts = np.bincount(components.ravel())
        remove_ids = np.flatnonzero(voxel_counts * voxel_volume_ml < float(minimum_ml))
        remove_ids = remove_ids[remove_ids != 0]
        if len(remove_ids):
            cleaned[np.isin(components, remove_ids)] = 0
    return cleaned


def sanitize_intermediates(values: Mapping[str, float]) -> dict[str, float]:
    """Clamp model outputs to physical domains and suppress sub-noise volumes."""
    cleaned = validate_intermediates(values)
    floors = config_section("competition", "postprocessing", "ich_min_component_ml")
    for key in VOLUME_KEYS:
        value = max(0.0, cleaned[key])
        cleaned[key] = 0.0 if value < float(floors[key]) else value
    fracture_low, fracture_high = config_section(
        "competition", "postprocessing", "fracture_probability_clip"
    )
    cleaned["fracture_prob"] = float(np.clip(cleaned["fracture_prob"], fracture_low, fracture_high))
    mls_low, mls_high = config_section("competition", "postprocessing", "mls_clip_mm")
    cleaned["MLS_mm"] = float(np.clip(cleaned["MLS_mm"], mls_low, mls_high))
    return cleaned
