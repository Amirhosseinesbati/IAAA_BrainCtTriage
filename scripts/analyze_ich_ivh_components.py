"""Aggregate patient-safe IVH connected-component statistics for one nested split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, file_sha256
from src.strategies.ich_2p5d.segmentation_data import (
    load_segmentation_manifest,
    split_segmentation_slices,
)


IVH_CLASS_ID = OUTPUT_LABELS.index("IVH")
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def component_measurements(
    mask: np.ndarray, *, voxel_volume_ml: float
) -> tuple[list[int], list[float]]:
    """Return 8-connected component areas and physical single-slice volumes."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask, got {binary.shape}")
    if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0:
        raise ValueError("voxel_volume_ml must be finite and positive")
    labels, count = connected_components(binary, structure=CONNECTIVITY_8)
    if count == 0:
        return [], []
    areas = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
    area_values = [int(value) for value in areas]
    return area_values, [float(value * voxel_volume_ml) for value in areas]


def numeric_summary(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize_scope(frame: pd.DataFrame) -> dict[str, Any]:
    supervised = frame.loc[frame["segmentation_known"].eq(1)].copy()
    arrays: dict[str, np.ndarray] = {}
    component_areas: list[int] = []
    component_volumes: list[float] = []
    components_per_positive_slice: list[int] = []
    studies: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "positive_slices": 0.0,
            "components": 0.0,
            "pixels": 0.0,
            "volume_ml": 0.0,
        }
    )
    mask_positive_slices = 0
    metadata_positive_slices = int(supervised["IVH"].sum())
    metadata_positive_mask_empty_slices = 0
    metadata_negative_mask_positive_slices = 0
    for row in supervised.itertuples(index=False):
        cache_path = str(row.label_cache_path)
        if cache_path not in arrays:
            arrays[cache_path] = np.load(cache_path, mmap_mode="r")
        target = np.asarray(arrays[cache_path][int(row.slice_index)])
        ivh = target == IVH_CLASS_ID
        areas, volumes = component_measurements(
            ivh,
            voxel_volume_ml=float(row.resized_voxel_volume_ml),
        )
        if not areas:
            if int(row.IVH) == 1:
                metadata_positive_mask_empty_slices += 1
            continue
        if int(row.IVH) == 0:
            metadata_negative_mask_positive_slices += 1
        mask_positive_slices += 1
        components_per_positive_slice.append(len(areas))
        component_areas.extend(areas)
        component_volumes.extend(volumes)
        study = studies[str(row.study_id)]
        study["positive_slices"] += 1
        study["components"] += len(areas)
        study["pixels"] += sum(areas)
        study["volume_ml"] += sum(volumes)

    study_values = list(studies.values())
    area_array = np.asarray(component_areas, dtype=np.float64)
    volume_array = np.asarray(component_volumes, dtype=np.float64)
    return {
        "slices": int(len(frame)),
        "studies": int(frame["study_id"].nunique()),
        "spatially_supervised_slices": int(len(supervised)),
        "metadata_positive_spatially_supervised_slices": metadata_positive_slices,
        "mask_positive_slices": int(mask_positive_slices),
        "metadata_positive_mask_empty_slices": int(
            metadata_positive_mask_empty_slices
        ),
        "metadata_negative_mask_positive_slices": int(
            metadata_negative_mask_positive_slices
        ),
        "mask_positive_studies": int(len(studies)),
        "components": int(len(component_areas)),
        "component_area_pixels": numeric_summary(component_areas),
        "component_single_slice_volume_ml": numeric_summary(component_volumes),
        "components_per_positive_slice": numeric_summary(
            components_per_positive_slice
        ),
        "positive_slices_per_positive_study": numeric_summary(
            [value["positive_slices"] for value in study_values]
        ),
        "components_per_positive_study": numeric_summary(
            [value["components"] for value in study_values]
        ),
        "mask_volume_ml_per_positive_study": numeric_summary(
            [value["volume_ml"] for value in study_values]
        ),
        "component_area_threshold_counts": {
            f"le_{threshold}px": int((area_array <= threshold).sum())
            for threshold in (4, 9, 16, 25, 49, 100, 256, 1024)
        },
        "component_volume_threshold_counts": {
            f"le_{threshold:g}ml": int((volume_array <= threshold).sum())
            for threshold in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        },
        "privacy": "aggregated_only_no_study_or_patient_identifiers",
    }


def analyze_manifest(
    manifest_path: Path, *, outer_fold: int, calibration_fold: int
) -> dict[str, Any]:
    frame = load_segmentation_manifest(manifest_path)
    training, calibration, outer = split_segmentation_slices(
        frame,
        outer_fold=outer_fold,
        calibration_fold=calibration_fold,
    )
    return {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "outer_fold": int(outer_fold),
        "calibration_fold": int(calibration_fold),
        "subtype": "IVH",
        "class_id": IVH_CLASS_ID,
        "connectivity": "8-connected_2d",
        "scopes": {
            "training": summarize_scope(training),
            "calibration": summarize_scope(calibration),
            "outer": summarize_scope(outer),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("Data/processed/ich_2p5d/slice_manifest.csv"),
    )
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_manifest(
        args.manifest_path,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
