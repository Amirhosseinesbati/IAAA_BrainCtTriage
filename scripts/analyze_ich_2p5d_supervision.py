"""Quantify subtype imbalance in the audited 2.5D ICH supervision cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategies.ich_2p5d.cache import CLASS_IDS, OUTPUT_LABELS, file_sha256
from src.strategies.ich_2p5d.segmentation_data import load_segmentation_manifest


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {name: 0.0 for name in ("min", "p10", "median", "p90", "max")}
    quantiles = np.quantile(values.astype(np.float64), [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "median", "p90", "max"), quantiles, strict=True
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path", default="Data/processed/ich_2p5d/slice_manifest.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ich_experiments/2p5d_segmentation/supervision_analysis"),
    )
    args = parser.parse_args()

    frame = load_segmentation_manifest(args.manifest_path).sort_values(
        ["study_id", "slice_index"]
    )
    rows: list[dict[str, object]] = []
    for study_id, group in frame.groupby("study_id", sort=True):
        ordered = group.sort_values("slice_index").reset_index(drop=True)
        paths = ordered["label_cache_path"].astype(str).unique()
        if len(paths) != 1:
            raise ValueError(f"Study {study_id} has inconsistent label cache paths")
        labels = np.load(paths[0], mmap_mode="r")
        if len(labels) != len(ordered):
            raise ValueError(f"Study {study_id} label depth does not match its manifest")
        for index, record in enumerate(ordered.itertuples(index=False)):
            row: dict[str, object] = {
                "study_id": str(record.study_id),
                "patient_id": str(record.patient_id),
                "fold": int(record.fold),
                "slice_index": int(record.slice_index),
                "classification_known": int(record.classification_known),
                "segmentation_known": int(record.segmentation_known),
                "metadata_missing": int(record.metadata_missing),
                "supervision_mismatch": int(record.supervision_mismatch),
                "voxel_volume_ml": float(record.resized_voxel_volume_ml),
            }
            for class_id, label in zip(CLASS_IDS, OUTPUT_LABELS[1:], strict=True):
                row[f"metadata_{label}"] = int(getattr(record, label))
                row[f"pixels_{label}"] = int(np.count_nonzero(labels[index] == class_id))
            rows.append(row)

    slices = pd.DataFrame(rows)
    known = slices["segmentation_known"] == 1
    height = int(frame["native_height"].iloc[0])
    width = int(frame["native_width"].iloc[0])
    cache_pixels_per_slice = int(
        np.load(frame["label_cache_path"].iloc[0], mmap_mode="r").shape[-1] ** 2
    )
    total_known_pixels = int(known.sum()) * cache_pixels_per_slice
    subtype_summary: dict[str, dict[str, object]] = {}
    fold_rows: list[dict[str, object]] = []
    for label in OUTPUT_LABELS[1:]:
        pixel_counts = slices[f"pixels_{label}"].to_numpy(dtype=np.int64)
        spatial_positive = known.to_numpy() & (pixel_counts > 0)
        metadata_positive = (
            (slices["classification_known"].to_numpy(dtype=np.uint8) > 0)
            & (slices[f"metadata_{label}"].to_numpy(dtype=np.uint8) > 0)
        )
        positive_pixels = pixel_counts[spatial_positive]
        subtype_summary[label] = {
            "metadata_positive_slices": int(metadata_positive.sum()),
            "spatial_positive_slices": int(spatial_positive.sum()),
            "metadata_positive_without_spatial_supervision": int(
                (metadata_positive & ~known.to_numpy()).sum()
            ),
            "spatial_positive_studies": int(
                slices.loc[spatial_positive, "study_id"].nunique()
            ),
            "spatial_positive_patients": int(
                slices.loc[spatial_positive, "patient_id"].nunique()
            ),
            "total_positive_pixels": int(positive_pixels.sum()),
            "fraction_of_known_pixels": float(
                positive_pixels.sum() / max(1, total_known_pixels)
            ),
            "positive_slice_pixel_quantiles": _quantiles(positive_pixels),
            "positive_slice_volume_ml_quantiles": _quantiles(
                positive_pixels
                * slices.loc[spatial_positive, "voxel_volume_ml"].to_numpy(
                    dtype=np.float64
                )
            ),
        }
        for fold, fold_group in slices.groupby("fold", sort=True):
            fold_spatial_positive = (
                (fold_group["segmentation_known"].to_numpy(dtype=np.uint8) > 0)
                & (fold_group[f"pixels_{label}"].to_numpy(dtype=np.int64) > 0)
            )
            fold_rows.append({
                "fold": int(fold),
                "label": label,
                "positive_slices": int(fold_spatial_positive.sum()),
                "positive_studies": int(
                    fold_group.loc[fold_spatial_positive, "study_id"].nunique()
                ),
                "positive_patients": int(
                    fold_group.loc[fold_spatial_positive, "patient_id"].nunique()
                ),
                "positive_pixels": int(
                    fold_group.loc[fold_spatial_positive, f"pixels_{label}"].sum()
                ),
            })

    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest_path),
        "manifest_sha256": file_sha256(args.manifest_path),
        "studies": int(slices["study_id"].nunique()),
        "patients": int(slices["patient_id"].nunique()),
        "slices": int(len(slices)),
        "classification_known_slices": int(slices["classification_known"].sum()),
        "segmentation_known_slices": int(slices["segmentation_known"].sum()),
        "metadata_missing_slices": int(slices["metadata_missing"].sum()),
        "supervision_mismatch_slices": int(slices["supervision_mismatch"].sum()),
        "cache_pixels_per_slice": cache_pixels_per_slice,
        "representative_native_shape": [height, width],
        "subtypes": subtype_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slices.to_csv(args.output_dir / "slice_pixel_counts.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        args.output_dir / "fold_subtype_counts.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
