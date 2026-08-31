"""Validate categorical 2.5D masks and resized physical-volume accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategies.ich_2p5d.cache import CLASS_IDS, OUTPUT_LABELS, file_sha256
from src.strategies.ich_2p5d.segmentation_data import load_segmentation_manifest
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path", default="Data/processed/ich_2p5d/slice_manifest.csv"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = load_segmentation_manifest(args.manifest_path)
    reconstructed_rows: list[dict[str, object]] = []
    total_bytes = 0
    for study_id, group in frame.groupby("study_id", sort=True):
        paths = group["label_cache_path"].astype(str).unique()
        if len(paths) != 1:
            raise ValueError(f"Study {study_id} has inconsistent label cache paths")
        path = Path(paths[0])
        labels = np.load(path, mmap_mode="r")
        total_bytes += path.stat().st_size
        if labels.dtype != np.uint8 or labels.ndim != 3:
            raise ValueError(f"Invalid label array for {study_id}: {labels.shape}/{labels.dtype}")
        if len(labels) != len(group):
            raise ValueError(f"Depth mismatch in cached labels for {study_id}")
        unique = set(int(value) for value in np.unique(labels))
        if not unique <= {0, *CLASS_IDS}:
            raise ValueError(f"Unsupported cached class ids for {study_id}: {sorted(unique)}")
        ordered = group.sort_values("slice_index")
        known = ordered["known"].to_numpy(dtype=np.uint8) > 0
        if np.any(labels[~known] != 0):
            raise ValueError(f"Unknown slices contain cached labels in study {study_id}")
        for class_id, output_label in zip(CLASS_IDS, OUTPUT_LABELS[1:], strict=True):
            observed_presence = np.any(labels == class_id, axis=(1, 2)).astype(np.uint8)
            manifest_presence = ordered[output_label].to_numpy(dtype=np.uint8)
            if not np.array_equal(observed_presence[known], manifest_presence[known]):
                raise ValueError(
                    f"Mask/manifest subtype mismatch for {study_id}/{output_label}"
                )
        voxel_values = ordered["resized_voxel_volume_ml"].to_numpy(dtype=np.float64)
        if not np.allclose(voxel_values, voxel_values[0]):
            raise ValueError(f"Inconsistent voxel volume within study {study_id}")
        row: dict[str, object] = {"study_id": str(study_id)}
        for class_id, volume_key in zip(CLASS_IDS, VOLUME_KEYS, strict=True):
            row[f"cached_{volume_key}"] = float(
                np.count_nonzero(labels == class_id) * voxel_values[0]
            )
        reconstructed_rows.append(row)

    cached = pd.DataFrame(reconstructed_rows)
    truth, metadata_source = ground_truth_ich_context()
    merged = cached.merge(
        truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != frame["study_id"].nunique():
        raise ValueError("Cached mask studies do not match ground-truth metadata")

    subtype_metrics: dict[str, dict[str, float | str]] = {}
    for key in VOLUME_KEYS:
        predicted = merged[f"cached_{key}"].to_numpy(dtype=np.float64)
        observed = merged[f"gt_{key}"].to_numpy(dtype=np.float64)
        absolute_error = np.abs(predicted - observed)
        worst_index = int(np.argmax(absolute_error))
        subtype_metrics[key] = {
            "mae_ml": float(np.mean(absolute_error)),
            "bias_ml": float(np.mean(predicted - observed)),
            "max_abs_error_ml": float(absolute_error[worst_index]),
            "max_abs_error_study_id": str(merged.iloc[worst_index]["study_id"]),
            "max_abs_error_cached_ml": float(predicted[worst_index]),
            "max_abs_error_ground_truth_ml": float(observed[worst_index]),
            "pearson": float(np.corrcoef(predicted, observed)[0, 1])
            if np.std(predicted) > 0 and np.std(observed) > 0
            else 0.0,
        }
    cached_total = merged[[f"cached_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    truth_total = merged[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    total_absolute_error = np.abs(cached_total - truth_total)
    total_worst_index = int(np.argmax(total_absolute_error.to_numpy()))
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest_path),
        "manifest_sha256": file_sha256(args.manifest_path),
        "metadata_source": str(metadata_source),
        "studies": int(frame["study_id"].nunique()),
        "slices": int(len(frame)),
        "known_slices": int(frame["known"].sum()),
        "unknown_slices": int((frame["known"] == 0).sum()),
        "label_cache_bytes": int(total_bytes),
        "spacing_thickness_mismatch_studies_gt_5pct": int(
            frame.loc[
                (frame["spacing_to_thickness_ratio"] - 1.0).abs() > 0.05,
                "study_id",
            ].nunique()
        ),
        "spacing_to_thickness_ratio_min": float(
            frame["spacing_to_thickness_ratio"].min()
        ),
        "spacing_to_thickness_ratio_max": float(
            frame["spacing_to_thickness_ratio"].max()
        ),
        "total_volume_mae_ml": float(np.mean(total_absolute_error)),
        "total_volume_bias_ml": float(np.mean(cached_total - truth_total)),
        "total_volume_pearson": float(np.corrcoef(cached_total, truth_total)[0, 1]),
        "total_volume_max_abs_error_ml": float(total_absolute_error.iloc[total_worst_index]),
        "total_volume_max_abs_error_study_id": str(
            merged.iloc[total_worst_index]["study_id"]
        ),
        "subtypes": subtype_metrics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
