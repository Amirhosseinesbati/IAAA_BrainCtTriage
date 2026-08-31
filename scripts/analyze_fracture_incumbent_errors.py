#!/usr/bin/env python3
"""Aggregate, privacy-safe phenotype analysis for fracture OOF errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom


NUMERIC_COLUMNS = [
    "n_slices",
    "positive_slices",
    "total_boxes",
    "prob_adjacent_pair",
    "mil_score",
    "reference_train_cdf",
    "candidate_train_cdf",
    "deployable_blend_score",
    "cdf_gap",
    "slice_thickness",
    "pixel_spacing_mean",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _text(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (list, tuple)):
        value = "|".join(str(item) for item in value)
    result = str(value).strip()
    return result if result else "missing"


def _dicom_metadata(raw_root: Path, study_id: str) -> dict[str, Any]:
    directory = raw_root / study_id
    files = sorted(directory.glob("*.dcm")) if directory.is_dir() else []
    if not files:
        return {"metadata_available": False}
    try:
        dataset = pydicom.dcmread(
            files[0],
            stop_before_pixels=True,
            specific_tags=[
                "Manufacturer",
                "ManufacturerModelName",
                "ConvolutionKernel",
                "SliceThickness",
                "PixelSpacing",
                "Rows",
                "Columns",
                "KVP",
            ],
        )
    except Exception:
        return {"metadata_available": False}
    spacing = getattr(dataset, "PixelSpacing", None)
    spacing_values = [] if spacing is None else [_float(value) for value in spacing]
    spacing_values = [value for value in spacing_values if value is not None]
    return {
        "metadata_available": True,
        "manufacturer": _text(getattr(dataset, "Manufacturer", None)).upper(),
        "model": _text(getattr(dataset, "ManufacturerModelName", None)).upper(),
        "kernel": _text(getattr(dataset, "ConvolutionKernel", None)).upper(),
        "slice_thickness": _float(getattr(dataset, "SliceThickness", None)),
        "pixel_spacing_mean": (
            float(np.mean(spacing_values)) if spacing_values else None
        ),
        "rows": _float(getattr(dataset, "Rows", None)),
        "columns": _float(getattr(dataset, "Columns", None)),
        "kvp": _float(getattr(dataset, "KVP", None)),
    }


def _load_validation_manifest(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("fold_*/manifest.csv")):
        fold = int(path.parent.name.split("_")[-1])
        frame = pd.read_csv(path)
        frame = frame.loc[frame["split"].astype(str).str.lower() == "val"].copy()
        frame["outer_fold"] = fold
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No fold manifests under {root}")
    slices = pd.concat(frames, ignore_index=True)
    result = (
        slices.groupby(["study_id", "outer_fold"], as_index=False)
        .agg(
            n_slices=("slice_index", "size"),
            positive_slices=("slice_fracture", "sum"),
            total_boxes=("n_boxes", "sum"),
            manifest_truth=("study_fracture", "first"),
        )
    )
    if result["study_id"].duplicated().any():
        raise ValueError("Validation study appears in more than one outer fold")
    return result


def _numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        result[column] = {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "q25": float(values.quantile(0.25)) if len(values) else None,
            "q75": float(values.quantile(0.75)) if len(values) else None,
        }
    return result


def _group_rates(
    frame: pd.DataFrame, column: str, minimum: int
) -> list[dict[str, Any]]:
    output = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        if len(group) < minimum:
            continue
        negative = group.loc[group["truth"] == 0]
        positive = group.loc[group["truth"] == 1]
        output.append(
            {
                "group": _text(value),
                "n": int(len(group)),
                "n_positive": int(len(positive)),
                "false_positive_rate": (
                    float((negative["error_type"] == "FP").mean())
                    if len(negative)
                    else None
                ),
                "false_negative_rate": (
                    float((positive["error_type"] == "FN").mean())
                    if len(positive)
                    else None
                ),
            }
        )
    return output


def main() -> None:
    args = _parse_args()
    predictions = pd.read_csv(args.predictions)
    manifest = _load_validation_manifest(args.manifest_root)
    frame = predictions.merge(
        manifest, on=["study_id", "outer_fold"], how="left", validate="one_to_one"
    )
    if frame["n_slices"].isna().any():
        raise ValueError("Manifest coverage is incomplete")
    if not np.array_equal(
        frame["truth"].astype(int).to_numpy(),
        frame["manifest_truth"].astype(int).to_numpy(),
    ):
        raise ValueError("Prediction and manifest truths disagree")

    metadata = pd.DataFrame(
        [
            {"study_id": study_id, **_dicom_metadata(args.raw_root, str(study_id))}
            for study_id in frame["study_id"]
        ]
    )
    frame = frame.merge(metadata, on="study_id", how="left", validate="one_to_one")
    truth = frame["truth"].astype(int)
    predicted = frame["candidate_binary"].astype(int)
    frame["error_type"] = np.select(
        [
            (truth == 1) & (predicted == 1),
            (truth == 0) & (predicted == 0),
            (truth == 0) & (predicted == 1),
            (truth == 1) & (predicted == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNRESOLVED",
    )
    if (frame["error_type"] == "UNRESOLVED").any():
        raise ValueError("A prediction could not be assigned to a confusion category")
    frame["cdf_gap"] = frame["candidate_train_cdf"] - frame["reference_train_cdf"]
    frame["slice_count_quartile"] = pd.qcut(
        frame["n_slices"], q=4, duplicates="drop"
    ).astype(str)
    frame["detector_cdf_quartile"] = pd.qcut(
        frame["reference_train_cdf"], q=4, duplicates="drop"
    ).astype(str)
    frame["mil_cdf_quartile"] = pd.qcut(
        frame["candidate_train_cdf"], q=4, duplicates="drop"
    ).astype(str)
    frame["thickness_group"] = pd.cut(
        frame["slice_thickness"],
        bins=[-np.inf, 5.5, np.inf],
        labels=["thin_or_standard_<=5.5mm", "thick_>5.5mm"],
    ).astype(str)
    frame["positive_slice_group"] = pd.cut(
        frame["positive_slices"],
        bins=[-0.5, 0.5, 5.5, 10.5, np.inf],
        labels=["none", "1_to_5", "6_to_10", "more_than_10"],
    ).astype(str)

    by_error = {}
    for error_type in ["TP", "TN", "FP", "FN"]:
        group = frame.loc[frame["error_type"] == error_type]
        by_error[error_type] = {
            "n": int(len(group)),
            "numeric": _numeric_summary(group),
        }

    payload = {
        "protocol": "aggregate_oof_error_phenotype_no_study_identifiers",
        "n_studies": int(len(frame)),
        "n_patients": int(frame["patient_id"].nunique()),
        "metadata_available": int(frame["metadata_available"].fillna(False).sum()),
        "confusion": {
            key: int((frame["error_type"] == key).sum())
            for key in ["TP", "TN", "FP", "FN"]
        },
        "by_error_type": by_error,
        "error_rates": {
            column: _group_rates(frame, column, args.min_group_size)
            for column in [
                "outer_fold",
                "slice_count_quartile",
                "detector_cdf_quartile",
                "mil_cdf_quartile",
                "thickness_group",
                "positive_slice_group",
                "manufacturer",
                "kernel",
            ]
        },
        "privacy": {
            "study_identifiers_emitted": False,
            "minimum_reported_group_size": args.min_group_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
