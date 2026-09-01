"""Audit through-plane ICH continuity before changing 2.5D context width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_v2.operations import file_sha256


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "q10": None, "q25": None,
                "median": None, "q75": None, "q90": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def contiguous_runs(binary: np.ndarray) -> list[int]:
    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (stops - starts).astype(int).tolist()


def analyze_context_manifest(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "study_id", "slice_index", "slice_count", "slice_spacing_mm",
        "slice_thickness_mm", "classification_known", *OUTPUT_LABELS[1:],
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"ICH context manifest is missing: {sorted(missing)}")
    data = frame.copy()
    numeric = required - {"study_id"}
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    data = data.sort_values(["study_id", "slice_index"]).reset_index(drop=True)
    groups = list(data.groupby("study_id", sort=False))
    study_rows = data.drop_duplicates("study_id")
    for name in ("slice_spacing_mm", "slice_thickness_mm", "slice_count"):
        if (data.groupby("study_id")[name].nunique() != 1).any():
            raise ValueError(f"{name} must be constant within every study")

    spacing = study_rows["slice_spacing_mm"].to_numpy(dtype=np.float64)
    slice_counts = study_rows["slice_count"].to_numpy(dtype=np.float64)
    context_geometry: dict[str, object] = {}
    for radius in (1, 2):
        duplicated_positions = sum(
            min(radius, offset) + min(radius, count - 1 - offset)
            for count in study_rows["slice_count"].astype(int)
            for offset in range(count)
        )
        # The expression above counts available neighbours. Convert it to the
        # number of edge-padded positions among all non-centre context slots.
        total_neighbour_positions = int(data.shape[0] * 2 * radius)
        edge_padded = total_neighbour_positions - int(duplicated_positions)
        context_geometry[f"radius_{radius}"] = {
            "physical_radius_mm": numeric_summary(spacing * radius),
            "full_context_span_mm": numeric_summary(spacing * radius * 2),
            "edge_padded_positions": edge_padded,
            "edge_padded_fraction_of_neighbours": (
                float(edge_padded / total_neighbour_positions)
                if total_neighbour_positions else 0.0
            ),
        }

    subtype_results: dict[str, object] = {}
    for label in OUTPUT_LABELS[1:]:
        run_lengths: list[int] = []
        run_spans_mm: list[float] = []
        positive_slices = 0
        adjacent_positive = 0
        distance_two_positive = 0
        isolated_positive = 0
        positive_studies = 0
        for _, group in groups:
            known = group["classification_known"].to_numpy(dtype=bool)
            positive = known & (group[label].to_numpy(dtype=np.float64) > 0)
            if positive.any():
                positive_studies += 1
            spacing_mm = float(group["slice_spacing_mm"].iloc[0])
            runs = contiguous_runs(positive)
            run_lengths.extend(runs)
            run_spans_mm.extend(length * spacing_mm for length in runs)
            positive_indices = np.flatnonzero(positive)
            positive_slices += int(len(positive_indices))
            for index in positive_indices:
                at_one = (
                    (index > 0 and positive[index - 1])
                    or (index + 1 < len(positive) and positive[index + 1])
                )
                at_two = (
                    (index > 1 and positive[index - 2])
                    or (index + 2 < len(positive) and positive[index + 2])
                )
                adjacent_positive += int(at_one)
                distance_two_positive += int(at_two)
                isolated_positive += int(not at_one)
        run_array = np.asarray(run_lengths, dtype=np.int64)
        denominator = max(1, positive_slices)
        run_denominator = max(1, len(run_lengths))
        subtype_results[label] = {
            "positive_studies": positive_studies,
            "positive_slices": positive_slices,
            "positive_slice_with_adjacent_positive_fraction": float(
                adjacent_positive / denominator
            ),
            "positive_slice_with_distance_two_positive_fraction": float(
                distance_two_positive / denominator
            ),
            "isolated_positive_slice_fraction": float(
                isolated_positive / denominator
            ),
            "runs": len(run_lengths),
            "single_slice_run_fraction": float(
                np.count_nonzero(run_array == 1) / run_denominator
            ),
            "run_le_two_slices_fraction": float(
                np.count_nonzero(run_array <= 2) / run_denominator
            ),
            "run_length_slices": numeric_summary(run_lengths),
            "run_span_mm": numeric_summary(run_spans_mm),
        }

    return {
        "analysis_kind": "ich_through_plane_context_audit",
        "studies": int(data["study_id"].nunique()),
        "slices": int(len(data)),
        "slice_count_per_study": numeric_summary(slice_counts),
        "slice_spacing_mm": numeric_summary(spacing),
        "slice_thickness_mm": numeric_summary(
            study_rows["slice_thickness_mm"].to_numpy(dtype=np.float64)
        ),
        "context_geometry": context_geometry,
        "subtypes": subtype_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest_path, dtype={"study_id": str})
    result = analyze_context_manifest(frame)
    result["manifest_path"] = str(args.manifest_path)
    result["manifest_sha256"] = file_sha256(args.manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
