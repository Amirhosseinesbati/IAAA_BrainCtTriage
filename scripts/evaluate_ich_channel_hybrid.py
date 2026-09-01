"""Evaluate a preregistered channel-wise ICH hybrid on calibration predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


IDENTITY_COLUMNS = (
    "study_id",
    "patient_id",
    "slice_index",
    "known",
    "voxel_volume_ml",
)
CHANNEL_PREFIXES = (
    "prob",
    "pred_pixels",
    "intersection",
    "predicted_known_pixels",
    "observed_known_pixels",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def build_channel_hybrid(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_labels: tuple[str, ...] = ("IVH",),
    any_source: str = "candidate",
) -> pd.DataFrame:
    """Use selected reference channels and candidate channels everywhere else."""
    valid_labels = set(OUTPUT_LABELS[1:])
    unknown = sorted(set(reference_labels) - valid_labels)
    if unknown:
        raise ValueError(f"Unknown reference labels: {unknown}")
    if any_source not in {"reference", "candidate"}:
        raise ValueError("any_source must be 'reference' or 'candidate'")

    required = {
        *IDENTITY_COLUMNS,
        "prob_any_ich",
        *{
            f"{prefix}_{label}"
            for label in OUTPUT_LABELS[1:]
            for prefix in CHANNEL_PREFIXES
        },
    }
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")

    sort_columns = ["study_id", "slice_index"]
    reference_sorted = reference.sort_values(sort_columns).reset_index(drop=True).copy()
    candidate_sorted = candidate.sort_values(sort_columns).reset_index(drop=True).copy()
    if len(reference_sorted) != len(candidate_sorted):
        raise ValueError("Reference and candidate prediction row counts differ")
    for column in IDENTITY_COLUMNS:
        left = reference_sorted[column]
        right = candidate_sorted[column]
        if column == "voxel_volume_ml":
            equal = np.allclose(
                left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-12
            )
        else:
            equal = left.astype(str).equals(right.astype(str))
        if not equal:
            raise ValueError(f"Reference and candidate differ in {column}")

    hybrid = candidate_sorted.copy()
    if any_source == "reference":
        hybrid["prob_any_ich"] = reference_sorted["prob_any_ich"]
    for label in reference_labels:
        for prefix in CHANNEL_PREFIXES:
            column = f"{prefix}_{label}"
            hybrid[column] = reference_sorted[column]
    return hybrid


def evaluate_channel_hybrid(
    reference_predictions: Path,
    candidate_predictions: Path,
    *,
    reference_run_summary: Path,
    candidate_run_summary: Path,
    output_dir: Path,
    reference_labels: tuple[str, ...] = ("IVH",),
    any_source: str = "candidate",
    evaluation_split: str = "calibration_only_no_outer",
) -> dict[str, Any]:
    if evaluation_split not in {"calibration_only_no_outer", "outer_fold"}:
        raise ValueError(
            "evaluation_split must be 'calibration_only_no_outer' or 'outer_fold'"
        )
    reference_run = _read_json(reference_run_summary)
    candidate_run = _read_json(candidate_run_summary)
    if reference_run.get("manifest_sha256") != candidate_run.get("manifest_sha256"):
        raise ValueError("Reference and candidate use different manifests")

    dtype = {"study_id": str, "patient_id": str}
    reference = pd.read_csv(reference_predictions, dtype=dtype)
    candidate = pd.read_csv(candidate_predictions, dtype=dtype)
    hybrid = build_channel_hybrid(
        reference,
        candidate,
        reference_labels=reference_labels,
        any_source=any_source,
    )
    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    studies, summary = summarize_segmentation_predictions(hybrid, truth)

    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(output_dir / "hybrid_slice_predictions.csv", index=False)
    studies.to_csv(output_dir / "hybrid_study_predictions.csv", index=False)
    payload = {
        "schema_version": 1,
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "evaluation_split": evaluation_split,
        "manifest_sha256": reference_run["manifest_sha256"],
        "metadata_source": str(metadata_source),
        "reference_labels": list(reference_labels),
        "candidate_labels": [
            label for label in OUTPUT_LABELS[1:] if label not in reference_labels
        ],
        "any_ich_source": any_source,
        "reference": {
            "run_id": reference_run.get("run_id"),
            "checkpoint_sha256": reference_run.get("checkpoint_sha256"),
            "predictions": str(reference_predictions),
        },
        "candidate": {
            "run_id": candidate_run.get("run_id"),
            "checkpoint_sha256": candidate_run.get("checkpoint_sha256"),
            "predictions": str(candidate_predictions),
        },
        "summary": summary,
    }
    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--reference-run-summary", required=True, type=Path)
    parser.add_argument("--candidate-run-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-label",
        action="append",
        dest="reference_labels",
        choices=OUTPUT_LABELS[1:],
        default=None,
    )
    parser.add_argument(
        "--any-source", choices=("reference", "candidate"), default="candidate"
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("calibration_only_no_outer", "outer_fold"),
        default="calibration_only_no_outer",
    )
    args = parser.parse_args()
    result = evaluate_channel_hybrid(
        args.reference_predictions,
        args.candidate_predictions,
        reference_run_summary=args.reference_run_summary,
        candidate_run_summary=args.candidate_run_summary,
        output_dir=args.output_dir,
        reference_labels=tuple(args.reference_labels or ("IVH",)),
        any_source=args.any_source,
        evaluation_split=args.evaluation_split,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
