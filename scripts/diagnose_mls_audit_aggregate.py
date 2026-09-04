"""Produce a privacy-preserving aggregate diagnostic for an MLS CUDA audit.

The input CSV contains per-study, per-slice predictions and is therefore kept on
the GPU server.  This utility emits only pooled counts and summary statistics;
it never writes identifiers, individual truth values, or slice predictions.
It is intentionally retrospective: it cannot select a checkpoint or alter a
pre-registered resource-screen decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.strategies.mls_heatmap.predict_multitask import (
    SliceMLSPrediction,
    aggregate_study_mls,
)


FIXED_POOLING = {
    "selector_threshold": 0.5,
    "top_k": 3,
    "aggregation": "p90",
    "min_active_slices": 1,
    "negative_value": 0.1,
}
TRUTH_BINS = (
    ("lt_1mm", -math.inf, 1.0),
    ("1_to_lt_3mm", 1.0, 3.0),
    ("3_to_lt_5mm", 3.0, 5.0),
    ("5_to_lt_10mm", 5.0, 10.0),
    ("gte_10mm", 10.0, math.inf),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}")
    return number


def _read_private_predictions(path: Path, expected_studies: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    truths: list[float] = []
    predictions: list[float] = []
    slice_counts: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"gt_MLS_mm", "slice_predictions_json", "error"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Private audit CSV has an incompatible schema")
        for row in reader:
            if str(row.get("error", "")).strip() not in {"", "nan", "None"}:
                raise ValueError("Private audit CSV contains an inference error")
            raw_slices = str(row.get("slice_predictions_json", "")).strip()
            try:
                slice_payload = json.loads(raw_slices)
            except json.JSONDecodeError as exc:
                raise ValueError("Unreadable private slice prediction payload") from exc
            if not isinstance(slice_payload, list) or not slice_payload:
                raise ValueError("Private audit CSV has an empty slice prediction payload")
            slices: list[SliceMLSPrediction] = []
            for item in slice_payload:
                if not isinstance(item, dict):
                    raise ValueError("Private audit CSV has an invalid slice item")
                slices.append(SliceMLSPrediction(
                    index=int(item["index"]),
                    selector_probability=_finite_number(item["selector_probability"], "selector_probability"),
                    mls_mm=_finite_number(item["mls_mm"], "mls_mm"),
                    heatmap_peak=_finite_number(item["heatmap_peak"], "heatmap_peak"),
                    peak_probability=(
                        None if item.get("peak_probability") is None
                        else _finite_number(item["peak_probability"], "peak_probability")
                    ),
                ))
            truths.append(_finite_number(row.get("gt_MLS_mm"), "gt_MLS_mm"))
            predictions.append(aggregate_study_mls(slices, **FIXED_POOLING))
            slice_counts.append(len(slices))
    if len(truths) != expected_studies:
        raise ValueError(
            f"Expected exactly {expected_studies} private studies, found {len(truths)}"
        )
    return np.asarray(truths, dtype=float), np.asarray(predictions, dtype=float), slice_counts


def _residual_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - truth
    return {
        "mae_mm": float(np.mean(np.abs(residual))),
        "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "mean_signed_error_mm": float(np.mean(residual)),
        "median_signed_error_mm": float(np.median(residual)),
        "within_1mm_fraction": float(np.mean(np.abs(residual) <= 1.0)),
        "overestimate_gt_1mm_fraction": float(np.mean(residual > 1.0)),
        "underestimate_lt_minus_1mm_fraction": float(np.mean(residual < -1.0)),
    }


def _truth_strata(truth: np.ndarray, prediction: np.ndarray) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []
    for name, low, high in TRUTH_BINS:
        mask = (truth >= low) & (truth < high)
        if not mask.any():
            output.append({"truth_bin": name, "n_studies": 0})
            continue
        residual = prediction[mask] - truth[mask]
        output.append({
            "truth_bin": name,
            "n_studies": int(mask.sum()),
            "mae_mm": float(np.mean(np.abs(residual))),
            "mean_signed_error_mm": float(np.mean(residual)),
            "within_1mm_fraction": float(np.mean(np.abs(residual) <= 1.0)),
        })
    return output


def _threshold_summary(truth: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual = truth >= threshold
    predicted = prediction >= threshold
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    return {
        "threshold_mm": threshold,
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def diagnose(input_csv: Path, output_json: Path, *, expected_studies: int = 70) -> dict:
    """Create an aggregate-only diagnostic without exposing private rows."""
    source = input_csv.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    truth, prediction, slice_counts = _read_private_predictions(source, expected_studies)
    result = {
        "schema_version": 1,
        "scope": "retrospective_aggregate_diagnostic_only",
        "privacy": {
            "contains_study_ids": False,
            "contains_individual_truth_or_prediction_values": False,
            "mlflow_logging_performed": False,
        },
        "private_input_sha256": _sha256(source),
        "n_studies": int(len(truth)),
        "fixed_pooling": FIXED_POOLING,
        "input_slice_count": {
            "total": int(sum(slice_counts)),
            "min_per_study": int(min(slice_counts)),
            "max_per_study": int(max(slice_counts)),
        },
        "residual_summary": _residual_summary(truth, prediction),
        "truth_strata": _truth_strata(truth, prediction),
        "thresholds": [_threshold_summary(truth, prediction, threshold) for threshold in (3.0, 5.0)],
        "interpretation_guard": (
            "Descriptive result for one rejected checkpoint only; it is not a new "
            "checkpoint screen, pooling search, promotion decision, or submission evidence."
        ),
    }
    _atomic_json(output_json.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-studies", type=int, default=70)
    args = parser.parse_args()
    result = diagnose(args.input_csv, args.output_json, expected_studies=args.expected_studies)
    print(json.dumps({
        "scope": result["scope"],
        "n_studies": result["n_studies"],
        "residual_summary": result["residual_summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
