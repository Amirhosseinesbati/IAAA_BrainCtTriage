"""Blend saved full-series MLS predictions without running model inference.

The inputs are ``study_slice_predictions.csv`` artifacts produced by
``evaluate_mls_multitask_checkpoint.py``.  Study identity, ground truth, and
slice indices are checked before selector, MLS, and heatmap outputs are
combined.  This keeps snapshot-ensemble experiments cheap and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_KEYS = ("selector_probability", "mls_mm", "heatmap_peak")


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=CSV_PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("input label cannot be empty")
    return label.strip(), Path(raw_path)


def _named_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("weight must be LABEL=VALUE")
    label, raw_weight = value.split("=", 1)
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be numeric") from exc
    if not np.isfinite(weight) or weight <= 0:
        raise argparse.ArgumentTypeError("weight must be finite and positive")
    return label.strip(), weight


def _clean_error(value: object) -> str:
    text = str(value).strip()
    return "" if text in {"", "nan", "None"} else text


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
    required = {
        "study_id",
        "patient_id",
        "triage_class",
        "gt_MLS_mm",
        "slice_predictions_json",
        "runtime_s",
        "error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    errors = frame["error"].map(_clean_error)
    if (errors != "").any():
        bad = frame.loc[errors != "", ["study_id", "error"]].head(3)
        raise RuntimeError(f"Evaluation errors in {path}: {bad.to_dict('records')}")
    if frame["study_id"].duplicated().any():
        raise ValueError(f"Duplicate study_id values in {path}")
    return frame.sort_values("study_id").reset_index(drop=True)


def _validate_frames(frames: dict[str, pd.DataFrame]) -> list[str]:
    labels = list(frames)
    reference = frames[labels[0]]
    study_ids = reference["study_id"].tolist()
    for label in labels[1:]:
        frame = frames[label]
        if frame["study_id"].tolist() != study_ids:
            raise ValueError(f"Study IDs/order differ for {label}")
        if frame["patient_id"].tolist() != reference["patient_id"].tolist():
            raise ValueError(f"Patient IDs differ for {label}")
        if not np.allclose(
            frame["gt_MLS_mm"].to_numpy(float),
            reference["gt_MLS_mm"].to_numpy(float),
            atol=1e-6,
        ):
            raise ValueError(f"Ground truth differs for {label}")
        if frame["triage_class"].tolist() != reference["triage_class"].tolist():
            raise ValueError(f"Triage labels differ for {label}")
    return study_ids


def _blend_study(
    payloads: list[list[dict]],
    mode: str,
    weights: np.ndarray,
) -> list[dict]:
    reference_indices = [int(item["index"]) for item in payloads[0]]
    for payload in payloads[1:]:
        indices = [int(item["index"]) for item in payload]
        if indices != reference_indices:
            raise ValueError("Slice indices differ between ensemble members")

    blended: list[dict] = []
    for item_index, slice_index in enumerate(reference_indices):
        result: dict[str, float | int] = {"index": slice_index}
        for key in PREDICTION_KEYS:
            values = np.asarray(
                [float(payload[item_index][key]) for payload in payloads],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {key} at slice {slice_index}")
            if mode == "mean":
                result[key] = float(np.average(values, weights=weights))
            else:
                result[key] = float(np.median(values))
        blended.append(result)
    return blended


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=_named_path,
        required=True,
        help="Repeat LABEL=study_slice_predictions.csv for each member.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=_named_weight,
        default=[],
        help="Optional positive LABEL=VALUE weight; unspecified labels use 1.",
    )
    parser.add_argument("--mode", choices=("mean", "median"), default="mean")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if len(args.input) < 2:
        raise ValueError("At least two inputs are required")
    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        raise ValueError("Input labels must be unique")
    supplied_weights = dict(args.weight)
    unknown_weights = sorted(set(supplied_weights) - set(labels))
    if unknown_weights:
        raise ValueError(f"Weights supplied for unknown labels: {unknown_weights}")
    if args.mode == "median" and args.weight:
        raise ValueError("Weights are only supported for mean blending")

    frames = {label: _load(path) for label, path in args.input}
    study_ids = _validate_frames(frames)
    weights = np.asarray([supplied_weights.get(label, 1.0) for label in labels])
    reference = frames[labels[0]]
    output_rows: list[dict] = []
    for row_index, study_id in enumerate(study_ids):
        payloads = [
            json.loads(frames[label].iloc[row_index]["slice_predictions_json"])
            for label in labels
        ]
        blended = _blend_study(payloads, args.mode, weights)
        output_rows.append({
            "study_id": study_id,
            "patient_id": str(reference.iloc[row_index]["patient_id"]),
            "triage_class": int(reference.iloc[row_index]["triage_class"]),
            "gt_MLS_mm": float(reference.iloc[row_index]["gt_MLS_mm"]),
            "slice_predictions_json": json.dumps(blended),
            "runtime_s": float(sum(
                frames[label].iloc[row_index]["runtime_s"] for label in labels
            )),
            "error": "",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    pd.DataFrame(output_rows).to_csv(temporary, index=False)
    temporary.replace(args.output_csv)
    print(json.dumps({
        "output_csv": str(args.output_csv),
        "mode": args.mode,
        "members": labels,
        "normalized_weights": {
            label: float(weight / weights.sum())
            for label, weight in zip(labels, weights)
        },
        "n_studies": len(output_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
