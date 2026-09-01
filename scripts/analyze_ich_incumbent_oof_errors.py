"""Audit patient-disjoint OOF errors of the standalone ICH incumbent.

This script is diagnostic only.  Its fixed score-gating grid must not be used
as a deployable threshold search; a promising rule has to be selected on each
fold's calibration data and then evaluated once on that fold's outer data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scripts.compare_ich_2p5d_segmentation_oof import VariantResult, _load_variant
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    VOLUME_TO_LABEL,
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


SPATIAL_PREDICTION_PREFIXES = (
    "pred_pixels",
    "intersection",
    "predicted_known_pixels",
)
LESION_STRATA = (
    ("small_le_2ml", 0.0, 2.0),
    ("medium_2_to_10ml", 2.0, 10.0),
    ("large_gt_10ml", 10.0, float("inf")),
)


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _presence_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fn = int(np.sum(truth & ~predicted))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "f1": float(f1_score(truth, predicted, zero_division=0)),
    }


def _max_consecutive(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _build_study_error_table(variant: VariantResult) -> pd.DataFrame:
    studies = variant.studies.copy()
    studies["study_id"] = studies["study_id"].astype(str)
    studies["gt_total_ml"] = studies[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    studies["pred_total_ml"] = studies[
        [f"pred_{key}" for key in VOLUME_KEYS]
    ].sum(axis=1)
    studies["gt_any"] = studies["gt_total_ml"] > 0.0
    studies["pred_any_0_1ml"] = studies["pred_total_ml"] >= 0.1
    studies["presence_error"] = np.select(
        [
            studies["gt_any"] & studies["pred_any_0_1ml"],
            ~studies["gt_any"] & studies["pred_any_0_1ml"],
            studies["gt_any"] & ~studies["pred_any_0_1ml"],
        ],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    studies["volume_error_ml"] = studies["pred_total_ml"] - studies["gt_total_ml"]
    studies["absolute_volume_error_ml"] = studies["volume_error_ml"].abs()

    prediction_columns = [f"pred_{key}" for key in VOLUME_KEYS]
    predicted_values = studies[prediction_columns].to_numpy(dtype=float)
    dominant = np.argmax(predicted_values, axis=1)
    studies["dominant_predicted_subtype"] = [VOLUME_KEYS[index] for index in dominant]
    studies.loc[studies["pred_total_ml"] <= 0.0, "dominant_predicted_subtype"] = "none"

    slices = variant.slices.sort_values(["study_id", "slice_index"]).copy()
    slices["study_id"] = slices["study_id"].astype(str)
    pixel_columns = [f"pred_pixels_{label}" for label in OUTPUT_LABELS[1:]]
    slices["predicted_any_slice"] = slices[pixel_columns].sum(axis=1) > 0
    slice_rows: list[dict[str, Any]] = []
    for study_id, frame in slices.groupby("study_id", sort=False):
        predicted = frame["predicted_any_slice"].to_numpy(bool)
        slice_rows.append({
            "study_id": str(study_id),
            "predicted_positive_slices": int(np.sum(predicted)),
            "max_consecutive_predicted_slices": _max_consecutive(predicted),
            "total_slices": int(len(frame)),
            "outer_fold": int(frame["outer_fold"].iloc[0]),
            "patient_id": str(frame["patient_id"].iloc[0]),
        })
    return studies.merge(
        pd.DataFrame(slice_rows), on="study_id", how="inner", validate="one_to_one"
    ).sort_values(["presence_error", "absolute_volume_error_ml"], ascending=[True, False])


def _zero_spatial_predictions(
    slices: pd.DataFrame,
    studies: pd.DataFrame,
    *,
    any_threshold: float | None = None,
    subtype_threshold: float | None = None,
) -> pd.DataFrame:
    gated = slices.copy()
    gated["study_id"] = gated["study_id"].astype(str)
    indexed = studies.assign(study_id=studies["study_id"].astype(str)).set_index("study_id")
    if any_threshold is not None:
        keep_any = indexed["score_any_ich"] >= any_threshold
        keep = gated["study_id"].map(keep_any).fillna(False).to_numpy(bool)
        for label in OUTPUT_LABELS[1:]:
            for prefix in SPATIAL_PREDICTION_PREFIXES:
                gated.loc[~keep, f"{prefix}_{label}"] = 0
    if subtype_threshold is not None:
        for label in OUTPUT_LABELS[1:]:
            keep_label = indexed[f"score_{label}"] >= subtype_threshold
            keep = gated["study_id"].map(keep_label).fillna(False).to_numpy(bool)
            for prefix in SPATIAL_PREDICTION_PREFIXES:
                gated.loc[~keep, f"{prefix}_{label}"] = 0
    return gated


def _gate_grid(
    variant: VariantResult,
    truth: pd.DataFrame,
    thresholds: list[float],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "any_score_study_gate": [],
        "per_subtype_score_gate": [],
        "combined_any_and_subtype_gate": [],
    }
    modes = (
        ("any_score_study_gate", True, False),
        ("per_subtype_score_gate", False, True),
        ("combined_any_and_subtype_gate", True, True),
    )
    for name, use_any, use_subtype in modes:
        for threshold in thresholds:
            gated = _zero_spatial_predictions(
                variant.slices,
                variant.studies,
                any_threshold=threshold if use_any else None,
                subtype_threshold=threshold if use_subtype else None,
            )
            _, summary = summarize_segmentation_predictions(gated, truth)
            result[name].append({"threshold": threshold, **summary})
    return result


def _error_summary(table: pd.DataFrame) -> dict[str, Any]:
    gt_any = table["gt_any"].to_numpy(bool)
    pred_any = table["pred_any_0_1ml"].to_numpy(bool)
    false_positive = table[table["presence_error"] == "false_positive"]
    false_negative = table[table["presence_error"] == "false_negative"]
    normal = table[~table["gt_any"]]

    dominant_counts = {
        str(key): int(value)
        for key, value in false_positive["dominant_predicted_subtype"]
        .value_counts()
        .items()
    }
    fold_presence = []
    for outer_fold, frame in table.groupby("outer_fold", sort=True):
        fold_presence.append({
            "outer_fold": int(outer_fold),
            **_presence_metrics(
                frame["gt_any"].to_numpy(bool),
                frame["pred_any_0_1ml"].to_numpy(bool),
            ),
        })

    lesion_strata = []
    positives = table[table["gt_any"]]
    for name, lower, upper in LESION_STRATA:
        selected = positives[
            (positives["gt_total_ml"] > lower) & (positives["gt_total_ml"] <= upper)
        ]
        lesion_strata.append({
            "stratum": name,
            "studies": int(len(selected)),
            "presence_sensitivity": (
                float(selected["pred_any_0_1ml"].mean()) if len(selected) else None
            ),
            "absolute_volume_error_ml": _quantiles(
                selected["absolute_volume_error_ml"].to_numpy(float)
            ),
            "signed_volume_bias_ml": (
                float(selected["volume_error_ml"].mean()) if len(selected) else None
            ),
        })

    return {
        "presence_at_0_1ml": _presence_metrics(gt_any, pred_any),
        "fold_presence_at_0_1ml": fold_presence,
        "normal_studies": int(len(normal)),
        "false_positive": {
            "studies": int(len(false_positive)),
            "predicted_total_volume_ml": _quantiles(false_positive["pred_total_ml"]),
            "any_score": _quantiles(false_positive["score_any_ich"]),
            "predicted_positive_slices": _quantiles(
                false_positive["predicted_positive_slices"]
            ),
            "max_consecutive_predicted_slices": _quantiles(
                false_positive["max_consecutive_predicted_slices"]
            ),
            "dominant_predicted_subtype": dominant_counts,
            "tiny_and_short_artifact_count": int(np.sum(
                (false_positive["pred_total_ml"] < 0.5)
                & (false_positive["max_consecutive_predicted_slices"] <= 2)
            )),
        },
        "false_negative": {
            "studies": int(len(false_negative)),
            "ground_truth_total_volume_ml": _quantiles(false_negative["gt_total_ml"]),
            "any_score": _quantiles(false_negative["score_any_ich"]),
        },
        "lesion_size_strata": lesion_strata,
        "score_any_by_presence_error": {
            str(name): _quantiles(frame["score_any_ich"])
            for name, frame in table.groupby("presence_error", sort=True)
        },
    }


def _subtype_summary(variant: VariantResult) -> list[dict[str, Any]]:
    frame = variant.sufficient
    rows: list[dict[str, Any]] = []
    for volume_key, label in VOLUME_TO_LABEL.items():
        truth = frame[f"gt_{volume_key}"].to_numpy(float) > 0
        predicted = frame[f"pred_{volume_key}"].to_numpy(float) >= 0.1
        intersection = float(frame[f"intersection_{label}"].sum())
        predicted_pixels = float(frame[f"predicted_known_pixels_{label}"].sum())
        observed_pixels = float(frame[f"observed_known_pixels_{label}"].sum())
        rows.append({
            "label": label,
            "positive_studies": int(np.sum(truth)),
            "pooled_dice": (
                2.0 * intersection / max(1.0, predicted_pixels + observed_pixels)
            ),
            "presence_at_0_1ml": _presence_metrics(truth, predicted),
            "predicted_volume_on_negative_studies_ml": _quantiles(
                frame.loc[~truth, f"pred_{volume_key}"]
            ),
            "score_positive_studies": _quantiles(frame.loc[truth, f"score_{label}"]),
            "score_negative_studies": _quantiles(frame.loc[~truth, f"score_{label}"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument(
        "--diagnostic-score-threshold",
        action="append",
        type=float,
        dest="thresholds",
    )
    args = parser.parse_args()
    thresholds = args.thresholds or [0.25, 0.5, 0.75, 0.9]
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Diagnostic score thresholds must be in [0, 1]")

    truth, metadata_source = ground_truth_ich_context()
    variant = _load_variant(
        "hardpixel_fprselect_incumbent",
        args.run_dir,
        truth,
        args.expected_studies,
    )
    table = _build_study_error_table(variant)
    payload = {
        "evaluation_scope": "standalone_ich_patient_disjoint_oof",
        "analysis_kind": "diagnostic_only_not_threshold_selection",
        "metadata_source": str(metadata_source),
        "studies": int(len(table)),
        "patients": int(table["patient_id"].nunique()),
        "outer_folds": sorted(int(value) for value in table["outer_fold"].unique()),
        "incumbent_summary": variant.summary,
        "error_analysis": _error_summary(table),
        "subtypes": _subtype_summary(variant),
        "fixed_score_gate_grid": _gate_grid(variant, truth, thresholds),
        "protocol_warning": (
            "The fixed grid is exploratory OOF diagnostics. Any gate must be chosen "
            "independently on each run's calibration fold before one-time outer use."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "study_error_table.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
