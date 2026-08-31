"""Study-level ICH evaluation aligned with the competition decision rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import RAW_DIR, TRAINING_CSV_PATH
from src.evaluation.metrics import compute_competition_metrics
from src.evaluation.triage import triage_from_intermediates
from src.strategies.ich_v2.geometry import LABEL_TO_VOLUME_KEY
from src.strategies.ich_v2.supervision import ICH_AREA_COLUMNS


AREA_TO_VOLUME_KEY = dict(zip(ICH_AREA_COLUMNS, LABEL_TO_VOLUME_KEY.values()))
VOLUME_KEYS = tuple(LABEL_TO_VOLUME_KEY.values())


def load_slice_metadata(path: str | Path = TRAINING_CSV_PATH) -> tuple[pd.DataFrame, Path]:
    """Load slice metadata from CSV or its DVC-tracked pickle fallback."""
    requested = Path(path)
    raw_pickle = RAW_DIR / "training_df.pkl"
    selected = (
        raw_pickle
        if requested == TRAINING_CSV_PATH and raw_pickle.is_file()
        else requested
    )
    if not selected.is_file():
        selected = raw_pickle
    if not selected.is_file():
        raise FileNotFoundError(f"Competition metadata is unavailable: {path}")
    if selected.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(selected), selected
    return pd.read_csv(selected), selected


def ground_truth_ich_context(
    path: str | Path = TRAINING_CSV_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Reconstruct five volumes plus fixed MLS/fracture context per study."""
    frame, source = load_slice_metadata(path)
    required = {
        "dicom_series.id",
        "dicom_series.PixelSpacing0",
        "dicom_series.PixelSpacing1",
        "dicom_series.SliceThickness",
        "SkullFracture",
        "MidlineShiftMM",
        "triage_class",
        *ICH_AREA_COLUMNS,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Metadata is missing ICH evaluation columns: {sorted(missing)}")
    result = frame.copy()
    result["study_id"] = result["dicom_series.id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    factor = (
        pd.to_numeric(result["dicom_series.PixelSpacing0"], errors="coerce")
        * pd.to_numeric(result["dicom_series.PixelSpacing1"], errors="coerce")
        * pd.to_numeric(result["dicom_series.SliceThickness"], errors="coerce")
        / 1000.0
    )
    for area_column, key in AREA_TO_VOLUME_KEY.items():
        result[f"gt_{key}"] = (
            pd.to_numeric(result[area_column], errors="coerce").fillna(0.0) * factor
        )
    aggregation: dict[str, str] = {f"gt_{key}": "sum" for key in VOLUME_KEYS}
    aggregation.update({
        "SkullFracture": "max",
        "MidlineShiftMM": "max",
        "triage_class": "first",
    })
    studies = result.groupby("study_id", as_index=False).agg(aggregation)
    studies = studies.rename(columns={
        "SkullFracture": "gt_fracture_prob",
        "MidlineShiftMM": "gt_MLS_mm",
        "triage_class": "gt_triage_class",
    })
    return studies, source


def add_oracle_context_triage(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply predicted ICH volumes with ground-truth MLS/fracture context.

    This isolates the ICH component and is explicitly *not* an end-to-end OOF
    estimate.  Final promotion still requires joining genuine task OOF files.
    """
    required = {
        "gt_fracture_prob", "gt_MLS_mm", "gt_triage_class",
        *{f"pred_{key}" for key in VOLUME_KEYS},
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing oracle-context columns: {sorted(missing)}")
    result = predictions.copy()
    result["pred_triage_oracle_context"] = [
        triage_from_intermediates({
            **{key: float(row[f"pred_{key}"]) for key in VOLUME_KEYS},
            "fracture_prob": float(row["gt_fracture_prob"]),
            "MLS_mm": float(row["gt_MLS_mm"]),
        })
        for _, row in result.iterrows()
    ]
    return result


def summarize_ich_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute volume, presence and ICH-isolated triage diagnostics."""
    evaluated = add_oracle_context_triage(predictions)
    metrics = compute_competition_metrics(
        evaluated["gt_triage_class"],
        evaluated["pred_triage_oracle_context"],
        patient_ids=evaluated["patient_id"] if "patient_id" in evaluated else None,
        bootstrap_samples=0,
    )
    summary: dict[str, Any] = {
        "evaluation_context": "predicted_ich_with_ground_truth_mls_and_fracture",
        "studies": int(len(evaluated)),
        "oracle_context_macro_f1": float(metrics["macro_f1"]),
        "oracle_context_metrics": metrics,
        "subtypes": {},
    }
    for key in VOLUME_KEYS:
        truth = evaluated[f"gt_{key}"].to_numpy(dtype=float)
        predicted = evaluated[f"pred_{key}"].to_numpy(dtype=float)
        gt_present = truth > 0.0
        pred_present = predicted >= 0.1
        summary["subtypes"][key] = {
            "mae_ml": float(np.mean(np.abs(predicted - truth))),
            "bias_ml": float(np.mean(predicted - truth)),
            "presence_f1_at_0_1ml": float(
                f1_score(gt_present, pred_present, zero_division=0)
            ),
            "false_positive_studies": int(np.sum(~gt_present & pred_present)),
            "false_negative_studies": int(np.sum(gt_present & ~pred_present)),
        }
    gt_total = evaluated[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    pred_total = evaluated[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1)
    summary["total"] = {
        "mae_ml": float(np.mean(np.abs(pred_total - gt_total))),
        "bias_ml": float(np.mean(pred_total - gt_total)),
        "presence_f1_at_0_1ml": float(
            f1_score(gt_total > 0.0, pred_total >= 0.1, zero_division=0)
        ),
        "normal_false_positive_rate": float(
            np.mean(pred_total[gt_total <= 0.0] >= 0.1) if np.any(gt_total <= 0.0) else 0.0
        ),
    }
    return summary


def write_evaluation(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    evaluated = add_oracle_context_triage(predictions)
    csv_path = destination / "study_predictions.csv"
    json_path = destination / "summary.json"
    evaluated.to_csv(csv_path, index=False)
    summary = summarize_ich_predictions(evaluated)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path, summary
