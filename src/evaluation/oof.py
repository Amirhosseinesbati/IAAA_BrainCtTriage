"""Strict assembly contract for the three task-level OOF prediction files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import TRAINING_CSV_PATH
from src.evaluation.calibration import INTERMEDIATE_KEYS
from src.evaluation.splits import load_fold_manifest, normalize_study_id
from src.evaluation.triage import triage_from_intermediates

ICH_AREA_COLUMNS = {
    "V_EDH": "EpiduralHemorrhage_Area",
    "V_SDH": "SubduralHemorrhage_Area",
    "V_IPH": "IntraparenchymalHemorrhage_Area",
    "V_SAH": "SubarachnoidHemorrhage_Area",
    "V_IVH": "IntraventricularHemorrhage_Area",
}


def ground_truth_intermediates(
    metadata_path: str | Path = TRAINING_CSV_PATH,
) -> pd.DataFrame:
    """Reconstruct the seven official study-level ground-truth quantities."""
    frame = pd.read_csv(metadata_path)
    required = {
        "dicom_series.id", "dicom_series.PixelSpacing0",
        "dicom_series.PixelSpacing1", "dicom_series.SliceThickness",
        "SkullFracture", "MidlineShiftMM", *ICH_AREA_COLUMNS.values(),
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Metadata is missing ground-truth columns: {sorted(missing)}")
    factor = (
        pd.to_numeric(frame["dicom_series.PixelSpacing0"], errors="coerce")
        * pd.to_numeric(frame["dicom_series.PixelSpacing1"], errors="coerce")
        * pd.to_numeric(frame["dicom_series.SliceThickness"], errors="coerce")
        / 1000.0
    )
    for key, area_column in ICH_AREA_COLUMNS.items():
        frame[f"gt_{key}"] = pd.to_numeric(frame[area_column], errors="coerce").fillna(0) * factor
    aggregation: dict[str, str] = {f"gt_{key}": "sum" for key in ICH_AREA_COLUMNS}
    aggregation.update({"SkullFracture": "max", "MidlineShiftMM": "max"})
    truth = frame.groupby("dicom_series.id", as_index=False).agg(aggregation)
    truth = truth.rename(columns={"dicom_series.id": "study_id"})
    truth["study_id"] = truth["study_id"].map(normalize_study_id)
    truth["gt_fracture_prob"] = truth.pop("SkullFracture").astype(float)
    truth["gt_MLS_mm"] = pd.to_numeric(truth.pop("MidlineShiftMM"), errors="coerce")
    return truth


def _normalize_task_predictions(
    frame: pd.DataFrame,
    *,
    task: str,
    prediction_columns: list[str],
    expected: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    if "study_id" not in result and "series_id" in result:
        result = result.rename(columns={"series_id": "study_id"})
    required = {"study_id", *prediction_columns}
    missing = required - set(result)
    if missing:
        raise ValueError(f"{task} OOF is missing columns: {sorted(missing)}")
    result["study_id"] = result["study_id"].map(normalize_study_id)
    if result["study_id"].duplicated().any():
        duplicate = result.loc[result["study_id"].duplicated(), "study_id"].tolist()
        raise ValueError(f"{task} OOF has duplicate studies: {duplicate[:10]}")

    expected_ids = set(expected["study_id"])
    actual_ids = set(result["study_id"])
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{task} OOF coverage mismatch: missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )
    result = expected[["study_id", "fold"]].merge(result, on="study_id", how="left", suffixes=("_expected", ""))
    if "fold" in frame:
        reported = pd.to_numeric(result["fold"], errors="coerce")
        expected_fold = pd.to_numeric(result["fold_expected"], errors="coerce")
        if not reported.equals(expected_fold):
            bad = result.loc[reported != expected_fold, "study_id"].tolist()
            raise ValueError(f"{task} OOF reports incorrect folds for studies: {bad[:10]}")
        result = result.drop(columns="fold")
    result = result.rename(columns={"fold_expected": "fold"})
    numeric = result[prediction_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{task} OOF contains non-finite predictions")
    result[prediction_columns] = numeric
    return result


def assemble_oof_predictions(
    ich: pd.DataFrame,
    fracture: pd.DataFrame,
    mls: pd.DataFrame,
    *,
    fold_manifest_path: str | Path | None = None,
    metadata_path: str | Path = TRAINING_CSV_PATH,
) -> pd.DataFrame:
    """Join task OOF outputs into the only CSV accepted by calibration."""
    manifest = load_fold_manifest(fold_manifest_path)[
        ["study_id", "patient_id", "triage_class", "fold"]
    ].copy()
    groups = {
        "ich": [f"pred_{key}" for key in ICH_AREA_COLUMNS],
        "fracture": ["pred_fracture_prob"],
        "mls": ["pred_MLS_mm"],
    }
    task_frames = {
        "ich": _normalize_task_predictions(
            ich, task="ich", prediction_columns=groups["ich"], expected=manifest,
        ),
        "fracture": _normalize_task_predictions(
            fracture, task="fracture", prediction_columns=groups["fracture"], expected=manifest,
        ),
        "mls": _normalize_task_predictions(
            mls, task="mls", prediction_columns=groups["mls"], expected=manifest,
        ),
    }
    assembled = manifest.copy()
    for task, task_frame in task_frames.items():
        assembled = assembled.merge(
            task_frame.drop(columns="fold"), on="study_id", how="left", validate="one_to_one",
        )
    assembled = assembled.merge(
        ground_truth_intermediates(metadata_path), on="study_id", how="left", validate="one_to_one",
    )
    required_numeric = [
        *(f"pred_{key}" for key in INTERMEDIATE_KEYS),
        *(f"gt_{key}" for key in INTERMEDIATE_KEYS),
    ]
    if not np.isfinite(assembled[required_numeric].to_numpy(dtype=float)).all():
        raise ValueError("Assembled OOF contains missing or non-finite truth/prediction values")
    assembled["pred_triage"] = [
        triage_from_intermediates({key: row[f"pred_{key}"] for key in INTERMEDIATE_KEYS})
        for _, row in assembled.iterrows()
    ]
    return assembled.sort_values("study_id").reset_index(drop=True)
