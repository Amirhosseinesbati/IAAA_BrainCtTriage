"""Patient-grouped, class-stratified folds for leakage-free OOF evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_SEED, TRAINING_CSV_PATH, config_section


def aggregate_studies(csv_path: str | Path = TRAINING_CSV_PATH) -> pd.DataFrame:
    slices = pd.read_csv(csv_path)
    required = {
        "dicom_series.id", "dicom_series.PatientID", "triage_class",
        "AnyICH", "SkullFracture", "MidlineShiftMM",
    }
    missing = required - set(slices.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")

    studies = slices.groupby("dicom_series.id", as_index=False).agg({
        "dicom_series.PatientID": "first",
        "triage_class": "first",
        "AnyICH": "max",
        "SkullFracture": "max",
        "MidlineShiftMM": "max",
        "dicom_series.NumDicomFiles": "first",
    })
    studies = studies.rename(columns={
        "dicom_series.id": "study_id",
        "dicom_series.PatientID": "patient_id",
    })
    studies["study_id"] = studies["study_id"].astype(str)
    studies["patient_id"] = studies["patient_id"].astype(str)
    return studies.sort_values("study_id").reset_index(drop=True)


def create_fold_manifest(
    studies: pd.DataFrame,
    *,
    n_folds: int | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    n_splits = n_folds or int(config_section("competition", "evaluation", "n_folds"))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    manifest = studies.copy()
    manifest["fold"] = -1
    for fold, (_, val_index) in enumerate(splitter.split(manifest, manifest["triage_class"], manifest["patient_id"])):
        manifest.loc[val_index, "fold"] = fold
    validate_fold_manifest(manifest, n_splits)
    return manifest


def validate_fold_manifest(manifest: pd.DataFrame, n_folds: int | None = None) -> None:
    required = {"study_id", "patient_id", "triage_class", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Fold manifest is missing columns: {sorted(missing)}")
    if manifest["study_id"].duplicated().any():
        raise ValueError("Each study must appear exactly once in the fold manifest")
    patient_fold_counts = manifest.groupby("patient_id")["fold"].nunique()
    if (patient_fold_counts > 1).any():
        leaked = patient_fold_counts[patient_fold_counts > 1].index.tolist()
        raise ValueError(f"Patient leakage across folds: {leaked[:10]}")
    if (manifest["fold"] < 0).any():
        raise ValueError("Some studies were not assigned to a fold")
    expected = set(range(n_folds)) if n_folds is not None else set(range(int(manifest["fold"].max()) + 1))
    if set(manifest["fold"].unique()) != expected:
        raise ValueError("Fold ids must be contiguous from zero")


def fold_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["fold", "triage_class"]).size().unstack(fill_value=0)
        .rename(columns=lambda value: f"class_{value}")
        .assign(studies=lambda frame: frame.sum(axis=1))
    )
