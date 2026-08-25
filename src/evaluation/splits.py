"""Shared competition-fold resolution for every training strategy.

The source manifest is study-level and patient-grouped.  All downstream
loaders use the same validation study IDs so their OOF predictions can be
joined safely and compared without split noise or patient leakage.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import pandas as pd

from src.config import FOLD_MANIFEST_PATH
from src.evaluation.folds import validate_fold_manifest

T = TypeVar("T")


def normalize_study_id(value: object) -> str:
    """Return the canonical string representation used in ``folds.csv``."""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text:
        raise ValueError("Study ID cannot be empty")
    return text


def study_id_from_path(path: str | Path) -> str:
    """Extract ``study_id`` from ``BRN_<id>[...].nii.gz`` paths."""
    name = Path(path).name
    match = re.match(r"(?:BRN_)?([^_\.]+)", name)
    if not match:
        raise ValueError(f"Cannot extract study ID from path: {path}")
    return normalize_study_id(match.group(1))


def load_fold_manifest(path: str | Path | None = None) -> pd.DataFrame:
    selected = Path(path or FOLD_MANIFEST_PATH)
    if not selected.is_file():
        raise FileNotFoundError(
            f"Competition fold manifest not found: {selected}. "
            "Run `python scripts/build_folds.py` first."
        )
    manifest = pd.read_csv(selected, dtype={"study_id": str, "patient_id": str})
    manifest["study_id"] = manifest["study_id"].map(normalize_study_id)
    manifest["patient_id"] = manifest["patient_id"].map(normalize_study_id)
    validate_fold_manifest(manifest)
    return manifest


def validation_study_ids(fold: int, path: str | Path | None = None) -> set[str]:
    manifest = load_fold_manifest(path)
    available = sorted(int(value) for value in manifest["fold"].unique())
    if fold not in available:
        raise ValueError(f"Fold {fold} is unavailable; expected one of {available}")
    return set(manifest.loc[manifest["fold"] == fold, "study_id"])


def split_study_ids(
    study_ids: Iterable[object],
    fold: int,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[set[str], set[str]]:
    """Split available studies into train/validation using the shared fold."""
    available = {normalize_study_id(value) for value in study_ids}
    known = set(load_fold_manifest(manifest_path)["study_id"])
    unknown = sorted(available - known)
    if unknown:
        raise ValueError(
            "Studies missing from the immutable fold manifest: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    validation = available & validation_study_ids(fold, manifest_path)
    training = available - validation
    if not validation or not training:
        raise ValueError(
            f"Fold {fold} produced an empty partition: "
            f"train={len(training)}, val={len(validation)}"
        )
    return training, validation


def split_items_by_fold(
    items: Sequence[T],
    fold: int,
    *,
    id_getter,
    manifest_path: str | Path | None = None,
) -> tuple[list[T], list[T]]:
    """Partition arbitrary items while preserving their input order."""
    item_ids = [normalize_study_id(id_getter(item)) for item in items]
    training_ids, validation_ids = split_study_ids(
        item_ids, fold, manifest_path=manifest_path
    )
    training = [item for item, study_id in zip(items, item_ids) if study_id in training_ids]
    validation = [item for item, study_id in zip(items, item_ids) if study_id in validation_ids]
    return training, validation


def build_nnunet_splits(
    case_names: Sequence[str],
    *,
    manifest_path: str | Path | None = None,
) -> list[dict[str, list[str]]]:
    """Translate the shared manifest to nnU-Net's ``splits_final.json``."""
    if not case_names:
        raise ValueError("Cannot build nnU-Net folds without cases")
    manifest = load_fold_manifest(manifest_path)
    folds = sorted(int(value) for value in manifest["fold"].unique())
    result: list[dict[str, list[str]]] = []
    for fold in folds:
        train, val = split_items_by_fold(
            list(case_names), fold, id_getter=study_id_from_path,
            manifest_path=manifest_path,
        )
        result.append({"train": list(train), "val": list(val)})
    return result


def write_nnunet_splits(
    output_path: str | Path,
    case_names: Sequence[str],
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Write and return a deterministic nnU-Net custom split file."""
    selected = Path(output_path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(
        json.dumps(build_nnunet_splits(case_names, manifest_path=manifest_path), indent=2),
        encoding="utf-8",
    )
    return selected
