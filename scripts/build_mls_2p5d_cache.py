"""Build the immutable, deploy-aligned float32 cache for MLS 2.5D training.

The cache stores a single three-window representation per ordered DICOM slice:
``[D, 3, 512, 512]``.  The training loader materializes z-1/z/z+1 context at
read time, so it never stores three redundant copies of the same study and the
same ``input_contract`` is used by CUDA inference.

This is preprocessing only: it never constructs a model, executes a model
forward, or falls back to CPU model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_TRAINING_DIR, TRAINING_CSV_PATH, TRAINING_PKL_PATH
from src.evaluation.splits import normalize_study_id
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.context_cache import (
    CACHE_SCHEMA_VERSION,
    load_mls_2p5d_cache_manifest,
    sha256_file,
)
from src.strategies.mls_heatmap.input_contract import (
    CONTEXT_CHANNELS,
    WINDOW_ORDER,
    create_windowed_input,
)


DEFAULT_SOURCE_LABELS = PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2" / "mls_labels_multitask.csv"
DEFAULT_SLICE_TARGETS = PROJECT_ROOT / "Data" / "processed" / "ich_v2" / "BrainICHPartial" / "slice_targets.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Data" / "processed" / "mls_2p5d_v1"
DEFAULT_FOLD_MANIFEST = PROJECT_ROOT / "config" / "folds.csv"


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def sop_uid_from_image_name(study_id: str, image_name: str) -> str:
    """Recover the exact SOP UID from either legacy or explicit-negative PNG name."""
    stem = Path(str(image_name)).stem
    prefix = f"{study_id}_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"MLS image name {image_name!r} does not begin with its study id {study_id!r}"
        )
    uid = stem[len(prefix):]
    if uid.endswith("_negative"):
        uid = uid[: -len("_negative")]
    if not uid:
        raise ValueError(f"MLS image name {image_name!r} has no SOP UID")
    return uid


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _load_metadata_contract(metadata_path: Path) -> pd.DataFrame:
    """Load only the raw, study-level facts needed by the cache contract.

    ``training_df.pkl`` is slice-level and deliberately incomplete for a small
    set of clean-negative studies.  It is therefore valid for it to contain
    fewer rows than the raw DICOM tree.  ``NumDicomFiles`` is the authoritative
    per-study depth; the raw reader is checked against it below.
    """
    required = [
        "dicom_series.id",
        "MidlineShiftMM",
        "dicom_series.PixelSpacing0",
        "dicom_series.PixelSpacing1",
        "dicom_series.NumDicomFiles",
    ]
    if metadata_path.suffix.lower() == ".csv":
        metadata = pd.read_csv(metadata_path, usecols=required)
    else:
        metadata = pd.read_pickle(metadata_path).loc[:, required].copy()
    metadata["dicom_series.id"] = metadata["dicom_series.id"].map(normalize_study_id)
    numeric_columns = [
        "MidlineShiftMM",
        "dicom_series.PixelSpacing0",
        "dicom_series.PixelSpacing1",
        "dicom_series.NumDicomFiles",
    ]
    for column in numeric_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    if (
        metadata["MidlineShiftMM"].isna().any()
        or (metadata["MidlineShiftMM"] < 0).any()
        or metadata[["dicom_series.PixelSpacing0", "dicom_series.PixelSpacing1"]].isna().any().any()
        or (metadata[["dicom_series.PixelSpacing0", "dicom_series.PixelSpacing1"]] <= 0).any().any()
        or metadata["dicom_series.NumDicomFiles"].isna().any()
        or (metadata["dicom_series.NumDicomFiles"] < 1).any()
    ):
        raise ValueError("Raw metadata contains invalid MLS cache geometry values")

    grouped = metadata.groupby("dicom_series.id", sort=True)
    inconsistent = {
        column: grouped[column].nunique(dropna=False)
        for column in (
            "dicom_series.PixelSpacing0",
            "dicom_series.PixelSpacing1",
            "dicom_series.NumDicomFiles",
        )
    }
    invalid_studies = sorted({
        study_id
        for values in inconsistent.values()
        for study_id, count in values.items()
        if int(count) != 1
    })
    if invalid_studies:
        raise ValueError(
            "Raw metadata has inconsistent spacing/depth within study: "
            f"{invalid_studies[:10]}"
        )
    contract = grouped.agg({
        "MidlineShiftMM": "max",
        "dicom_series.PixelSpacing0": "first",
        "dicom_series.PixelSpacing1": "first",
        "dicom_series.NumDicomFiles": "first",
    }).rename(columns={
        "MidlineShiftMM": "study_mls_mm",
        # DICOM PixelSpacing is (row=y, column=x); the deployed reader maps
        # metadata[0] -> spacing_y and metadata[1] -> spacing_x.
        "dicom_series.PixelSpacing1": "spacing_x",
        "dicom_series.PixelSpacing0": "spacing_y",
        "dicom_series.NumDicomFiles": "raw_dicom_count",
    })
    contract["raw_dicom_count"] = contract["raw_dicom_count"].astype(int)
    return contract


def _load_fold_assignments(fold_manifest: Path) -> pd.DataFrame:
    folds = pd.read_csv(fold_manifest, dtype={"study_id": str})
    _require_columns(folds, {"study_id", "patient_id", "triage_class", "fold"}, "fold manifest")
    folds = folds.loc[:, ["study_id", "patient_id", "triage_class", "fold"]].copy()
    folds["study_id"] = folds["study_id"].map(normalize_study_id)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    folds["triage_class"] = pd.to_numeric(folds["triage_class"], errors="raise").astype(int)
    if (
        folds["study_id"].duplicated().any()
        or not folds["fold"].between(0, 4).all()
        or not folds["triage_class"].between(0, 2).all()
        or folds["patient_id"].isna().any()
    ):
        raise ValueError("Fold manifest must have one study row with fold in [0, 4]")
    return folds.set_index("study_id")


def attach_slice_indices(
    labels: pd.DataFrame,
    slice_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Map every training row to one unique ordered DICOM slice, fail-closed."""
    _require_columns(labels, {"patient_id", "image_name", "is_target"}, "MLS source labels")
    _require_columns(slice_targets, {"study_id", "sop_instance_uid", "slice_index"}, "slice_targets")
    source = labels.copy().reset_index(names="source_row_index")
    source["patient_id"] = source["patient_id"].map(normalize_study_id)
    source["sop_instance_uid"] = [
        sop_uid_from_image_name(study_id, image_name)
        for study_id, image_name in zip(source["patient_id"], source["image_name"])
    ]
    if source.duplicated(["patient_id", "sop_instance_uid"]).any():
        duplicates = source.loc[
            source.duplicated(["patient_id", "sop_instance_uid"], keep=False),
            ["patient_id", "sop_instance_uid"],
        ].head(5).to_dict(orient="records")
        raise ValueError(f"MLS source labels map duplicate rows to one DICOM: {duplicates}")

    indexed = slice_targets.copy()
    indexed["study_id"] = indexed["study_id"].map(normalize_study_id)
    indexed["sop_instance_uid"] = indexed["sop_instance_uid"].astype(str)
    indexed["slice_index"] = pd.to_numeric(indexed["slice_index"], errors="raise").astype(int)
    if indexed.duplicated(["study_id", "sop_instance_uid"]).any():
        raise ValueError("slice_targets has duplicate study/SOP rows")
    if indexed.duplicated(["study_id", "slice_index"]).any():
        raise ValueError("slice_targets has duplicate study/slice_index rows")
    indexed["slice_count"] = indexed.groupby("study_id")["slice_index"].transform("size").astype(int)
    merged = source.merge(
        indexed[["study_id", "sop_instance_uid", "slice_index", "slice_count"]],
        how="left",
        left_on=["patient_id", "sop_instance_uid"],
        right_on=["study_id", "sop_instance_uid"],
        validate="one_to_one",
    )
    if merged["slice_index"].isna().any():
        missing = merged.loc[
            merged["slice_index"].isna(), ["patient_id", "image_name", "sop_instance_uid"]
        ].head(10).to_dict(orient="records")
        raise ValueError(f"MLS source labels missing from slice_targets: {missing}")
    merged["slice_index"] = merged["slice_index"].astype(int)
    merged["slice_count"] = merged["slice_count"].astype(int)
    merged = merged.drop(columns=["study_id"])
    return merged.sort_values("source_row_index").reset_index(drop=True)


def _full_slice_target_orders(slice_targets: pd.DataFrame) -> dict[str, list[str]]:
    """Return the complete expected SOP order for every raw study.

    Context can touch an unlabeled neighbour, so validating only the 3,484
    selected MLS centre SOPs is insufficient.  ``slice_targets`` is produced
    from the full sorted raw DICOM series and lets the cache fail closed if any
    unlabelled neighbour was reordered, omitted, or substituted.
    """
    _require_columns(slice_targets, {"study_id", "sop_instance_uid", "slice_index"}, "slice_targets")
    ordered = slice_targets.loc[:, ["study_id", "sop_instance_uid", "slice_index"]].copy()
    ordered["study_id"] = ordered["study_id"].map(normalize_study_id)
    ordered["sop_instance_uid"] = ordered["sop_instance_uid"].astype(str)
    ordered["slice_index"] = pd.to_numeric(ordered["slice_index"], errors="raise").astype(int)
    if ordered.duplicated(["study_id", "sop_instance_uid"]).any():
        raise ValueError("slice_targets has duplicate study/SOP rows")
    if ordered.duplicated(["study_id", "slice_index"]).any():
        raise ValueError("slice_targets has duplicate study/slice_index rows")
    orders: dict[str, list[str]] = {}
    for study_id, group in ordered.groupby("study_id", sort=True):
        group = group.sort_values("slice_index")
        indices = group["slice_index"].tolist()
        if indices != list(range(len(group))):
            raise ValueError(f"slice_targets has a non-contiguous raw z order for study {study_id}")
        orders[str(study_id)] = group["sop_instance_uid"].tolist()
    return orders


def _ordered_sop_uids(reader: BrainDicomReader, study_id: str) -> list[str]:
    uids = [str(getattr(item, "SOPInstanceUID", "")) for item in reader.slices]
    if any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError(f"Study {study_id} has missing or duplicate SOPInstanceUID values")
    return uids


def _validate_study_geometry(reader: BrainDicomReader, study_id: str, image_size: int) -> None:
    if not reader.slices:
        raise ValueError(f"Study {study_id} contains no sorted DICOM slices")
    positions: list[float] = []
    reference_orientation: np.ndarray | None = None
    for offset, dataset in enumerate(reader.slices):
        shape = (int(dataset.Rows), int(dataset.Columns))
        if shape != (image_size, image_size):
            raise ValueError(
                f"Study {study_id} slice {offset} has native shape {shape}; "
                f"the 2.5D contract requires {(image_size, image_size)} without silent resize"
            )
        position = getattr(dataset, "ImagePositionPatient", None)
        orientation = getattr(dataset, "ImageOrientationPatient", None)
        if position is None or len(position) < 3 or orientation is None or len(orientation) != 6:
            raise ValueError(f"Study {study_id} lacks complete DICOM position/orientation metadata")
        positions.append(float(position[2]))
        current_orientation = np.asarray(orientation, dtype=float)
        if reference_orientation is None:
            reference_orientation = current_orientation
        elif not np.allclose(current_orientation, reference_orientation, rtol=0.0, atol=1e-5):
            raise ValueError(f"Study {study_id} has inconsistent ImageOrientationPatient values")
    if any(next_position <= current for current, next_position in zip(positions, positions[1:])):
        raise ValueError(f"Study {study_id} has non-strict z ordering after DICOM sort")


def _validate_reader_spacing(
    reader: BrainDicomReader,
    study_id: str,
    *,
    expected_spacing_x: float,
    expected_spacing_y: float,
) -> None:
    """Bind train-label millimetres to the same raw DICOM spacing used at inference."""
    actual_x = float(reader.metadata["spacing_x"])
    actual_y = float(reader.metadata["spacing_y"])
    if not np.isclose(actual_x, expected_spacing_x, rtol=0.0, atol=1e-7) or not np.isclose(
        actual_y, expected_spacing_y, rtol=0.0, atol=1e-7,
    ):
        raise ValueError(
            f"Study {study_id} metadata/DICOM spacing mismatch: "
            f"metadata=({expected_spacing_x}, {expected_spacing_y}), "
            f"DICOM=({actual_x}, {actual_y})"
        )
    for index, dataset in enumerate(reader.slices):
        spacing = getattr(dataset, "PixelSpacing", None)
        if spacing is None or len(spacing) != 2:
            raise ValueError(f"Study {study_id} slice {index} lacks two-value PixelSpacing")
        slice_x, slice_y = float(spacing[1]), float(spacing[0])
        if not np.isfinite([slice_x, slice_y]).all() or slice_x <= 0.0 or slice_y <= 0.0:
            raise ValueError(f"Study {study_id} slice {index} has invalid PixelSpacing")
        if not np.isclose(slice_x, actual_x, rtol=0.0, atol=1e-7) or not np.isclose(
            slice_y, actual_y, rtol=0.0, atol=1e-7,
        ):
            raise ValueError(f"Study {study_id} has inconsistent raw DICOM PixelSpacing")


def _build_study_volume(reader: BrainDicomReader, image_size: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    for dataset in reader.slices:
        hu = reader._pixel_to_hu(dataset.pixel_array, dataset)
        if tuple(hu.shape) != (image_size, image_size):
            raise ValueError(f"HU array shape changed during DICOM decode: {hu.shape}")
        frames.append(create_windowed_input(hu, 3))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def _raw_study_fingerprint(reader: BrainDicomReader, study_id: str) -> dict[str, Any]:
    """Bind each cached volume to the exact raw DICOM bytes it was built from."""
    records: list[str] = []
    total_bytes = 0
    filenames: set[str] = set()
    for index, dataset in enumerate(reader.slices):
        path = Path(str(dataset.filename))
        if not path.is_file():
            raise FileNotFoundError(f"Study {study_id} DICOM file vanished during cache build: {path}")
        if path.name in filenames:
            raise ValueError(f"Study {study_id} has duplicate raw DICOM filename {path.name}")
        filenames.add(path.name)
        digest = sha256_file(path)
        size = int(path.stat().st_size)
        records.append(f"{index}\t{path.name}\t{size}\t{digest}")
        total_bytes += size
    return {
        "raw_dicom_sha256": hashlib.sha256(
            ("\n".join(records) + "\n").encode("utf-8")
        ).hexdigest(),
        "raw_dicom_bytes": total_bytes,
        "raw_dicom_files": len(records),
    }


def _study_record(
    cache_path: Path,
    volume: np.ndarray,
    ordered_uids: list[str],
    raw_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "file": cache_path.name,
        "sha256": sha256_file(cache_path),
        "bytes": int(cache_path.stat().st_size),
        "shape": [int(value) for value in volume.shape],
        "sop_order_sha256": hashlib.sha256(
            "\n".join(ordered_uids).encode("utf-8")
        ).hexdigest(),
        **raw_fingerprint,
    }


def _load_compatible_existing_manifest(
    output_root: Path,
    *,
    labels_sha256: str,
    slice_targets_sha256: str,
    raw_metadata_sha256: str,
    fold_manifest_sha256: str,
    input_contract_sha256: str,
    builder_sha256: str,
    image_size: int,
) -> dict[str, Any] | None:
    """Return a reusable manifest only when every source-affecting fact matches.

    In particular, a partial/interrupted cache without a finalized manifest is
    not reusable.  Silently reusing it could pair images made with old CT
    windows or raw ordering with a newly written provenance manifest.
    """
    path = output_root / "cache_manifest.json"
    if not path.is_file():
        stale_studies = output_root / "studies"
        if stale_studies.is_dir() and any(stale_studies.glob("*.npy")):
            raise ValueError(
                "MLS 2.5D output contains study volumes but no finalized manifest; "
                "refusing to reuse an interrupted cache"
            )
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    sources = existing.get("sources", {})
    compatible = (
        int(existing.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
        and int(existing.get("image_size", -1)) == image_size
        and sources.get("source_labels_sha256") == labels_sha256
        and sources.get("slice_targets_sha256") == slice_targets_sha256
        and sources.get("raw_metadata_sha256") == raw_metadata_sha256
        and sources.get("fold_manifest_sha256") == fold_manifest_sha256
        and sources.get("input_contract_sha256") == input_contract_sha256
        and sources.get("builder_sha256") == builder_sha256
    )
    if not compatible:
        raise ValueError(
            "Existing MLS 2.5D cache manifest has a different source/runtime contract; "
            "choose a new output root rather than reusing incompatible volumes"
        )
    if not isinstance(existing.get("study_files"), dict):
        raise ValueError("Existing MLS 2.5D cache manifest lacks per-study integrity records")
    return existing


def build_cache(
    *,
    source_labels: Path,
    slice_targets_path: Path,
    raw_root: Path,
    output_root: Path,
    fold_manifest: Path,
    image_size: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Build/resume cache and return a public, non-prediction receipt."""
    for path, label in ((source_labels, "source labels"), (slice_targets_path, "slice targets"), (raw_root, "raw DICOM root"), (fold_manifest, "fold manifest")):
        if not path.exists():
            raise FileNotFoundError(f"MLS 2.5D {label} is missing: {path}")
    metadata_path = TRAINING_CSV_PATH if TRAINING_CSV_PATH.is_file() else TRAINING_PKL_PATH
    if not metadata_path.is_file():
        raise FileNotFoundError("MLS 2.5D cache requires DVC raw training metadata")

    labels_sha256 = sha256_file(source_labels)
    slice_targets_sha256 = sha256_file(slice_targets_path)
    raw_metadata_sha256 = sha256_file(metadata_path)
    fold_manifest_sha256 = sha256_file(fold_manifest)
    input_contract_sha256 = sha256_file(
        PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py"
    )
    builder_sha256 = sha256_file(Path(__file__))
    labels = pd.read_csv(source_labels, dtype={"patient_id": str})
    slice_targets = pd.read_csv(slice_targets_path, dtype={"study_id": str, "sop_instance_uid": str})
    context_labels = attach_slice_indices(labels, slice_targets)
    full_target_orders = _full_slice_target_orders(slice_targets)
    metadata_contract = _load_metadata_contract(metadata_path)
    fold_assignments = _load_fold_assignments(fold_manifest)
    labelled_studies = set(context_labels["patient_id"].astype(str))
    metadata_studies = set(metadata_contract.index.astype(str))
    fold_studies = set(fold_assignments.index.astype(str))
    target_studies = set(full_target_orders)
    if not (
        labelled_studies == metadata_studies == fold_studies == target_studies
        and len(labelled_studies) == 338
    ):
        raise ValueError(
            "MLS 2.5D source membership must be the same 338 studies across "
            "labels, raw metadata, fold manifest, and full slice targets"
        )
    for column in ("study_mls_mm", "spacing_x", "spacing_y", "raw_dicom_count"):
        context_labels[column] = context_labels["patient_id"].map(metadata_contract[column])
    context_labels["fold"] = context_labels["patient_id"].map(fold_assignments["fold"])
    required_context = ["study_mls_mm", "spacing_x", "spacing_y", "raw_dicom_count", "fold"]
    if context_labels[required_context].isna().any().any():
        missing = context_labels.loc[
            context_labels[required_context].isna().any(axis=1), "patient_id"
        ].drop_duplicates().tolist()
        raise ValueError(
            "MLS 2.5D cache cannot attach raw metadata/fold facts to every training row: "
            f"{missing[:10]}"
        )
    context_labels["fold"] = context_labels["fold"].astype(int)
    context_labels["raw_dicom_count"] = context_labels["raw_dicom_count"].astype(int)

    result: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "dry_run_validated" if dry_run else "building",
        "source_rows": int(len(context_labels)),
        "positive_rows": int(pd.to_numeric(context_labels["is_target"], errors="raise").eq(1).sum()),
        "negative_rows": int(pd.to_numeric(context_labels["is_target"], errors="raise").eq(0).sum()),
        "studies": int(context_labels["patient_id"].nunique()),
        "source_labels_sha256": labels_sha256,
        "slice_targets_sha256": slice_targets_sha256,
        "model_compute": "none",
        "pixel_decode": False,
    }
    if overwrite:
        raise ValueError(
            "MLS 2.5D cache is immutable: --overwrite is forbidden. "
            "Use a new versioned output root instead."
        )
    if not dry_run:
        if output_root.exists():
            if not output_root.is_dir():
                raise ValueError(f"MLS 2.5D output root is not a directory: {output_root}")
            active_output_root = output_root
            existing_manifest = _load_compatible_existing_manifest(
                active_output_root,
                labels_sha256=labels_sha256,
                slice_targets_sha256=slice_targets_sha256,
                raw_metadata_sha256=raw_metadata_sha256,
                fold_manifest_sha256=fold_manifest_sha256,
                input_contract_sha256=input_contract_sha256,
                builder_sha256=builder_sha256,
                image_size=image_size,
            )
            if existing_manifest is None:
                raise ValueError(
                    "MLS 2.5D output root exists without a finalized cache manifest; "
                    "use a new versioned output root"
                )
            # A finalized root is immutable.  Rewriting labels, manifest or
            # receipt after a successful build would make provenance depend on
            # when this command happened to be rerun.  Training separately
            # requires a fresh full raw-DICOM validation receipt.
            _, existing_manifest_sha256 = load_mls_2p5d_cache_manifest(active_output_root)
            result.update({
                "status": "existing_immutable_cache_reused",
                "pixel_decode": False,
                "cache_manifest": str(active_output_root / "cache_manifest.json"),
                "cache_manifest_sha256": existing_manifest_sha256,
                "cache_receipt": str(active_output_root / "cache_receipt.json"),
                "validation_required_before_cuda_training": True,
            })
            return result
        else:
            active_output_root = output_root.parent / f".{output_root.name}.building"
            if active_output_root.exists():
                raise ValueError(
                    "MLS 2.5D staging root already exists, likely from an interrupted build: "
                    f"{active_output_root}. Preserve it for audit and choose a new output root."
                )
            active_output_root.mkdir(parents=True, exist_ok=False)
            existing_manifest = None
        studies_dir = active_output_root / "studies"
        studies_dir.mkdir(parents=True, exist_ok=True)
    else:
        existing_manifest = None
        active_output_root = output_root
        studies_dir = active_output_root / "studies"

    study_records: dict[str, dict[str, Any]] = {}
    rows: list[pd.DataFrame] = []
    for study_id, group in context_labels.groupby("patient_id", sort=True):
        reader = BrainDicomReader(str(raw_root / study_id)).load_and_sort()
        _validate_study_geometry(reader, study_id, image_size)
        _validate_reader_spacing(
            reader,
            study_id,
            expected_spacing_x=float(group["spacing_x"].iloc[0]),
            expected_spacing_y=float(group["spacing_y"].iloc[0]),
        )
        ordered_uids = _ordered_sop_uids(reader, study_id)
        expected_raw_depth = int(group["raw_dicom_count"].iloc[0])
        if len(reader.dicom_files) != expected_raw_depth or len(ordered_uids) != expected_raw_depth:
            raise ValueError(
                f"Study {study_id} raw DICOM depth mismatch: files={len(reader.dicom_files)}, "
                f"readable_sorted={len(ordered_uids)}, metadata_NumDicomFiles={expected_raw_depth}"
            )
        expected_full_order = full_target_orders.get(str(study_id))
        if expected_full_order is None:
            raise ValueError(f"Study {study_id} has no full slice_targets DICOM order")
        if len(expected_full_order) != expected_raw_depth or ordered_uids != expected_full_order:
            raise ValueError(
                f"Study {study_id} raw DICOM SOP order disagrees with the full slice_targets contract"
            )
        raw_index_by_uid = {uid: index for index, uid in enumerate(ordered_uids)}
        enriched = group.copy()
        enriched["slice_target_index"] = enriched["slice_index"].astype(int)
        enriched["slice_index"] = enriched["sop_instance_uid"].map(raw_index_by_uid)
        if enriched["slice_index"].isna().any():
            missing = enriched.loc[
                enriched["slice_index"].isna(), "sop_instance_uid"
            ].head(5).tolist()
            raise ValueError(
                f"Study {study_id} MLS centre SOP is absent from raw DICOM order: {missing}"
            )
        enriched["slice_index"] = enriched["slice_index"].astype(int)
        if not enriched["slice_index"].eq(enriched["slice_target_index"]).all():
            mismatch = enriched.loc[
                ~enriched["slice_index"].eq(enriched["slice_target_index"]),
                ["sop_instance_uid", "slice_target_index", "slice_index"],
            ].head(5).to_dict(orient="records")
            raise ValueError(
                f"Study {study_id} raw DICOM order disagrees with slice_targets: {mismatch}"
            )
        if dry_run:
            continue
        raw_fingerprint = _raw_study_fingerprint(reader, study_id)
        cache_path = studies_dir / f"{study_id}.npy"
        if cache_path.exists() and not overwrite:
            if existing_manifest is None:
                raise RuntimeError("Internal error: cache reuse without a verified manifest")
            previous_record = existing_manifest["study_files"].get(str(study_id))
            if not isinstance(previous_record, dict):
                raise ValueError(
                    f"Existing MLS 2.5D manifest lacks a record for cached study {study_id}"
                )
            volume = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (len(ordered_uids), 3, image_size, image_size)
            if tuple(volume.shape) != expected_shape or volume.dtype != np.float32:
                raise ValueError(
                    f"Existing MLS 2.5D cache is incompatible for study {study_id}: "
                    f"shape={tuple(volume.shape)} dtype={volume.dtype}"
                )
            observed_record = _study_record(cache_path, volume, ordered_uids, raw_fingerprint)
            if observed_record != previous_record:
                raise ValueError(
                    f"Existing MLS 2.5D cache integrity mismatch for study {study_id}; "
                    "refusing to reuse a changed volume"
                )
        else:
            volume = _build_study_volume(reader, image_size)
            _atomic_npy(volume, cache_path)
        study_records[str(study_id)] = _study_record(
            cache_path, volume, ordered_uids, raw_fingerprint
        )
        enriched["cache_relpath"] = f"studies/{study_id}.npy"
        enriched["source_dicom_name"] = [
            Path(reader.slices[index].filename).name for index in enriched["slice_index"].astype(int)
        ]
        enriched["native_height"] = image_size
        enriched["native_width"] = image_size
        enriched["cache_schema_version"] = CACHE_SCHEMA_VERSION
        rows.append(enriched)

    if dry_run:
        result.update({
            "status": "dry_run_preflight_passed",
            "pixel_decode": False,
            "validated_raw_studies": int(context_labels["patient_id"].nunique()),
            "validated_raw_dicoms": int(context_labels[["patient_id", "raw_dicom_count"]]
                                        .drop_duplicates()["raw_dicom_count"].sum()),
            "source_membership": "338 studies matched across labels/metadata/folds/slice_targets",
        })
        return result

    output_labels = pd.concat(rows, ignore_index=True).sort_values("source_row_index").reset_index(drop=True)
    output_labels = output_labels.drop(columns=["source_row_index"])
    labels_output_path = active_output_root / "labels_context.csv"
    _atomic_csv(output_labels, labels_output_path)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_contract": "mls_2p5d_float32_v1",
        "image_size": int(image_size),
        "base_input_channels": 3,
        "context_input_channels": CONTEXT_CHANNELS,
        "window_order": list(WINDOW_ORDER),
        "edge_policy": "replicate",
        "cache_dtype": "float32",
        "labels_csv": labels_output_path.name,
        "labels_sha256": sha256_file(labels_output_path),
        "study_cache_dir": studies_dir.name,
        "rows": int(len(output_labels)),
        "studies": int(output_labels["patient_id"].nunique()),
        "study_files": study_records,
        "sources": {
            "source_labels": str(source_labels),
            "source_labels_sha256": labels_sha256,
            "slice_targets": str(slice_targets_path),
            "slice_targets_sha256": slice_targets_sha256,
            "raw_metadata": str(metadata_path),
            "raw_metadata_sha256": raw_metadata_sha256,
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": fold_manifest_sha256,
            "input_contract_sha256": input_contract_sha256,
            "builder_sha256": builder_sha256,
        },
        "model_compute": "none",
    }
    manifest_path = active_output_root / "cache_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = sha256_file(manifest_path)
    receipt_path = active_output_root / "cache_receipt.json"
    _atomic_text(receipt_path, json.dumps({
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_manifest": manifest_path.name,
        "cache_manifest_sha256": manifest_sha256,
        "cache_bytes": int(sum(item["bytes"] for item in study_records.values())),
        "raw_dicom_files": int(sum(item["raw_dicom_files"] for item in study_records.values())),
        "raw_dicom_bytes": int(sum(item["raw_dicom_bytes"] for item in study_records.values())),
        "model_compute": "none",
        "pixel_decode": True,
    }, indent=2, sort_keys=True) + "\n")
    if active_output_root != output_root:
        os.replace(active_output_root, output_root)
    result.update({
        "status": "completed",
        "pixel_decode": True,
        "cache_manifest": str(output_root / manifest_path.name),
        "cache_manifest_sha256": manifest_sha256,
        "cache_receipt": str(output_root / receipt_path.name),
        "cache_bytes": int(sum(item["bytes"] for item in study_records.values())),
        "labels_context_sha256": manifest["labels_sha256"],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-labels", type=Path, default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--slice-targets", type=Path, default=DEFAULT_SLICE_TARGETS)
    parser.add_argument("--raw-root", type=Path, default=RAW_TRAINING_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.image_size != 512:
        raise ValueError("The initial MLS 2.5D contract is locked to native 512x512 slices")
    result = build_cache(
        source_labels=args.source_labels.resolve(),
        slice_targets_path=args.slice_targets.resolve(),
        raw_root=args.raw_root.resolve(),
        output_root=args.output_root.resolve(),
        fold_manifest=args.fold_manifest.resolve(),
        image_size=args.image_size,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
