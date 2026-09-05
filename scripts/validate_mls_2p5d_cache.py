"""Fail-closed integrity validation for the immutable MLS G1 image cache.

This program performs no model construction or model forward.  It validates
the final cache only on the remote host where the DICOM tree lives, binding
every cached study to the raw DICOM order/content, raw metadata and immutable
fold manifest before a CUDA training campaign is allowed to start.
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

from scripts.build_mls_2p5d_cache import (
    _full_slice_target_orders,
    _load_fold_assignments,
    _load_metadata_contract,
    _ordered_sop_uids,
    _raw_study_fingerprint,
    _validate_study_geometry,
    _validate_reader_spacing,
)
from src.config import RAW_TRAINING_DIR
from src.evaluation.splits import normalize_study_id
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.context_cache import (
    CACHE_SCHEMA_VERSION,
    load_mls_2p5d_cache_manifest,
    sha256_file,
)
from src.strategies.mls_heatmap.input_contract import CONTEXT_CHANNELS, WINDOW_ORDER


DEFAULT_CACHE_ROOT = PROJECT_ROOT / "Data" / "processed" / "mls_2p5d_v1"
DEFAULT_SLICE_TARGETS = PROJECT_ROOT / "Data" / "processed" / "ich_v2" / "BrainICHPartial" / "slice_targets.csv"
DEFAULT_FOLD_MANIFEST = PROJECT_ROOT / "config" / "folds.csv"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"MLS 2.5D {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(
            f"MLS 2.5D {label} SHA-256 mismatch: expected {expected}, got {observed}"
        )


def _validate_source_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("MLS 2.5D cache manifest has no source contract")
    required = {
        "source_labels", "source_labels_sha256", "slice_targets", "slice_targets_sha256",
        "raw_metadata", "raw_metadata_sha256", "fold_manifest", "fold_manifest_sha256",
        "input_contract_sha256", "builder_sha256",
    }
    missing = required - set(sources)
    if missing:
        raise ValueError(f"MLS 2.5D cache manifest source contract is incomplete: {sorted(missing)}")
    source_labels = Path(str(sources["source_labels"]))
    slice_targets = Path(str(sources["slice_targets"]))
    raw_metadata = Path(str(sources["raw_metadata"]))
    fold_manifest = Path(str(sources["fold_manifest"]))
    _require_sha(source_labels, str(sources["source_labels_sha256"]), "source labels")
    _require_sha(slice_targets, str(sources["slice_targets_sha256"]), "slice targets")
    _require_sha(raw_metadata, str(sources["raw_metadata_sha256"]), "raw metadata")
    _require_sha(fold_manifest, str(sources["fold_manifest_sha256"]), "fold manifest")
    _require_sha(
        PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
        str(sources["input_contract_sha256"]),
        "input contract source",
    )
    _require_sha(Path(__file__).with_name("build_mls_2p5d_cache.py"), str(sources["builder_sha256"]), "cache builder source")
    return source_labels, slice_targets, raw_metadata, fold_manifest


def validate_cache(
    *,
    cache_root: Path,
    expected_manifest_sha256: str | None,
    raw_root: Path,
    verify_raw_fingerprints: bool,
) -> dict[str, Any]:
    """Validate every cache file and return a hash-bound, non-prediction receipt."""
    manifest, manifest_sha256 = load_mls_2p5d_cache_manifest(
        cache_root, expected_sha256=expected_manifest_sha256,
    )
    if int(manifest["schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError("Unexpected MLS 2.5D cache schema")
    if manifest.get("cache_contract") != "mls_2p5d_float32_v1":
        raise ValueError("Unexpected MLS 2.5D cache contract")
    if list(manifest.get("window_order", ())) != list(WINDOW_ORDER):
        raise ValueError("MLS 2.5D cache window order differs from the input contract")
    if int(manifest["context_input_channels"]) != CONTEXT_CHANNELS:
        raise ValueError("MLS 2.5D cache context channel count is invalid")
    source_labels, slice_targets_path, raw_metadata, fold_manifest = _validate_source_paths(manifest)
    # The source locations recorded by a finalized cache are authoritative;
    # command-line paths cannot silently replace them.
    del source_labels

    labels_path = cache_root / str(manifest["labels_csv"])
    labels = pd.read_csv(labels_path, dtype={"patient_id": str, "sop_instance_uid": str})
    required_labels = {
        "patient_id", "sop_instance_uid", "slice_index", "slice_target_index",
        "fold", "raw_dicom_count", "spacing_x", "spacing_y", "study_mls_mm",
    }
    missing_labels = required_labels - set(labels)
    if missing_labels:
        raise ValueError(f"MLS 2.5D cache labels lack required columns: {sorted(missing_labels)}")
    labels["patient_id"] = labels["patient_id"].map(normalize_study_id)
    labels["slice_index"] = pd.to_numeric(labels["slice_index"], errors="raise").astype(int)
    labels["slice_target_index"] = pd.to_numeric(labels["slice_target_index"], errors="raise").astype(int)
    labels["fold"] = pd.to_numeric(labels["fold"], errors="raise").astype(int)
    labels["raw_dicom_count"] = pd.to_numeric(labels["raw_dicom_count"], errors="raise").astype(int)
    if len(labels) != int(manifest["rows"]) or labels.duplicated(["patient_id", "sop_instance_uid"]).any():
        raise ValueError("MLS 2.5D labels row count or centre-SOP uniqueness is invalid")

    metadata_contract = _load_metadata_contract(Path(raw_metadata))
    folds = _load_fold_assignments(Path(fold_manifest))
    full_orders = _full_slice_target_orders(pd.read_csv(
        slice_targets_path, dtype={"study_id": str, "sop_instance_uid": str},
    ))
    studies = set(labels["patient_id"])
    if not (
        studies == set(metadata_contract.index.astype(str)) == set(folds.index.astype(str)) == set(full_orders)
        and len(studies) == int(manifest["studies"]) == 338
    ):
        raise ValueError("MLS 2.5D validation found inconsistent 338-study membership")

    study_dir = cache_root / str(manifest["study_cache_dir"])
    study_records = manifest.get("study_files")
    if not isinstance(study_records, dict) or set(study_records) != studies:
        raise ValueError("MLS 2.5D manifest study records do not exactly cover the label studies")

    raw_bytes = 0
    cache_bytes = 0
    for study_id, group in labels.groupby("patient_id", sort=True):
        record = study_records[str(study_id)]
        if not isinstance(record, dict):
            raise ValueError(f"MLS 2.5D study record is invalid for {study_id}")
        reader = BrainDicomReader(str(raw_root / str(study_id))).load_and_sort()
        _validate_study_geometry(reader, str(study_id), int(manifest["image_size"]))
        _validate_reader_spacing(
            reader,
            str(study_id),
            expected_spacing_x=float(metadata_contract.loc[study_id, "spacing_x"]),
            expected_spacing_y=float(metadata_contract.loc[study_id, "spacing_y"]),
        )
        uids = _ordered_sop_uids(reader, str(study_id))
        expected_depth = int(group["raw_dicom_count"].iloc[0])
        if (
            len(reader.dicom_files) != expected_depth
            or len(uids) != expected_depth
            or expected_depth != int(metadata_contract.loc[study_id, "raw_dicom_count"])
            or uids != full_orders[str(study_id)]
        ):
            raise ValueError(f"MLS 2.5D raw DICOM depth/order mismatch for study {study_id}")
        uid_to_index = {uid: index for index, uid in enumerate(uids)}
        actual_indices = group["sop_instance_uid"].map(uid_to_index)
        if actual_indices.isna().any() or not actual_indices.astype(int).eq(group["slice_index"]).all():
            raise ValueError(f"MLS 2.5D label centres do not match raw DICOM order for study {study_id}")
        if not group["slice_index"].eq(group["slice_target_index"]).all():
            raise ValueError(f"MLS 2.5D labels differ from immutable slice_targets order for study {study_id}")
        if not group["fold"].eq(int(folds.loc[study_id, "fold"])).all():
            raise ValueError(f"MLS 2.5D label fold mismatch for study {study_id}")
        for column in ("spacing_x", "spacing_y", "study_mls_mm"):
            expected_value = float(metadata_contract.loc[study_id, column])
            observed = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            if not np.isfinite(observed).all() or not np.allclose(observed, expected_value, rtol=0.0, atol=1e-7):
                raise ValueError(f"MLS 2.5D label {column} differs from raw metadata for study {study_id}")

        cache_path = study_dir / str(record.get("file", ""))
        if not cache_path.is_file():
            raise FileNotFoundError(f"MLS 2.5D study cache is missing: {cache_path}")
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_depth, 3, int(manifest["image_size"]), int(manifest["image_size"]))
        if tuple(cache.shape) != expected_shape or cache.dtype != np.float32:
            raise ValueError(f"MLS 2.5D cache shape/dtype mismatch for study {study_id}")
        if sha256_file(cache_path) != str(record.get("sha256", "")):
            raise ValueError(f"MLS 2.5D cache checksum mismatch for study {study_id}")
        if int(cache_path.stat().st_size) != int(record.get("bytes", -1)):
            raise ValueError(f"MLS 2.5D cache byte count mismatch for study {study_id}")
        if str(record.get("sop_order_sha256", "")) != hashlib.sha256(
            "\n".join(uids).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"MLS 2.5D SOP-order checksum mismatch for study {study_id}")
        if verify_raw_fingerprints:
            raw = _raw_study_fingerprint(reader, str(study_id))
            for field, value in raw.items():
                if record.get(field) != value:
                    raise ValueError(f"MLS 2.5D raw DICOM fingerprint mismatch for study {study_id}")
            raw_bytes += int(raw["raw_dicom_bytes"])
        cache_bytes += int(cache_path.stat().st_size)

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_manifest": str((cache_root / "cache_manifest.json").resolve()),
        "cache_manifest_sha256": manifest_sha256,
        "studies": len(studies),
        "rows": len(labels),
        "cache_bytes": cache_bytes,
        "raw_fingerprints_verified": bool(verify_raw_fingerprints),
        "raw_dicom_bytes": raw_bytes if verify_raw_fingerprints else None,
        "model_compute": "none",
        "pixel_decode": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--raw-root", type=Path, default=RAW_TRAINING_DIR)
    parser.add_argument("--skip-raw-fingerprints", action="store_true")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    result = validate_cache(
        cache_root=args.cache_root.resolve(),
        expected_manifest_sha256=args.expected_manifest_sha256,
        raw_root=args.raw_root.resolve(),
        verify_raw_fingerprints=not args.skip_raw_fingerprints,
    )
    receipt = args.receipt or (args.cache_root / "validation_receipt.json")
    _atomic_json(receipt.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
