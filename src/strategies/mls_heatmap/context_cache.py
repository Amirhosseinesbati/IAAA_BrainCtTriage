"""Validation helpers for the immutable MLS 2.5D image cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.strategies.mls_heatmap.input_contract import WINDOW_ORDER


CACHE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 without loading a possibly large cache in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mls_2p5d_cache_manifest(
    cache_root: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and validate a cache manifest before model/data-loader execution.

    The manifest deliberately records only public data-contract metadata and
    checksums. It never contains predictions, model outputs, or credentials.
    """
    root = Path(cache_root)
    manifest_path = root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MLS 2.5D cache manifest is missing: {manifest_path}")
    # Read once: hashing one inode and parsing a later replacement would bind
    # the wrong manifest to workers. The cache is immutable, so fail closed
    # if the exact bytes do not match the configured digest.
    manifest_bytes = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "MLS 2.5D cache manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("MLS 2.5D cache manifest must be a JSON object")
    required = {
        "schema_version",
        "cache_contract",
        "image_size",
        "base_input_channels",
        "context_input_channels",
        "cache_dtype",
        "edge_policy",
        "labels_csv",
        "labels_sha256",
        "study_cache_dir",
        "studies",
        "study_files",
        "rows",
        "window_order",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"MLS 2.5D cache manifest missing keys: {sorted(missing)}")
    if int(manifest["schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MLS 2.5D cache schema {manifest['schema_version']}; "
            f"expected {CACHE_SCHEMA_VERSION}"
        )
    if str(manifest["cache_contract"]) != "mls_2p5d_float32_v1":
        raise ValueError("Unsupported MLS 2.5D cache contract")
    if list(manifest["window_order"]) != list(WINDOW_ORDER):
        raise ValueError("MLS 2.5D cache window order does not match the shared input contract")
    if int(manifest["base_input_channels"]) != 3 or int(manifest["context_input_channels"]) != 9:
        raise ValueError("MLS 2.5D cache must encode three windows and nine context channels")
    if str(manifest["cache_dtype"]) != "float32":
        raise ValueError("MLS 2.5D cache must use float32 to match CUDA inference windows")
    if str(manifest["edge_policy"]) != "replicate":
        raise ValueError("MLS 2.5D cache edge policy must be 'replicate'")
    if (
        not isinstance(manifest["study_files"], dict)
        or len(manifest["study_files"]) != int(manifest["studies"])
    ):
        raise ValueError("MLS 2.5D cache must include one integrity record per study")
    for study_id, record in manifest["study_files"].items():
        if not isinstance(record, dict):
            raise ValueError(f"MLS 2.5D cache study record is invalid for {study_id}")
        file_name = record.get("file")
        shape = record.get("shape")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(not isinstance(value, int) for value in shape)
            or shape[0] < 1
            or shape[1:] != [3, int(manifest["image_size"]), int(manifest["image_size"])]
            or not isinstance(record.get("bytes"), int)
            or int(record["bytes"]) < 1
        ):
            raise ValueError(f"MLS 2.5D cache study integrity record is invalid for {study_id}")
    labels_path = root / str(manifest["labels_csv"])
    study_dir = root / str(manifest["study_cache_dir"])
    if not labels_path.is_file() or not study_dir.is_dir():
        raise FileNotFoundError(
            "MLS 2.5D cache manifest references missing labels or study cache directory"
        )
    if sha256_file(labels_path) != str(manifest["labels_sha256"]):
        raise ValueError("MLS 2.5D cache labels CSV checksum does not match manifest")
    return manifest, actual_sha256


def load_passing_mls_2p5d_validation_receipt(
    receipt_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the independent cache-validation receipt and bind it to a config.

    A manifest proves which cache *should* be used; the validation receipt
    proves that every cached study was subsequently checked against the raw
    DICOM tree.  Keeping this lightweight check in shared source prevents a
    direct trainer invocation from silently bypassing that second proof.
    """
    path = Path(receipt_path)
    if not path.is_file():
        raise FileNotFoundError(f"MLS 2.5D validation receipt is missing: {path}")
    actual_sha256 = sha256_file(path)
    if expected_receipt_sha256 is not None and actual_sha256 != expected_receipt_sha256.lower():
        raise ValueError(
            "MLS 2.5D validation receipt SHA-256 mismatch: "
            f"expected {expected_receipt_sha256}, got {actual_sha256}"
        )
    with path.open("r", encoding="utf-8") as stream:
        receipt = json.load(stream)
    if not isinstance(receipt, dict):
        raise ValueError("MLS 2.5D validation receipt must be a JSON object")
    if (
        receipt.get("status") != "passed"
        or receipt.get("cache_manifest_sha256") != expected_manifest_sha256.lower()
        or receipt.get("raw_fingerprints_verified") is not True
        or int(receipt.get("studies", -1)) != 338
        or int(receipt.get("rows", -1)) != 3484
        or receipt.get("model_compute") != "none"
    ):
        raise ValueError(
            "MLS 2.5D validation receipt does not prove the required immutable cache"
        )
    return receipt, actual_sha256
