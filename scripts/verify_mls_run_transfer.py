"""Verify a locally downloaded MLS run against its remote SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_RUN = re.compile(r"^[a-zA-Z0-9_.-]+$")
REQUIRED_ARTIFACT_KEYS = {
    "training_manifest",
    "launcher_status",
    "fixed_epoch_checkpoint",
    "report",
    "epoch_metrics",
    "run_log",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_transfer(
    *,
    manifest: Path,
    expected_manifest_sha256: str,
    artifact_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    artifact_dir = artifact_dir.resolve()
    output = output.resolve()
    expected_manifest_sha256 = expected_manifest_sha256.strip().lower()
    if not SHA256.fullmatch(expected_manifest_sha256):
        raise ValueError("Expected manifest SHA-256 must be 64 lowercase hex characters")
    actual_manifest_sha256 = _sha256(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Downloaded manifest checksum differs from remote checksum")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported transfer manifest schema")
    if payload.get("status") != "ready_for_checksum_transfer":
        raise RuntimeError("Transfer manifest is not in a ready state")
    run_name = str(payload.get("run_name", ""))
    if not SAFE_RUN.fullmatch(run_name):
        raise ValueError(f"Unsafe or absent run name: {run_name!r}")
    fixed_epoch = int(payload.get("fixed_audit_epoch", -1))
    expected_epochs = int(payload.get("expected_epochs", -1))
    epochs_completed = int(payload.get("epochs_completed", -1))
    last_epoch = int(payload.get("last_epoch", -1))
    if (
        fixed_epoch < 1
        or expected_epochs < fixed_epoch
        or epochs_completed != expected_epochs
        or last_epoch != expected_epochs
    ):
        raise ValueError("Invalid fixed-epoch or completed-history metadata")
    if payload.get("compute_performed") is not False:
        raise ValueError("Transfer manifest must describe metadata-only finalization")
    if payload.get("raw_medical_predictions_included") is not False:
        raise ValueError("Private raw medical predictions must not be in transfer package")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Transfer manifest contains no artifacts")
    if set(artifacts) != REQUIRED_ARTIFACT_KEYS:
        raise ValueError(
            "Transfer manifest artifact contract mismatch: "
            f"missing={sorted(REQUIRED_ARTIFACT_KEYS - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - REQUIRED_ARTIFACT_KEYS)}"
        )
    expected_filenames = {
        "training_manifest": "training_manifest.yaml",
        "launcher_status": "launcher_status.json",
        "fixed_epoch_checkpoint": f"mls_multitask_epoch_{fixed_epoch:03d}.pth",
        "report": "report.md",
        "epoch_metrics": "epoch_metrics.jsonl",
        "run_log": "run.log",
    }

    checks: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid artifact metadata: {name}")
        filename = str(metadata.get("transfer_filename", ""))
        if not filename or Path(filename).name != filename or filename in filenames:
            raise ValueError(f"Unsafe or duplicate transfer filename: {filename!r}")
        if filename != expected_filenames[name]:
            raise ValueError(
                f"Unexpected transfer filename for {name}: {filename!r}"
            )
        filenames.add(filename)
        local_path = artifact_dir / filename
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        actual_bytes = local_path.stat().st_size
        actual_sha = _sha256(local_path)
        expected_bytes = int(metadata["bytes"])
        expected_sha = str(metadata["sha256"]).lower()
        if expected_bytes < 0 or not SHA256.fullmatch(expected_sha):
            raise ValueError(f"Invalid size or SHA-256 metadata for {name}")
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise ValueError(
                f"Artifact integrity mismatch for {name}: "
                f"bytes={actual_bytes}/{expected_bytes}, sha256={actual_sha}/{expected_sha}"
            )
        checks[name] = {
            "local_path": str(local_path),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "verified": True,
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "manifest_path": str(manifest),
        "manifest_sha256": actual_manifest_sha256,
        "artifacts_expected": len(artifacts),
        "artifacts_verified": len(checks),
        "raw_medical_predictions_included": False,
        "checks": checks,
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_transfer(
        manifest=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        artifact_dir=args.artifact_dir,
        output=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "run_name": result["run_name"],
        "artifacts_verified": result["artifacts_verified"],
        "manifest_sha256": result["manifest_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
