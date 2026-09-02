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
    if payload.get("status") != "ready_for_checksum_transfer":
        raise RuntimeError("Transfer manifest is not in a ready state")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Transfer manifest contains no artifacts")

    checks: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid artifact metadata: {name}")
        filename = str(metadata.get("transfer_filename", ""))
        if not filename or Path(filename).name != filename or filename in filenames:
            raise ValueError(f"Unsafe or duplicate transfer filename: {filename!r}")
        filenames.add(filename)
        local_path = artifact_dir / filename
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        actual_bytes = local_path.stat().st_size
        actual_sha = _sha256(local_path)
        expected_bytes = int(metadata["bytes"])
        expected_sha = str(metadata["sha256"])
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
        "run_name": payload.get("run_name"),
        "manifest_path": str(manifest),
        "manifest_sha256": actual_manifest_sha256,
        "artifacts_expected": len(artifacts),
        "artifacts_verified": len(checks),
        "raw_medical_predictions_included": bool(
            payload.get("raw_medical_predictions_included", False)
        ),
        "checks": checks,
    }
    if result["raw_medical_predictions_included"]:
        raise ValueError("Private raw medical predictions must not be in transfer package")
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
