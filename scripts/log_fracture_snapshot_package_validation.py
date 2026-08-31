"""Verify a fracture snapshot package and log only aggregate validation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from scripts.package_fracture_mil_candidate import _sha256


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--dicom-benchmark", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.package / "manifest.json"
    manifest = _load(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Package manifest does not contain artifacts")
    verified_bytes = 0
    for relative, expected in artifacts.items():
        if not isinstance(expected, dict):
            raise TypeError(f"Invalid artifact manifest entry: {relative}")
        path = args.package / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if actual_hash != str(expected["sha256"]):
            raise RuntimeError(f"Package artifact checksum mismatch: {relative}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Package artifact size mismatch: {relative}")
        verified_bytes += path.stat().st_size

    benchmark = _load(args.benchmark)
    dicom_benchmark = _load(args.dicom_benchmark)
    if benchmark.get("parity") != "passed" or dicom_benchmark.get("parity") != "passed":
        raise RuntimeError("A package benchmark did not pass parity")
    studies = dicom_benchmark.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError("DICOM benchmark has no studies")
    max_image_difference = max(
        int(row["preprocessing"]["image_max_abs_difference"])
        for row in studies
    )
    max_component_error = max(
        max(
            float(row[key])
            for key in (
                "assigned_fold_adjacent_error",
                "assigned_fold_mil_error",
                "assigned_fold_blend_error",
                "assigned_fold_snapshot_raw_error",
                "assigned_fold_snapshot_cdf_error",
                "assigned_fold_fusion_error",
                "assigned_fold_decision_error",
            )
        )
        for row in studies
    )
    runtime = dicom_benchmark["runtime"]
    if not isinstance(runtime, dict):
        raise TypeError("Benchmark runtime must be an object")
    summary = {
        "run_id": args.run_id,
        "candidate": manifest["candidate"],
        "schema_version": manifest["schema_version"],
        "manifest_sha256": _sha256(manifest_path),
        "artifact_count": len(artifacts),
        "artifact_bytes": verified_bytes,
        "artifact_checksums_verified": True,
        "parity": "passed",
        "benchmark_study_count": len(studies),
        "maximum_component_absolute_error": max_component_error,
        "dicom_image_maximum_absolute_difference": max_image_difference,
        "runtime_mean_seconds": float(runtime["mean_seconds"]),
        "runtime_worst_seconds": float(runtime["worst_seconds"]),
        "projected_68_studies_mean_seconds": float(
            runtime["projected_68_studies_mean_seconds"]
        ),
        "peak_gpu_bytes": int(runtime["peak_gpu_bytes"]),
        "private_study_rows_logged": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "package_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(
            {
                "package_artifact_count": float(summary["artifact_count"]),
                "package_artifact_bytes": float(summary["artifact_bytes"]),
                "package_maximum_component_error": max_component_error,
                "package_dicom_image_maximum_difference": float(
                    max_image_difference
                ),
                "package_runtime_mean_seconds": float(runtime["mean_seconds"]),
                "package_runtime_worst_seconds": float(runtime["worst_seconds"]),
                "package_peak_gpu_bytes": float(runtime["peak_gpu_bytes"]),
            }
        )
        mlflow.set_tags(
            {
                "package_validation": "passed",
                "package_checksums_verified": "true",
                "package_dicom_preprocessing_parity": "passed",
                "package_private_study_rows_logged": "false",
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="package_validation")
        mlflow.log_artifact(str(manifest_path), artifact_path="package_validation")
        mlflow.log_artifact(
            str(args.package / "README.md"), artifact_path="package_validation"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
