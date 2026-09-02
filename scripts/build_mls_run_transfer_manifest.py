"""Build a strict checksum manifest for one completed MLS training run.

This is intentionally a metadata-only post-run gate.  It never loads a model
or performs inference.  A transfer manifest is emitted only when the launcher,
fixed epoch checkpoint, training history, and report all prove completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SAFE_RUN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def build_manifest(
    *,
    project_root: Path,
    run_name: str,
    training_manifest: Path,
    artifact_root: Path,
    output: Path,
    fixed_epoch: int = 15,
) -> dict[str, Any]:
    if not SAFE_RUN.fullmatch(run_name):
        raise ValueError(f"Unsafe run name: {run_name!r}")
    project_root = project_root.resolve()
    training_manifest = training_manifest.resolve()
    artifact_root = artifact_root.resolve()
    output = output.resolve()

    launcher_status = artifact_root / "launcher_status.json"
    checkpoint = (
        project_root / "models" / "checkpoints" / "mls_multitask" / run_name
        / f"mls_multitask_epoch_{fixed_epoch:03d}.pth"
    )
    report_dir = project_root / "reports" / "mls_experiments" / run_name
    report = report_dir / "report.md"
    history = report_dir / "epoch_metrics.jsonl"
    run_log = artifact_root / "run.log"
    required = {
        "training_manifest": training_manifest,
        "launcher_status": launcher_status,
        "fixed_epoch_checkpoint": checkpoint,
        "report": report,
        "epoch_metrics": history,
        "run_log": run_log,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required transfer artifacts missing: {missing}")

    launcher = _read_json(launcher_status)
    if launcher.get("status") != "completed" or int(launcher.get("exit_code", -1)) != 0:
        raise RuntimeError(f"Launcher is not successfully terminal: {launcher}")
    manifest_payload = _read_yaml(training_manifest)
    tags = manifest_payload.get("tags", {})
    training_config = manifest_payload.get("training_config", {})
    if not isinstance(tags, dict) or not isinstance(training_config, dict):
        raise ValueError("Training manifest tags/training_config must be mappings")
    manifest_checks = {
        "run_name": manifest_payload.get("run_name") == run_name,
        "task": manifest_payload.get("task") == "mls",
        "strategy": manifest_payload.get("strategy") == "mls_heatmap",
        "compute_policy_tag": tags.get("compute_policy") == "cuda_only_no_cpu_fallback",
        "launcher_compute_policy": (
            launcher.get("compute_policy") == "cuda_only_no_cpu_fallback"
        ),
        "fixed_audit_epoch": int(tags.get("fixed_audit_epoch", -1)) == fixed_epoch,
    }
    if failed := sorted(name for name, passed in manifest_checks.items() if not passed):
        raise ValueError(f"Training manifest contract failed: {failed}")
    expected_epochs = int(training_config.get("epochs", -1))
    if expected_epochs < fixed_epoch:
        raise ValueError("Training manifest epoch count is below the fixed audit epoch")
    manifest_sha = _sha256(training_manifest)
    if launcher.get("manifest_sha256") != manifest_sha:
        raise ValueError("Training manifest checksum does not match launcher status")

    report_text = report.read_text(encoding="utf-8")
    if "- Status: `completed`" not in report_text:
        raise RuntimeError("Training report does not prove completed status")
    rows = [
        json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [int(float(row["epoch"])) for row in rows]
    expected_epoch_sequence = list(range(1, expected_epochs + 1))
    if epochs != expected_epoch_sequence:
        raise ValueError(
            "Epoch history does not match the complete locked training schedule: "
            f"observed={epochs}, expected={expected_epoch_sequence}"
        )

    transfer_filenames = {
        "training_manifest": "training_manifest.yaml",
        "launcher_status": "launcher_status.json",
        "fixed_epoch_checkpoint": checkpoint.name,
        "report": "report.md",
        "epoch_metrics": "epoch_metrics.jsonl",
        "run_log": "run.log",
    }
    artifacts = {
        name: {
            "source_path": str(path),
            "transfer_filename": transfer_filenames[name],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in required.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready_for_checksum_transfer",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "fixed_audit_epoch": fixed_epoch,
        "expected_epochs": expected_epochs,
        "epochs_completed": len(epochs),
        "last_epoch": epochs[-1],
        "fold": int(training_config.get("fold", -1)),
        "seed": int(training_config.get("seed", -1)),
        "compute_performed": False,
        "raw_medical_predictions_included": False,
        "artifacts": artifacts,
    }
    _atomic_json(output, payload)
    payload["manifest_path"] = str(output)
    payload["manifest_sha256"] = _sha256(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-epoch", type=int, default=15)
    args = parser.parse_args()
    result = build_manifest(
        project_root=args.project_root,
        run_name=args.run_name,
        training_manifest=args.training_manifest,
        artifact_root=args.artifact_root,
        output=args.output,
        fixed_epoch=args.fixed_epoch,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
