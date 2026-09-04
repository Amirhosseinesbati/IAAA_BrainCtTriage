"""Fail fast on an MLS experiment's static, non-data contract.

This intentionally performs no model construction, CUDA work, DICOM access, or
raw/processed-data hashing.  It exists to catch source, epoch, runner, and
MLflow-shape mismatches *before* a Supervisor GPU job is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class ContractError(ValueError):
    """Raised when a declared static experiment contract cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"Contract path escapes repository root: {relative}") from exc
    return path


def _read_json(relative: str) -> dict[str, Any]:
    path = _inside_root(relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"Missing JSON contract input: {relative}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON contract input: {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract input must be an object: {relative}")
    return value


def _lookup(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ContractError(f"Missing declared JSON field: {dotted_path}")
        current = current[part]
    return current


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ContractError(message)


def _normalized_evaluator_equivalence(spec: dict[str, Any]) -> dict[str, Any]:
    candidate_path = _inside_root(str(spec["candidate"]))
    reference_path = _inside_root(str(spec["reference"]))
    marker = str(spec.get("body_marker", "from __future__ import annotations"))
    candidate = candidate_path.read_text(encoding="utf-8")
    reference = reference_path.read_text(encoding="utf-8")
    _require(marker in candidate and marker in reference, "Evaluator body marker is missing")
    candidate_body = candidate[candidate.index(marker):].rstrip() + "\n"
    reference_body = reference[reference.index(marker):].rstrip() + "\n"
    substitutions = spec.get("substitutions", [])
    _require(isinstance(substitutions, list) and substitutions, "No evaluator substitutions declared")
    for substitution in substitutions:
        actual = str(substitution["candidate"])
        expected = str(substitution["reference"])
        count = int(substitution.get("count", 1))
        _require(
            candidate_body.count(actual) == count,
            f"Evaluator diff is not uniquely declared: {actual!r}",
        )
        candidate_body = candidate_body.replace(actual, expected, count)
    _require(
        candidate_body == reference_body,
        "Evaluator body differs beyond explicitly declared contract substitutions",
    )
    return {
        "candidate": str(spec["candidate"]),
        "reference": str(spec["reference"]),
        "substitution_count": len(substitutions),
        "verified": True,
    }


def validate(contract: dict[str, Any], *, require_remote_tracking: bool) -> dict[str, Any]:
    required = {"schema_version", "campaign_id", "experiment_key", "files", "training", "audit"}
    missing = sorted(required - set(contract))
    _require(not missing, "Missing top-level contract fields: " + ", ".join(missing))
    _require(contract["schema_version"] == 1, "Unsupported static-contract schema")
    _require(bool(str(contract["campaign_id"]).strip()), "campaign_id is empty")
    _require(bool(str(contract["experiment_key"]).strip()), "experiment_key is empty")

    checked_files: list[dict[str, str]] = []
    files = contract["files"]
    _require(isinstance(files, dict) and files, "files must be a nonempty path-to-SHA mapping")
    for relative, expected in sorted(files.items()):
        path = _inside_root(str(relative))
        _require(path.is_file(), f"Pinned source is missing: {relative}")
        actual = _sha256(path)
        _require(actual == expected, f"Pinned source changed: {relative}")
        checked_files.append({"path": str(relative), "sha256": actual})

    training = contract["training"]
    audit = contract["audit"]
    for name, section in (("training", training), ("audit", audit)):
        _require(isinstance(section, dict), f"{name} must be an object")
    for key in ("fold", "seed", "fixed_epoch", "steps_per_epoch", "expected_optimizer_steps"):
        _require(key in training, f"training.{key} is required")
    _require(
        int(training["fixed_epoch"]) * int(training["steps_per_epoch"])
        == int(training["expected_optimizer_steps"]),
        "fixed_epoch * steps_per_epoch must equal expected_optimizer_steps",
    )
    for key in ("fold", "seed", "fixed_epoch", "expected_studies"):
        _require(key in audit, f"audit.{key} is required")
    for key in ("fold", "seed", "fixed_epoch"):
        _require(training[key] == audit[key], f"Training/audit mismatch for {key}")

    documents: dict[str, dict[str, Any]] = {}
    for document_name, path_key in (("training", "protocol"), ("audit", "protocol")):
        relative = str(contract[document_name][path_key])
        documents[document_name] = _read_json(relative)
    protocol_checks = contract.get("protocol_value_checks", [])
    _require(isinstance(protocol_checks, list) and protocol_checks, "No protocol value checks declared")
    checked_fields: list[dict[str, Any]] = []
    for check in protocol_checks:
        document = str(check["document"])
        _require(document in documents, f"Unknown protocol document: {document}")
        path = str(check["path"])
        actual = _lookup(documents[document], path)
        expected = check["equals"]
        _require(actual == expected, f"Protocol mismatch {document}.{path}: {actual!r} != {expected!r}")
        checked_fields.append({"document": document, "path": path, "value": actual})

    text_checks = contract.get("source_text_checks", [])
    _require(isinstance(text_checks, list) and text_checks, "No source-text checks declared")
    for check in text_checks:
        relative = str(check["path"])
        text = _inside_root(relative).read_text(encoding="utf-8")
        count = len(re.findall(str(check["pattern"]), text, flags=re.MULTILINE))
        expected_count = int(check.get("count", 1))
        _require(count == expected_count, f"Source-text contract mismatch for {relative}: {check['pattern']!r}")

    evaluator = _normalized_evaluator_equivalence(contract["evaluator_normalized_equivalence"])

    tracking_env_present = None
    if require_remote_tracking:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("DAGSHUB_TRACKING_URI")
        _require(bool(tracking_uri), "Remote MLflow tracking environment is not configured")
        _require(
            not str(tracking_uri).lower().startswith(("file:", "sqlite:")),
            "Refusing a local MLflow tracking URI",
        )
        tracking_env_present = True

    return {
        "status": "passed",
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "experiment_key": contract["experiment_key"],
        "scope": "static_source_protocol_runner_contract_only",
        "cuda_or_model_work_performed": False,
        "raw_or_processed_data_rehashed": False,
        "checked_files": checked_files,
        "checked_protocol_fields": checked_fields,
        "evaluator_normalized_equivalence": evaluator,
        "remote_tracking_environment_present": tracking_env_present,
        "declared_training": training,
        "declared_audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="Repository-relative static contract JSON")
    parser.add_argument("--receipt", help="Repository-relative JSON receipt; overwritten atomically")
    parser.add_argument("--require-remote-tracking", action="store_true")
    args = parser.parse_args()
    receipt = _inside_root(args.receipt) if args.receipt else None
    try:
        contract = _read_json(args.contract)
        result = validate(contract, require_remote_tracking=args.require_remote_tracking)
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        result = {
            "status": "failed",
            "scope": "static_source_protocol_runner_contract_only",
            "cuda_or_model_work_performed": False,
            "raw_or_processed_data_rehashed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if receipt is not None:
            _atomic_json(receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 2
    if receipt is not None:
        _atomic_json(receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
