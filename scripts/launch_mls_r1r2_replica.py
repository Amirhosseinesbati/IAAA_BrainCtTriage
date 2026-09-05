"""Fail-closed launcher for one sealed R1R2 CUDA training replica.

The Supervisor service must never invoke the trainer directly.  This small
wrapper first validates the checksum-bound R1R2 contract, the exact manifest,
the selected checkout and the selected project configuration.  Only then does
it ``exec`` the ordinary GPU-only training command.  It never loads a model or
decodes DICOM itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_mls_r1_replication_matrix import ARMS, FIXED_EPOCH, FIXED_FOLD
from scripts.validate_mls_r1_replication_matrix import validate_contract


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_contract_sha256(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"R1R2 contract SHA companion is unreadable: {path}") from exc
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("R1R2 contract SHA companion is not one lowercase SHA-256")
    return value


def _require_sealed_environment(contract: dict[str, Any]) -> Path:
    source_root = Path(str(contract["training_source"]["root"])).resolve()
    if PROJECT_ROOT.resolve() != source_root:
        raise ValueError("R1R2 launcher itself is not executing from the sealed source checkout")
    configured_root = os.environ.get("IAAA_PROJECT_ROOT")
    if not configured_root or Path(configured_root).resolve() != source_root:
        raise ValueError("IAAA_PROJECT_ROOT must equal the R1R2 sealed source checkout")
    selected_config = os.environ.get("IAAA_CONFIG_PATH")
    expected_config = Path(str(contract["data"]["project_config"])).resolve()
    if not selected_config or Path(selected_config).resolve() != expected_config:
        raise ValueError("IAAA_CONFIG_PATH must equal the R1R2 sealed project configuration")
    if _sha256(expected_config) != contract["data"]["project_config_sha256"]:
        raise ValueError("R1R2 sealed project configuration hash differs before training")
    return source_root


def _require_planned_member(
    *, contract: dict[str, Any], arm: str, seed: int,
) -> tuple[Path, str]:
    member = contract["members"][arm][f"seed{seed}"]
    if member.get("member_kind") != "planned_r1r_replica":
        raise ValueError("only a planned R1R2 replica may be launched")
    manifest = Path(str(member.get("config_path", ""))).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"sealed R1R2 manifest is missing: {manifest}")
    if _sha256(manifest) != member.get("config_sha256"):
        raise ValueError("sealed R1R2 manifest hash differs before training")
    try:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"sealed R1R2 manifest is unreadable: {manifest}") from exc
    if not isinstance(payload, dict):
        raise ValueError("sealed R1R2 manifest must be a YAML object")
    expected_run_name = f"mls-r1r2-reflection-{ARMS[arm]['slug']}-fold{FIXED_FOLD}-seed{seed}"
    training = payload.get("training_config")
    tags = payload.get("tags")
    checks = {
        "run_name": payload.get("run_name") == expected_run_name,
        "training_config": isinstance(training, dict),
        "tags": isinstance(tags, dict),
        "fold": isinstance(training, dict) and int(training.get("fold", -1)) == FIXED_FOLD,
        "seed": isinstance(training, dict) and int(training.get("seed", -1)) == seed,
        "no_resume": isinstance(training, dict) and training.get("resume_checkpoint") is None,
        "flip": isinstance(training, dict) and float(training.get("horizontal_flip_prob", -1.0)) == float(ARMS[arm]["horizontal_flip_prob"]),
        "campaign": isinstance(tags, dict) and tags.get("campaign_id") == "mls_reflection_r1r2_20260905",
        "arm": isinstance(tags, dict) and tags.get("arm") == arm,
        "fixed_epoch": isinstance(tags, dict) and int(tags.get("fixed_audit_epoch", -1)) == FIXED_EPOCH,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"sealed R1R2 replica manifest is incompatible: {failed}")
    return manifest, expected_run_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256-file", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=(2026, 3407), required=True)
    parser.add_argument("--allow-training", action="store_true")
    args = parser.parse_args()
    if not args.allow_training:
        raise ValueError("R1R2 replica launcher requires explicit --allow-training")
    contract_path = args.contract.resolve()
    expected_sha = _read_contract_sha256(args.contract_sha256_file.resolve())
    # This is static provenance/config validation only; no checkpoint is loaded
    # because a new replica does not exist before the training child starts.
    validate_contract(
        contract_path,
        expected_contract_sha256=expected_sha,
        require_checkpoints=False,
    )
    contract = _load_json(contract_path, label="R1R2 contract")
    source_root = _require_sealed_environment(contract)
    manifest, run_name = _require_planned_member(contract=contract, arm=args.arm, seed=args.seed)
    command = [
        sys.executable,
        str(source_root / "scripts" / "run_vast_mls_experiment.py"),
        "--manifest", str(manifest), "--allow-training",
    ]
    print(json.dumps({
        "status": "validated_before_cuda_training",
        "arm": args.arm,
        "seed": args.seed,
        "run_name": run_name,
        "manifest_sha256": _sha256(manifest),
        "contract_sha256": expected_sha,
    }, sort_keys=True), flush=True)
    completed = subprocess.run(command, cwd=source_root, check=False, env=dict(os.environ))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
