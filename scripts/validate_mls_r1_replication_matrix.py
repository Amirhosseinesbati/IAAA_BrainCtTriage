"""Fail-closed static and checkpoint validation for the immutable R1R matrix.

This validator is deliberately independent of model inference.  Before a
replica training starts it validates the sealed continuation contract, its
parent R1 evidence and all YAML recipes.  With ``--require-checkpoints`` it
also validates the six epoch-15 checkpoint metadata/configurations, but never
runs a model forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.materialize_mls_r1_replication_matrix import (
    ARMS,
    AUDIT_SEEDS,
    FIXED_EPOCH,
    FIXED_FOLD,
    INHERITED_SEED,
    _canonical_sha256,
    _field_differences,
    _fold_roster,
    _load_json,
    _load_yaml_manifest,
    _model_signature,
    _normalise_sha256,
    _require_file_sha256,
    _source_hashes,
)

# ``sha256_file`` is intentionally imported separately: keeping the materializer
# import list declarative makes the model-free validation surface obvious.
from src.strategies.mls_heatmap.context_cache import sha256_file
from src.strategies.config_models import MLSHeatmapConfig


EXPECTED_STATUS = "locked_after_passed_r1_mls_screen_before_r1r_replication_cuda"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_contract_sha(path: Path, expected: str | None) -> str:
    observed = sha256_file(path)
    if expected is not None and observed != _normalise_sha256(expected, label="R1R contract"):
        raise ValueError("R1R continuation contract checksum differs")
    return observed


def _validate_cache_receipt(path: Path, expected_sha: str, expected_manifest: str) -> None:
    _require_file_sha256(path, expected_sha, label="R1R cache validation receipt")
    receipt = _load_json(path, label="R1R cache validation receipt")
    if receipt.get("status") != "passed" or receipt.get("cache_manifest_sha256") != expected_manifest:
        raise ValueError("R1R cache validation receipt is not bound to the locked cache")


def _validate_parent_evidence(contract: dict[str, Any]) -> None:
    parent = _require_mapping(contract.get("parent_r1"), label="parent_r1")
    parent_path = Path(str(parent.get("preregistration", ""))).resolve()
    parent_sha = _normalise_sha256(parent.get("preregistration_sha256", ""), label="parent R1 preregistration")
    _require_file_sha256(parent_path, parent_sha, label="parent R1 preregistration")
    document = _load_json(parent_path, label="parent R1 preregistration")
    checks = {
        "status": document.get("status") == "locked_before_any_r1_cuda_outcome",
        "fold": int(document.get("fold", -1)) == FIXED_FOLD,
        "seed": int(document.get("seed", -1)) == INHERITED_SEED,
        "epoch": int(document.get("fixed_audit_epoch", -1)) == FIXED_EPOCH,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"parent R1 preregistration is incompatible: {failed}")
    screen_path = Path(str(parent.get("paired_mls_screen_gate", ""))).resolve()
    _require_file_sha256(
        screen_path,
        str(parent.get("paired_mls_screen_gate_sha256", "")),
        label="parent paired MLS screen gate",
    )
    screen = _load_json(screen_path, label="parent paired MLS screen gate")
    expected_evaluation_sha = {
        "control": parent.get("control_strict_evaluation_sha256"),
        "candidate": parent.get("candidate_strict_evaluation_sha256"),
    }
    checks = {
        "status": screen.get("status") == "passed",
        "authorization": screen.get("next_gate_authorized") is True,
        "not_promoted": screen.get("promotion_eligible") is False,
        "not_zip": screen.get("submission_zip_allowed") is False,
        "parent": screen.get("preregistration_sha256") == parent_sha,
        "control": screen.get("control_receipt_sha256") == expected_evaluation_sha["control"],
        "candidate": screen.get("candidate_receipt_sha256") == expected_evaluation_sha["candidate"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"parent paired MLS screen evidence is incompatible: {failed}")


def _member_config(
    contract: dict[str, Any], *, arm: str, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    members = _require_mapping(contract.get("members"), label="members")
    arm_members = _require_mapping(members.get(arm), label=f"members.{arm}")
    member = _require_mapping(arm_members.get(f"seed{seed}"), label=f"members.{arm}.seed{seed}")
    path = Path(str(member.get("config_path", ""))).resolve()
    _require_file_sha256(path, str(member.get("config_sha256", "")), label=f"R1R {arm}/seed{seed} YAML")
    _, config = _load_yaml_manifest(path)
    if int(config["fold"]) != FIXED_FOLD or int(config["seed"]) != seed:
        raise ValueError(f"R1R {arm}/seed{seed} YAML has wrong fold/seed")
    if config.get("resume_checkpoint") is not None:
        raise ValueError(f"R1R {arm}/seed{seed} must not resume from a checkpoint")
    arms = _require_mapping(contract.get("arms"), label="arms")
    arm_contract = _require_mapping(arms.get(arm), label=f"arms.{arm}")
    signature = _model_signature(config)
    if signature != arm_contract.get("model_config_signature"):
        raise ValueError(f"R1R {arm}/seed{seed} differs from its arm signature")
    if _canonical_sha256(signature) != arm_contract.get("model_config_signature_sha256"):
        raise ValueError(f"R1R {arm}/seed{seed} arm signature checksum differs")
    return member, config, path


def _validate_config_family(contract: dict[str, Any]) -> dict[str, dict[int, tuple[dict[str, Any], Path]]]:
    arms = _require_mapping(contract.get("arms"), label="arms")
    if set(arms) != set(ARMS):
        raise ValueError("R1R contract must contain exactly control and candidate arms")
    collected: dict[str, dict[int, tuple[dict[str, Any], Path]]] = {}
    for arm in ARMS:
        per_seed: dict[int, tuple[dict[str, Any], Path]] = {}
        configs: dict[int, dict[str, Any]] = {}
        for seed in AUDIT_SEEDS:
            member, config, path = _member_config(contract, arm=arm, seed=seed)
            expected_kind = "inherited_r1_seed42" if seed == INHERITED_SEED else "planned_r1r_replica"
            if member.get("member_kind") != expected_kind:
                raise ValueError(f"R1R {arm}/seed{seed} has an unexpected member kind")
            if seed == INHERITED_SEED:
                if "checkpoint_path" not in member or "checkpoint_sha256" not in member:
                    raise ValueError(f"R1R inherited {arm}/seed42 lacks checkpoint provenance")
            else:
                expected_checkpoint = Path(str(member.get("expected_checkpoint_path", ""))).resolve()
                if expected_checkpoint.name != f"mls_multitask_epoch_{FIXED_EPOCH:03d}.pth":
                    raise ValueError(f"R1R planned {arm}/seed{seed} checkpoint path is not epoch {FIXED_EPOCH}")
            per_seed[seed] = (config, path)
            configs[seed] = config
        reference = configs[INHERITED_SEED]
        for seed in (2026, 3407):
            if _field_differences(reference, configs[seed]) != ["seed"]:
                raise ValueError(f"R1R {arm} configs differ by more than seed")
        collected[arm] = per_seed
    for seed in AUDIT_SEEDS:
        control_signature = _model_signature(collected["control"][seed][0])
        candidate_signature = _model_signature(collected["candidate"][seed][0])
        if _field_differences(control_signature, candidate_signature) != ["horizontal_flip_prob"]:
            raise ValueError(f"R1R arms differ by more than horizontal_flip_prob for seed{seed}")
        expected_control = float(ARMS["control"]["horizontal_flip_prob"])
        expected_candidate = float(ARMS["candidate"]["horizontal_flip_prob"])
        if (
            float(control_signature["horizontal_flip_prob"]) != expected_control
            or float(candidate_signature["horizontal_flip_prob"]) != expected_candidate
        ):
            raise ValueError("R1R horizontal flip probabilities differ from the locked intervention")
    return collected


def _validate_checkpoint(
    *, member: dict[str, Any], config: dict[str, Any], source_hashes: dict[str, str],
    expected_path: Path, label: str,
) -> dict[str, Any]:
    """Validate metadata only; no model is instantiated or executed."""
    if not expected_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {expected_path}")
    import torch

    payload = torch.load(expected_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} checkpoint payload is invalid")
    if int(payload.get("epoch", -1)) != FIXED_EPOCH:
        raise ValueError(f"{label} checkpoint is not fixed epoch {FIXED_EPOCH}")
    observed_config = MLSHeatmapConfig.model_validate(payload.get("config")).model_dump(mode="json")
    if observed_config != config:
        raise ValueError(f"{label} checkpoint config differs from locked YAML")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_sha256"), dict):
        raise ValueError(f"{label} checkpoint lacks source provenance")
    mismatched = [
        name for name, digest in source_hashes.items()
        if provenance["source_sha256"].get(name) != digest
    ]
    if mismatched:
        raise ValueError(f"{label} checkpoint source provenance differs: {sorted(mismatched)}")
    observed_sha = sha256_file(expected_path)
    inherited_sha = member.get("checkpoint_sha256")
    if inherited_sha is not None and observed_sha != inherited_sha:
        raise ValueError(f"{label} inherited checkpoint checksum differs")
    return {
        "path": str(expected_path.resolve()), "sha256": observed_sha,
        "bytes": expected_path.stat().st_size, "epoch": FIXED_EPOCH,
        "seed": int(config["seed"]),
    }


def validate_contract(
    contract_path: Path, *, expected_contract_sha256: str | None = None,
    require_checkpoints: bool = False,
) -> dict[str, Any]:
    """Validate an R1R contract and optionally all member checkpoint metadata."""
    contract_path = contract_path.resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"R1R contract is missing: {contract_path}")
    contract_sha = _require_contract_sha(contract_path, expected_contract_sha256)
    contract = _load_json(contract_path, label="R1R continuation contract")
    if contract.get("status") != EXPECTED_STATUS:
        raise ValueError("R1R continuation contract is not locked in the required state")
    protocol = _require_mapping(contract.get("protocol"), label="protocol")
    checks = {
        "fold": int(protocol.get("fold", -1)) == FIXED_FOLD,
        "studies": int(protocol.get("studies", -1)) == 67,
        "epoch": int(protocol.get("fixed_epoch", -1)) == FIXED_EPOCH,
        "seeds": list(protocol.get("seeds", [])) == list(AUDIT_SEEDS),
        "new_trainings": int(protocol.get("new_cuda_trainings", -1)) == 4,
        "within_arm": protocol.get("within_arm_training_config_differences") == ["seed"],
        "cross_arm": protocol.get("cross_arm_training_config_differences") == ["horizontal_flip_prob"],
        "not_promoted": protocol.get("promotion_eligible") is False,
        "not_zip": protocol.get("submission_zip_allowed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"R1R protocol is incompatible: {failed}")
    source = _require_mapping(contract.get("training_source"), label="training_source")
    source_root = Path(str(source.get("root", ""))).resolve()
    observed_source_hashes = _source_hashes(source_root)
    if observed_source_hashes != source.get("core_source_sha256"):
        raise ValueError("R1R training source hashes differ from the locked contract")
    data = _require_mapping(contract.get("data"), label="data")
    _require_file_sha256(
        Path(str(data.get("fold_manifest", ""))).resolve(),
        str(data.get("fold_manifest_sha256", "")), label="R1R fold manifest",
    )
    studies, roster_sha = _fold_roster(source_root)
    if studies != 67 or roster_sha != protocol.get("fold1_roster_sha256"):
        raise ValueError("R1R fold-1 roster differs from the locked contract")
    _require_file_sha256(
        Path(str(data.get("truth_table", ""))).resolve(),
        str(data.get("truth_table_sha256", "")), label="R1R truth table",
    )
    _require_file_sha256(
        Path(str(data.get("frozen_champion_predictions", ""))).resolve(),
        str(data.get("frozen_champion_predictions_sha256", "")), label="R1R frozen Champion predictions",
    )
    _validate_cache_receipt(
        Path(str(data.get("cache_validation_receipt", ""))).resolve(),
        str(data.get("cache_validation_receipt_sha256", "")),
        str(data.get("cache_manifest_sha256", "")),
    )
    _validate_parent_evidence(contract)
    configurations = _validate_config_family(contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "phase": contract["phase"],
        "compute_policy": "metadata_only_no_model_forward",
        "require_checkpoints": require_checkpoints,
        "validated_at_utc": _utc_now(),
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    if require_checkpoints:
        observed_members: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            arm_members = _require_mapping(contract["members"][arm], label=f"members.{arm}")
            for seed in AUDIT_SEEDS:
                member = _require_mapping(arm_members[f"seed{seed}"], label=f"members.{arm}.seed{seed}")
                config, _ = configurations[arm][seed]
                expected_path = Path(str(
                    member.get("checkpoint_path") if seed == INHERITED_SEED
                    else member.get("expected_checkpoint_path")
                )).resolve()
                observed_members[arm][f"seed{seed}"] = _validate_checkpoint(
                    member=member, config=config, source_hashes=observed_source_hashes,
                    expected_path=expected_path, label=f"R1R {arm}/seed{seed}",
                )
        result["checkpoint_members"] = observed_members
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite R1R validation receipt: {args.output}")
    result = validate_contract(
        args.contract,
        expected_contract_sha256=args.contract_sha256,
        require_checkpoints=args.require_checkpoints,
    )
    if args.output is not None:
        _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "contract_sha256": result["contract_sha256"],
        "require_checkpoints": result["require_checkpoints"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
