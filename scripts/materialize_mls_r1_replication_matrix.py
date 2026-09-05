"""Lock the conditional R1 fold-1 three-seed replication before CUDA work.

R1's original matrix intentionally covered only fold 1 / seed 42.  It must
not be edited retrospectively after its outcome.  This materializer creates a
new immutable continuation contract for the four still-unseen replicas:

* control / candidate x seeds 2026 and 3407;
* inherited, checksum-bound seed-42 members for both arms.

It performs only metadata, YAML and hash validation; it never loads a model or
raw DICOM.  The resulting contract records the exact future CUDA recipe and
the frozen triage inputs that a later development-only gate must consume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file


FIXED_FOLD = 1
FIXED_EPOCH = 15
INHERITED_SEED = 42
REPLICA_SEEDS = (2026, 3407)
AUDIT_SEEDS = (INHERITED_SEED, *REPLICA_SEEDS)
AUDIT_BATCH_SIZE = 8
ARMS = {
    "control": {"slug": "control", "horizontal_flip_prob": 0.0},
    "candidate": {"slug": "reflect", "horizontal_flip_prob": 0.5},
}
CORE_SOURCE_RELATIVE_PATHS = {
    "config_models": "src/strategies/config_models.py",
    "dataset": "src/strategies/mls_heatmap/dataset.py",
    "train_multitask": "src/strategies/mls_heatmap/train_multitask.py",
    "predict_multitask": "src/strategies/mls_heatmap/predict_multitask.py",
    "model": "src/strategies/mls_heatmap/model.py",
    "mls_utils": "src/strategies/mls_heatmap/utils.py",
    "input_contract": "src/strategies/mls_heatmap/input_contract.py",
    "fold_manifest": "config/folds.csv",
}
# R1's inherited seed-42 receipts can only prove the historical training
# surface above.  R1R2 additionally seals every mutable dependency used after
# training: raw-DICOM inference, fold selection, configuration resolution and
# canonical triage.  Keep this list explicit rather than pretending a git
# revision alone protects a dirty remote checkout.
AUDIT_SOURCE_RELATIVE_PATHS = {
    "three_seed_cuda_evaluator": "scripts/evaluate_mls_three_seed_fold_cuda.py",
    "replica_launch_wrapper": "scripts/launch_mls_r1r2_replica.py",
    "mlflow_environment_wrapper": "scripts/run_with_mls_mlflow_env.sh",
    "development_triage_evaluator": "scripts/evaluate_mls_r1r_fold1_development_triage.py",
    "development_gate_runner": "scripts/run_vast_mls_r1r_three_seed_development_gate.sh",
    "canonical_triage_evaluator": "scripts/evaluate_mls_deploy_aligned_seed_medians.py",
    "config_loader": "src/config.py",
    "project_config": "config/project.yaml",
    "fold_split_loader": "src/evaluation/splits.py",
    "fold_validator": "src/evaluation/folds.py",
    "triage_rules": "src/evaluation/triage.py",
    "dicom_reader": "src/preprocessing/core/dicom_reader.py",
    "multitask_predictor": "src/strategies/mls_heatmap/predict_multitask.py",
}
R1R2_STATUS = "locked_after_passed_r1_mls_screen_before_r1r2_replication_cuda"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _normalise_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_file_sha256(path: Path, expected: str, *, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if observed != _normalise_sha256(expected, label=label):
        raise ValueError(f"{label} checksum differs from its locked value")
    return observed


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_yaml_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"locked manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("training_config"), dict):
        raise ValueError(f"locked manifest lacks training_config: {path}")
    config = MLSHeatmapConfig.model_validate(payload["training_config"]).model_dump(mode="json")
    return payload, config


def _model_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"fold", "seed"}}


def _field_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in left.keys() | right.keys()
        if left.get(key) != right.get(key)
    )


def _git_commit(source_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to resolve source commit under {source_root}") from exc


def _source_hashes(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in CORE_SOURCE_RELATIVE_PATHS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"training source file is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _audit_source_hashes(source_root: Path) -> dict[str, str]:
    """Hash the exact code/config surface used by raw-DICOM and triage gates."""
    result: dict[str, str] = {}
    for name, relative in AUDIT_SOURCE_RELATIVE_PATHS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"R1R2 audit source file is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _raw_dicom_binding(raw_dicom_root: Path, cache_receipt: dict[str, Any]) -> dict[str, Any]:
    """Bind raw-root identity to the existing validated cache receipt.

    A fresh whole-tree DICOM hash would violate the campaign's no-heavy-CPU
    policy.  The pre-existing cache receipt already records a full raw
    fingerprint validation, so R1R2 verifies that receipt, its raw byte count,
    and the exact resolved root used by CUDA inference instead.
    """
    resolved = raw_dicom_root.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"R1R2 raw DICOM root is not a directory: {resolved}")
    if cache_receipt.get("raw_fingerprints_verified") is not True:
        raise ValueError("cache validation receipt does not verify raw DICOM fingerprints")
    raw_bytes = int(cache_receipt.get("raw_dicom_bytes", 0))
    if raw_bytes <= 0:
        raise ValueError("cache validation receipt has no positive raw DICOM byte count")
    return {
        "root": str(raw_dicom_root),
        "resolved_root": str(resolved),
        "identity_protocol": "resolved_root_plus_prevalidated_cache_fingerprint_no_fresh_whole_tree_rehash",
        "cache_receipt_raw_fingerprints_verified": True,
        "cache_receipt_raw_dicom_bytes": raw_bytes,
    }


def _fold_roster(source_root: Path) -> tuple[int, str]:
    folds_path = source_root / CORE_SOURCE_RELATIVE_PATHS["fold_manifest"]
    frame = pd.read_csv(folds_path, dtype={"study_id": str, "patient_id": str})
    required = {"study_id", "patient_id", "triage_class", "fold"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"fold manifest lacks columns: {missing}")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise")
    heldout = frame.loc[frame["fold"] == FIXED_FOLD, [
        "study_id", "patient_id", "triage_class",
    ]].copy()
    heldout["study_id"] = heldout["study_id"].astype(str)
    heldout["patient_id"] = heldout["patient_id"].astype(str)
    heldout["triage_class"] = pd.to_numeric(heldout["triage_class"], errors="raise").astype(int)
    if len(heldout) < 1 or heldout["study_id"].duplicated().any():
        raise ValueError("fold-1 roster is empty or contains duplicate study IDs")
    records = heldout.sort_values("study_id").to_dict(orient="records")
    return len(heldout), _canonical_sha256(records)


def _validate_truth(path: Path, expected_sha256: str, *, studies: int) -> str:
    observed = _require_file_sha256(path, expected_sha256, label="truth table")
    truth = pd.read_csv(path, dtype={"dicom_series.id": str})
    required = {"dicom_series.id", "MLS_mm", "fracture_prob", "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"}
    if missing := sorted(required - set(truth.columns)):
        raise ValueError(f"truth table lacks canonical triage fields: {missing}")
    if truth["dicom_series.id"].duplicated().any() or len(truth) < studies:
        raise ValueError("truth table cannot provide a unique fold-1 context")
    return observed


def _validate_parent_matrix(
    *, parent_preregistration: Path, parent_sha256: str, parent_matrix_dir: Path,
    source_root: Path, cache_manifest_sha256: str, cache_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require_file_sha256(parent_preregistration, parent_sha256, label="parent R1 preregistration")
    document = _load_json(parent_preregistration, label="parent R1 preregistration")
    checks = {
        "status": document.get("status") == "locked_before_any_r1_cuda_outcome",
        "campaign": document.get("campaign") == "mls_reflection_paired",
        "fold": int(document.get("fold", -1)) == FIXED_FOLD,
        "seed": int(document.get("seed", -1)) == INHERITED_SEED,
        "fixed_epoch": int(document.get("fixed_audit_epoch", -1)) == FIXED_EPOCH,
        "cache_manifest": document.get("cache_manifest_sha256") == cache_manifest_sha256,
        "cache_receipt": document.get("cache_validation_receipt_sha256") == cache_receipt_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"parent R1 preregistration contract failed: {failed}")
    parent_sources = document.get("source_sha256")
    if not isinstance(parent_sources, dict):
        raise ValueError("parent R1 preregistration lacks source hashes")
    current_sources = _source_hashes(source_root)
    mismatched_sources = [
        name for name, observed in current_sources.items()
        if parent_sources.get(name) != observed
    ]
    if mismatched_sources:
        raise ValueError(
            "continuation training source differs from inherited seed-42 source: "
            + ", ".join(sorted(mismatched_sources))
        )

    configs = document.get("configs")
    arms = document.get("arms")
    if not isinstance(configs, list) or not isinstance(arms, dict):
        raise ValueError("parent R1 preregistration lacks arm configs")
    entries = {str(item.get("arm")): item for item in configs if isinstance(item, dict)}
    if set(entries) != set(ARMS):
        raise ValueError("parent R1 preregistration must contain control and candidate configs")
    payloads: dict[str, dict[str, Any]] = {}
    normalised: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        entry = entries[arm]
        path = (parent_matrix_dir / str(entry.get("file", ""))).resolve()
        _require_file_sha256(path, str(entry.get("sha256", "")), label=f"parent {arm} YAML")
        payload, config = _load_yaml_manifest(path)
        signature = _model_signature(config)
        expected_arm = arms.get(arm)
        if not isinstance(expected_arm, dict):
            raise ValueError(f"parent R1 arm is invalid: {arm}")
        if signature != expected_arm.get("model_config_signature"):
            raise ValueError(f"parent {arm} YAML differs from its locked signature")
        if _canonical_sha256(signature) != expected_arm.get("model_config_signature_sha256"):
            raise ValueError(f"parent {arm} signature checksum differs")
        if int(config["fold"]) != FIXED_FOLD or int(config["seed"]) != INHERITED_SEED:
            raise ValueError(f"parent {arm} YAML has wrong fold/seed")
        payloads[arm] = payload
        normalised[arm] = config
        signatures[arm] = signature
    if _field_differences(signatures["control"], signatures["candidate"]) != ["horizontal_flip_prob"]:
        raise ValueError("parent R1 arms do not differ only by horizontal_flip_prob")
    return document, entries, payloads, normalised


def _validate_evaluation_receipt(
    path: Path, *, arm: str, parent_sha256: str, cache_receipt_sha256: str,
    truth_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt = _load_json(path, label=f"parent {arm} raw-DICOM receipt")
    required = {
        "status", "campaign", "scope", "arm", "fold", "seed", "studies",
        "fixed_epoch", "checkpoint", "checkpoint_sha256", "preregistration_sha256",
        "cache_validation_receipt_sha256", "truth_table_sha256", "private_predictions_sha256",
    }
    if missing := sorted(required - set(receipt)):
        raise ValueError(f"parent {arm} raw-DICOM receipt lacks fields: {missing}")
    checks = {
        "status": receipt["status"] == "completed",
        "campaign": receipt["campaign"] == "mls_reflection_paired",
        "scope": receipt["scope"] == "raw_dicom_single_fold_mls_screen_only",
        "arm": receipt["arm"] == arm,
        "fold": int(receipt["fold"]) == FIXED_FOLD,
        "seed": int(receipt["seed"]) == INHERITED_SEED,
        "studies": int(receipt["studies"]) == 67,
        "epoch": int(receipt["fixed_epoch"]) == FIXED_EPOCH,
        "parent": receipt["preregistration_sha256"] == parent_sha256,
        "cache": receipt["cache_validation_receipt_sha256"] == cache_receipt_sha256,
        "truth": receipt["truth_table_sha256"] == truth_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"parent {arm} raw-DICOM receipt failed validation: {failed}")
    checkpoint = Path(str(receipt["checkpoint"])).resolve()
    _require_file_sha256(checkpoint, str(receipt["checkpoint_sha256"]), label=f"parent {arm} checkpoint")
    return receipt, sha256_file(path)


def _validate_parity_receipt(path: Path, *, arm: str) -> tuple[dict[str, Any], str]:
    receipt = _load_json(path, label=f"parent {arm} package-parity receipt")
    if receipt.get("status") != "passed":
        raise ValueError(f"parent {arm} package-parity receipt did not pass")
    observed_arm = receipt.get("arm")
    if observed_arm is not None and observed_arm != arm:
        raise ValueError(f"parent {arm} package-parity receipt belongs to {observed_arm!r}")
    return receipt, sha256_file(path)


def _validate_screen_gate(
    path: Path, *, parent_sha256: str, control_receipt_sha256: str,
    candidate_receipt_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt = _load_json(path, label="parent paired MLS screen gate")
    checks = {
        "status": receipt.get("status") == "passed",
        "authorization": receipt.get("next_gate_authorized") is True,
        "not_promoted": receipt.get("promotion_eligible") is False,
        "not_zip": receipt.get("submission_zip_allowed") is False,
        "parent": receipt.get("preregistration_sha256") == parent_sha256,
        "control": receipt.get("control_receipt_sha256") == control_receipt_sha256,
        "candidate": receipt.get("candidate_receipt_sha256") == candidate_receipt_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"parent paired MLS screen gate failed validation: {failed}")
    return receipt, sha256_file(path)


def _replica_manifest(
    parent_payload: dict[str, Any], *, arm: str, seed: int,
    parent_sha256: str, cache_manifest_sha256: str, cache_receipt_sha256: str,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = copy.deepcopy(parent_payload)
    training_config = dict(payload["training_config"])
    training_config.update({"fold": FIXED_FOLD, "seed": seed, "resume_checkpoint": None})
    config = MLSHeatmapConfig.model_validate(training_config).model_dump(mode="json")
    payload["training_config"] = config
    slug = ARMS[arm]["slug"]
    # Never reuse the partially stopped R1R run names: no non-fixed snapshot
    # from the invalidated contract may become an accidental resume source.
    run_name = f"mls-r1r2-reflection-{slug}-fold{FIXED_FOLD}-seed{seed}"
    payload["run_name"] = run_name
    payload["notes"] = (
        "R1R2 conditional three-seed replication. This immutable manifest is paired "
        "with inherited R1 seed-42 evidence; training config differs from its arm's "
        "other seeds only by seed, and the two arms differ only by horizontal flip."
    )
    tags = dict(payload.get("tags") or {})
    tags.update({
        "campaign_id": "mls_reflection_r1r2_20260905",
        "experiment_key": "R1R2",
        "phase": "conditional_fold1_three_seed_replication_after_contract_hardening",
        "parent_r1_preregistration_sha256": parent_sha256,
        "arm": arm,
        "fold": FIXED_FOLD,
        "seed": seed,
        "horizontal_flip_prob": ARMS[arm]["horizontal_flip_prob"],
        "changed_training_factor": "horizontal_flip_prob_only",
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_validation_receipt_sha256": cache_receipt_sha256,
        "fixed_audit_epoch": FIXED_EPOCH,
        "compute_policy": "cuda_only_no_cpu_fallback",
        "mlflow_filter_note": "Use campaign_id/experiment_key/phase/arm; legacy G1 tags are known trainer metadata debt.",
    })
    payload["tags"] = tags
    runtime = dict(payload.get("runtime") or {})
    runtime.update({"git_branch": "codex/mls-a2-geometry-20260904", "prepare_data": False, "auto_destroy": False})
    payload["runtime"] = runtime
    hardware = dict(payload.get("hardware") or {})
    hardware.update({"gpu_profile": "RTX_3090", "disk_gb": 100})
    payload["hardware"] = hardware
    payload["continuation_provenance"] = {
        "parent_r1_preregistration_sha256": parent_sha256,
        "training_source_commit": source_commit,
        "phase": "R1R2",
    }
    return payload, config


def materialize(
    *, parent_preregistration: Path, parent_preregistration_sha256: str,
    parent_matrix_dir: Path, parent_control_evaluation: Path,
    parent_candidate_evaluation: Path, parent_control_parity: Path,
    parent_candidate_parity: Path, parent_screen_gate: Path,
    cache_manifest_sha256: str, cache_validation_receipt: Path,
    truth_table: Path, truth_table_sha256: str, frozen_champion_predictions: Path,
    expected_frozen_champion_sha256: str, output_dir: Path,
    raw_dicom_root: Path,
    training_source_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create a new immutable R1R2 matrix without inspecting future outcomes."""
    source_root = training_source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R1R matrix directory: {output_dir}")
    parent_preregistration = parent_preregistration.resolve()
    parent_matrix_dir = parent_matrix_dir.resolve()
    cache_validation_receipt = cache_validation_receipt.resolve()
    truth_table = truth_table.resolve()
    frozen_champion_predictions = frozen_champion_predictions.resolve()
    raw_dicom_root = raw_dicom_root.resolve()
    parent_sha = _normalise_sha256(parent_preregistration_sha256, label="parent R1 preregistration")
    cache_sha = _normalise_sha256(cache_manifest_sha256, label="cache manifest")
    truth_sha = _normalise_sha256(truth_table_sha256, label="truth table")
    frozen_sha = _normalise_sha256(expected_frozen_champion_sha256, label="frozen Champion")
    source_hashes = _source_hashes(source_root)
    audit_hashes = _audit_source_hashes(source_root)
    studies, roster_sha = _fold_roster(source_root)
    if studies != 67:
        raise ValueError(f"R1R requires exactly 67 fold-1 studies, observed {studies}")
    cache_receipt = _load_json(cache_validation_receipt, label="cache validation receipt")
    cache_receipt_sha = sha256_file(cache_validation_receipt)
    if (
        cache_receipt.get("status") != "passed"
        or cache_receipt.get("cache_manifest_sha256") != cache_sha
    ):
        raise ValueError("cache validation receipt is not bound to the requested cache manifest")
    raw_binding = _raw_dicom_binding(raw_dicom_root, cache_receipt)
    _validate_truth(truth_table, truth_sha, studies=studies)
    _require_file_sha256(frozen_champion_predictions, frozen_sha, label="frozen Champion predictions")
    parent, parent_entries, parent_payloads, parent_configs = _validate_parent_matrix(
        parent_preregistration=parent_preregistration,
        parent_sha256=parent_sha,
        parent_matrix_dir=parent_matrix_dir,
        source_root=source_root,
        cache_manifest_sha256=cache_sha,
        cache_receipt_sha256=cache_receipt_sha,
    )
    control_eval, control_eval_sha = _validate_evaluation_receipt(
        parent_control_evaluation.resolve(), arm="control", parent_sha256=parent_sha,
        cache_receipt_sha256=cache_receipt_sha, truth_sha256=truth_sha,
    )
    candidate_eval, candidate_eval_sha = _validate_evaluation_receipt(
        parent_candidate_evaluation.resolve(), arm="candidate", parent_sha256=parent_sha,
        cache_receipt_sha256=cache_receipt_sha, truth_sha256=truth_sha,
    )
    _, control_parity_sha = _validate_parity_receipt(parent_control_parity.resolve(), arm="control")
    _, candidate_parity_sha = _validate_parity_receipt(parent_candidate_parity.resolve(), arm="candidate")
    screen_gate, screen_gate_sha = _validate_screen_gate(
        parent_screen_gate.resolve(), parent_sha256=parent_sha,
        control_receipt_sha256=control_eval_sha, candidate_receipt_sha256=candidate_eval_sha,
    )
    if screen_gate["control_checkpoint_sha256"] != control_eval["checkpoint_sha256"]:
        raise ValueError("parent screen gate/control checkpoint linkage differs")
    if screen_gate["candidate_checkpoint_sha256"] != candidate_eval["checkpoint_sha256"]:
        raise ValueError("parent screen gate/candidate checkpoint linkage differs")

    source_commit = _git_commit(source_root)
    parent_signatures = {arm: _model_signature(parent_configs[arm]) for arm in ARMS}
    planned_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    planned_configs: dict[tuple[str, int], dict[str, Any]] = {}
    config_texts: dict[tuple[str, int], str] = {}
    for arm in ARMS:
        for seed in REPLICA_SEEDS:
            payload, config = _replica_manifest(
                parent_payloads[arm], arm=arm, seed=seed, parent_sha256=parent_sha,
                cache_manifest_sha256=cache_sha, cache_receipt_sha256=cache_receipt_sha,
                source_commit=source_commit,
            )
            signature = _model_signature(config)
            if signature != parent_signatures[arm]:
                raise ValueError(f"R1R {arm}/seed{seed} differs from inherited seed42 recipe")
            text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
            planned_payloads[(arm, seed)] = payload
            planned_configs[(arm, seed)] = config
            config_texts[(arm, seed)] = text
    for arm in ARMS:
        if _field_differences(planned_configs[(arm, REPLICA_SEEDS[0])], planned_configs[(arm, REPLICA_SEEDS[1])]) != ["seed"]:
            raise ValueError(f"R1R {arm} replica configs differ by more than seed")
    for seed in REPLICA_SEEDS:
        if _field_differences(
            _model_signature(planned_configs[("control", seed)]),
            _model_signature(planned_configs[("candidate", seed)]),
        ) != ["horizontal_flip_prob"]:
            raise ValueError(f"R1R arms differ by more than horizontal_flip_prob for seed{seed}")

    members: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    parent_eval_by_arm = {"control": (control_eval, control_eval_sha), "candidate": (candidate_eval, candidate_eval_sha)}
    parent_parity_by_arm = {"control": (parent_control_parity.resolve(), control_parity_sha), "candidate": (parent_candidate_parity.resolve(), candidate_parity_sha)}
    for arm in ARMS:
        inherited_path = (parent_matrix_dir / str(parent_entries[arm]["file"])).resolve()
        inherited_eval, inherited_eval_sha = parent_eval_by_arm[arm]
        parity_path, parity_sha = parent_parity_by_arm[arm]
        members[arm][f"seed{INHERITED_SEED}"] = {
            "member_kind": "inherited_r1_seed42",
            "seed": INHERITED_SEED,
            "config_path": str(inherited_path),
            "config_sha256": str(parent_entries[arm]["sha256"]),
            "checkpoint_path": str(Path(str(inherited_eval["checkpoint"])).resolve()),
            "checkpoint_sha256": str(inherited_eval["checkpoint_sha256"]),
            "strict_evaluation_receipt": str((parent_control_evaluation if arm == "control" else parent_candidate_evaluation).resolve()),
            "strict_evaluation_receipt_sha256": inherited_eval_sha,
            "package_parity_receipt": str(parity_path),
            "package_parity_receipt_sha256": parity_sha,
        }
        for seed in REPLICA_SEEDS:
            payload = planned_payloads[(arm, seed)]
            path = output_dir / f"{payload['run_name']}.yaml"
            members[arm][f"seed{seed}"] = {
                "member_kind": "planned_r1r_replica",
                "seed": seed,
                "config_path": str(path),
                "config_sha256": _text_sha256(config_texts[(arm, seed)]),
                "expected_checkpoint_path": str(
                    source_root / "models" / "checkpoints" / "mls_multitask"
                    / str(payload["run_name"]) / f"mls_multitask_epoch_{FIXED_EPOCH:03d}.pth"
                ),
            }

    contract = {
        "schema_version": 2,
        "status": R1R2_STATUS,
        "campaign": "mls_reflection_r1r2_replication",
        "phase": "conditional_fold1_three_seed_replication_after_contract_hardening",
        "created_at_utc": _utc_now(),
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "training_source": {
            "root": str(source_root), "git_commit": source_commit,
            "core_source_sha256": source_hashes,
        },
        "audit_source": {
            "root": str(source_root),
            "source_sha256": audit_hashes,
        },
        "parent_r1": {
            "preregistration": str(parent_preregistration),
            "preregistration_sha256": parent_sha,
            "matrix_dir": str(parent_matrix_dir),
            "paired_mls_screen_gate": str(parent_screen_gate.resolve()),
            "paired_mls_screen_gate_sha256": screen_gate_sha,
            "control_strict_evaluation_sha256": control_eval_sha,
            "candidate_strict_evaluation_sha256": candidate_eval_sha,
            "control_package_parity_sha256": control_parity_sha,
            "candidate_package_parity_sha256": candidate_parity_sha,
        },
        "protocol": {
            "fold": FIXED_FOLD,
            "studies": studies,
            "fold1_roster_sha256": roster_sha,
            "fixed_epoch": FIXED_EPOCH,
            "cuda_audit_batch_size": AUDIT_BATCH_SIZE,
            "seeds": list(AUDIT_SEEDS),
            "new_cuda_trainings": 4,
            "within_arm_training_config_differences": ["seed"],
            "cross_arm_training_config_differences": ["horizontal_flip_prob"],
            "allowed_new_recipe_changes": [],
            "next_evidence": "raw_dicom_three_seed_audit_then_fold1_development_triage_only",
            "promotion_eligible": False,
            "submission_zip_allowed": False,
        },
        "data": {
            "cache_manifest_sha256": cache_sha,
            "cache_validation_receipt": str(cache_validation_receipt),
            "cache_validation_receipt_sha256": cache_receipt_sha,
            "truth_table": str(truth_table),
            "truth_table_sha256": truth_sha,
            "fold_manifest": str(source_root / CORE_SOURCE_RELATIVE_PATHS["fold_manifest"]),
            "fold_manifest_sha256": source_hashes["fold_manifest"],
            "frozen_champion_predictions": str(frozen_champion_predictions),
            "frozen_champion_predictions_sha256": frozen_sha,
            "raw_dicom": raw_binding,
            "project_config": str(source_root / "config/project.yaml"),
            "project_config_sha256": audit_hashes["project_config"],
            "canonical_triage_evaluator_sha256": audit_hashes["canonical_triage_evaluator"],
            "triage_rules_sha256": audit_hashes["triage_rules"],
        },
        "arms": {
            arm: {
                "horizontal_flip_prob": ARMS[arm]["horizontal_flip_prob"],
                "model_config_signature": parent_signatures[arm],
                "model_config_signature_sha256": _canonical_sha256(parent_signatures[arm]),
            }
            for arm in ARMS
        },
        "members": members,
        "mlflow": {
            "authoritative_manifest_tags": ["campaign_id", "experiment_key", "phase", "arm", "fold", "seed"],
            "known_legacy_tag_caveat": "trainer hard-codes G1 convenience tags for multitask_2p5d_v1; do not filter R1R by those tags",
        },
        "model_compute": "none_metadata_yaml_hash_validation_only",
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for (arm, seed), text in config_texts.items():
        path = Path(members[arm][f"seed{seed}"]["config_path"])
        _atomic_text(path, text)
        if sha256_file(path) != members[arm][f"seed{seed}"]["config_sha256"]:
            raise RuntimeError(f"written R1R config checksum mismatch: {path}")
    contract_path = output_dir / "r1r2_fold1_replication_contract.json"
    _atomic_text(contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    contract_sha = sha256_file(contract_path)
    # Supervisor launchers consume this immutable companion rather than an
    # operator-copied hash.  A mutated contract now fails before CUDA training.
    _atomic_text(output_dir / "r1r2_fold1_replication_contract.sha256", contract_sha + "\n")
    return {
        "status": contract["status"],
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "planned_cuda_trainings": contract["protocol"]["new_cuda_trainings"],
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-preregistration", type=Path, required=True)
    parser.add_argument("--parent-preregistration-sha256", required=True)
    parser.add_argument("--parent-matrix-dir", type=Path, required=True)
    parser.add_argument("--parent-control-evaluation", type=Path, required=True)
    parser.add_argument("--parent-candidate-evaluation", type=Path, required=True)
    parser.add_argument("--parent-control-parity", type=Path, required=True)
    parser.add_argument("--parent-candidate-parity", type=Path, required=True)
    parser.add_argument("--parent-screen-gate", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--truth-table", type=Path, required=True)
    parser.add_argument("--truth-table-sha256", required=True)
    parser.add_argument("--frozen-champion-predictions", type=Path, required=True)
    parser.add_argument("--expected-frozen-champion-sha256", required=True)
    parser.add_argument("--raw-dicom-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-source-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    result = materialize(
        parent_preregistration=args.parent_preregistration,
        parent_preregistration_sha256=args.parent_preregistration_sha256,
        parent_matrix_dir=args.parent_matrix_dir,
        parent_control_evaluation=args.parent_control_evaluation,
        parent_candidate_evaluation=args.parent_candidate_evaluation,
        parent_control_parity=args.parent_control_parity,
        parent_candidate_parity=args.parent_candidate_parity,
        parent_screen_gate=args.parent_screen_gate,
        cache_manifest_sha256=args.cache_manifest_sha256,
        cache_validation_receipt=args.cache_validation_receipt,
        truth_table=args.truth_table,
        truth_table_sha256=args.truth_table_sha256,
        frozen_champion_predictions=args.frozen_champion_predictions,
        expected_frozen_champion_sha256=args.expected_frozen_champion_sha256,
        raw_dicom_root=args.raw_dicom_root,
        output_dir=args.output_dir,
        training_source_root=args.training_source_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
