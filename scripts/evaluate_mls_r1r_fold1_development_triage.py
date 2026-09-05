"""Run the checksum-bound, development-only R1R2 fold-1 triage comparison.

This program never performs MLS inference.  It accepts only the two completed
CUDA three-seed audits that belong to a sealed R1R continuation contract,
revalidates their checkpoint/config provenance, then delegates triage math to
the canonical reducer.  Its output is permanently scoped to one held-out fold;
it can never authorize a production checkpoint or submission ZIP.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_three_seed_fold_cuda import _sha256
from scripts.materialize_mls_r1_replication_matrix import (
    ARMS,
    AUDIT_SEEDS,
    FIXED_EPOCH,
    FIXED_FOLD,
    _fold_roster,
)
from scripts.validate_mls_r1_replication_matrix import validate_contract


AUDIT_PROTOCOL = "heldout_fold_fixed_epoch15_three_distinct_seed_median"
COMPUTE_POLICY = "cuda_only_no_cpu_model_fallback"
CANONICAL_PROTOCOL = "deploy_aligned_fixed_three_seed_median_canonical_triage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require_sealed_project_config(contract: dict[str, Any]) -> Path:
    """Reject ambient config overrides before any canonical reducer is invoked."""
    data = contract["data"]
    expected = Path(str(data["project_config"])).resolve()
    selected = Path(os.environ.get("IAAA_CONFIG_PATH", "")).resolve()
    if selected != expected:
        raise ValueError(
            "R1R2 requires IAAA_CONFIG_PATH to equal its sealed project configuration"
        )
    if _sha256(selected) != data["project_config_sha256"]:
        raise ValueError("R1R2 project configuration differs from the sealed contract")
    return selected


def _expected_roster(contract: dict[str, Any]) -> pd.DataFrame:
    data = contract["data"]
    fold_manifest = Path(str(data["fold_manifest"])).resolve()
    frame = pd.read_csv(fold_manifest, dtype={"study_id": str, "patient_id": str})
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise")
    expected = frame.loc[frame["fold"] == FIXED_FOLD, [
        "study_id", "patient_id", "triage_class",
    ]].copy()
    expected["study_id"] = expected["study_id"].astype(str)
    expected["patient_id"] = expected["patient_id"].astype(str)
    expected["triage_class"] = pd.to_numeric(expected["triage_class"], errors="raise").astype(int)
    studies, roster_sha = _fold_roster(Path(str(contract["training_source"]["root"])))
    if studies != 67 or roster_sha != contract["protocol"]["fold1_roster_sha256"]:
        raise ValueError("current fold-1 roster differs from the sealed R1R contract")
    return expected


def _validate_truth(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    truth_path = Path(str(contract["data"]["truth_table"])).resolve()
    truth = pd.read_csv(truth_path, dtype={"dicom_series.id": str})[[
        "dicom_series.id", "MLS_mm",
    ]].rename(columns={"dicom_series.id": "study_id", "MLS_mm": "authoritative_gt_MLS_mm"})
    truth["study_id"] = truth["study_id"].astype(str)
    merged = frame[["study_id", "gt_MLS_mm"]].merge(
        truth, on="study_id", how="left", validate="one_to_one",
    )
    observed = pd.to_numeric(merged["gt_MLS_mm"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(merged["authoritative_gt_MLS_mm"], errors="coerce").to_numpy(float)
    if not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise ValueError("R1R audit has missing/non-finite authoritative MLS truth")
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-6):
        raise ValueError("R1R audit ground truth differs from the locked truth table")


def _member_checkpoint_path(member: dict[str, Any], seed: int) -> Path:
    value = member.get("checkpoint_path") if seed == 42 else member.get("expected_checkpoint_path")
    return Path(str(value)).resolve()


def _validate_arm_audit(
    *, contract: dict[str, Any], checkpoint_members: dict[str, Any], arm: str,
    summary_path: Path, private_path: Path,
) -> dict[str, Any]:
    """Check a generic CUDA audit against the sealed R1R2 arm identity."""
    summary_path = summary_path.resolve()
    private_path = private_path.resolve()
    summary = _load_json(summary_path, label=f"R1R {arm} audit summary")
    required = {
        "schema_version", "status", "protocol", "compute_policy", "fold", "studies", "fixed_epoch",
        "seeds", "config_differences", "checkpoint_manifest", "private_predictions_sha256",
        "raw_predictions_uploaded_to_mlflow",
        "data_sources",
    }
    if missing := sorted(required - set(summary)):
        raise ValueError(f"R1R {arm} audit summary lacks fields: {missing}")
    checks = {
        "schema_version": int(summary["schema_version"]) == 1,
        "status": summary["status"] == "completed",
        "protocol": summary["protocol"] == AUDIT_PROTOCOL,
        "compute_policy": summary["compute_policy"] == COMPUTE_POLICY,
        "fold": int(summary["fold"]) == FIXED_FOLD,
        "studies": int(summary["studies"]) == 67,
        "epoch": int(summary["fixed_epoch"]) == FIXED_EPOCH,
        "seeds": sorted(int(value) for value in summary["seeds"]) == list(AUDIT_SEEDS),
        "config_differences": list(summary["config_differences"]) == ["seed"],
        "private_hash": summary["private_predictions_sha256"] == _sha256(private_path),
        "private_not_uploaded": summary["raw_predictions_uploaded_to_mlflow"] is False,
    }
    data = contract["data"]
    sources = summary["data_sources"]
    if not isinstance(sources, dict):
        raise ValueError(f"R1R2 {arm} audit data_sources is invalid")
    raw_root = Path(str(data["raw_dicom"]["resolved_root"])).resolve()
    audit_hashes = contract["audit_source"]["source_sha256"]
    checks.update({
        "audit_evaluator": sources.get("evaluator_sha256") == audit_hashes["three_seed_cuda_evaluator"],
        "fold_manifest": sources.get("fold_manifest_sha256") == data["fold_manifest_sha256"],
        "truth_table": sources.get("truth_table_sha256") == data["truth_table_sha256"],
        "raw_root": Path(str(sources.get("data_root", ""))).resolve() == raw_root,
    })
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"R1R {arm} audit provenance failed: {failed}")
    manifest = summary["checkpoint_manifest"]
    if not isinstance(manifest, dict) or set(manifest) != {f"seed{seed}" for seed in AUDIT_SEEDS}:
        raise ValueError(f"R1R {arm} audit checkpoint labels are not the locked seeds")
    for seed in AUDIT_SEEDS:
        label = f"seed{seed}"
        expected = checkpoint_members[arm][label]
        observed = manifest[label]
        if not isinstance(observed, dict):
            raise ValueError(f"R1R {arm}/{label} audit metadata is invalid")
        expected_path = _member_checkpoint_path(contract["members"][arm][label], seed)
        checks = {
            "path": Path(str(observed.get("path", ""))).resolve() == expected_path,
            "sha256": observed.get("sha256") == expected["sha256"],
            "bytes": int(observed.get("bytes", -1)) == int(expected["bytes"]),
            "epoch": int(observed.get("epoch", -1)) == FIXED_EPOCH,
            "seed": int(observed.get("seed", -1)) == seed,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"R1R {arm}/{label} audit checkpoint mismatch: {failed}")

    frame = pd.read_csv(private_path, dtype={"study_id": str, "patient_id": str})
    member_columns = [f"seed{seed}_MLS_mm" for seed in AUDIT_SEEDS]
    required_columns = {
        "study_id", "patient_id", "triage_class", "gt_MLS_mm", "median_MLS_mm", "error",
        *member_columns,
    }
    if missing := sorted(required_columns - set(frame.columns)):
        raise ValueError(f"R1R {arm} private audit table lacks columns: {missing}")
    if frame["error"].fillna("").astype(str).ne("").any():
        raise ValueError(f"R1R {arm} private audit table contains CUDA inference errors")
    numeric = frame[["gt_MLS_mm", "median_MLS_mm", *member_columns]].apply(
        pd.to_numeric, errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"R1R {arm} private audit table contains non-finite MLS values")
    recomputed_median = numeric[member_columns].median(axis=1).to_numpy(float)
    if not np.allclose(numeric["median_MLS_mm"].to_numpy(float), recomputed_median, rtol=0.0, atol=1e-6):
        raise ValueError(f"R1R {arm} stored median differs from its three members")
    expected = _expected_roster(contract)
    frame["study_id"] = frame["study_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    if len(frame) != len(expected) or frame["study_id"].duplicated().any():
        raise ValueError(f"R1R {arm} private audit table has invalid fold coverage")
    joined = frame[["study_id", "patient_id", "triage_class"]].merge(
        expected, on="study_id", suffixes=("_actual", "_expected"), validate="one_to_one",
    )
    if len(joined) != len(expected) or not (
        joined["patient_id_actual"].eq(joined["patient_id_expected"]).all()
        and pd.to_numeric(joined["triage_class_actual"], errors="coerce").eq(
            joined["triage_class_expected"]
        ).all()
    ):
        raise ValueError(f"R1R {arm} private audit table differs from the immutable fold roster")
    _validate_truth(frame, contract)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "private_predictions_path": str(private_path),
        "private_predictions_sha256": _sha256(private_path),
        "checkpoint_sha256": {label: manifest[label]["sha256"] for label in sorted(manifest)},
        "studies": len(frame),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R1R development triage output: {output_dir}")
    contract_path = args.contract.resolve()
    validation = validate_contract(
        contract_path,
        expected_contract_sha256=args.contract_sha256,
        require_checkpoints=True,
    )
    contract = _load_json(contract_path, label="R1R continuation contract")
    project_config = _require_sealed_project_config(contract)
    data = contract["data"]
    source_root = Path(str(contract["training_source"]["root"])).resolve()
    canonical = source_root / "scripts" / "evaluate_mls_deploy_aligned_seed_medians.py"
    triage_rules = source_root / "src" / "evaluation" / "triage.py"
    if _sha256(canonical) != data["canonical_triage_evaluator_sha256"]:
        raise ValueError("canonical triage evaluator differs from the sealed R1R contract")
    if _sha256(triage_rules) != data["triage_rules_sha256"]:
        raise ValueError("triage rules differ from the sealed R1R contract")
    checkpoint_members = validation.get("checkpoint_members")
    if not isinstance(checkpoint_members, dict):
        raise RuntimeError("R1R checkpoint metadata was not validated")
    control = _validate_arm_audit(
        contract=contract, checkpoint_members=checkpoint_members, arm="control",
        summary_path=args.control_summary, private_path=args.control_private,
    )
    candidate = _validate_arm_audit(
        contract=contract, checkpoint_members=checkpoint_members, arm="candidate",
        summary_path=args.candidate_summary, private_path=args.candidate_private,
    )
    # A failed canonical reducer must never leave a final directory that blocks
    # a corrected, checksum-identical retry.  Work in a sibling staging path
    # and atomically publish only a fully validated receipt.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    canonical_output = staging_dir / "canonical"
    command = [
        sys.executable, str(canonical),
        "--baseline-fold", f"{FIXED_FOLD}={control['private_predictions_path']}",
        "--baseline-fold-summary", f"{FIXED_FOLD}={control['summary_path']}",
        "--candidate-fold", f"{FIXED_FOLD}={candidate['private_predictions_path']}",
        "--candidate-fold-summary", f"{FIXED_FOLD}={candidate['summary_path']}",
        "--fold-manifest", str(data["fold_manifest"]),
        "--frozen-champion-predictions", str(data["frozen_champion_predictions"]),
        "--expected-frozen-champion-sha256", str(data["frozen_champion_predictions_sha256"]),
        "--truth-table", str(data["truth_table"]),
        "--output-dir", str(canonical_output),
    ]
    environment = dict(os.environ)
    environment["IAAA_CONFIG_PATH"] = str(project_config)
    subprocess.run(command, cwd=source_root, check=True, env=environment)
    canonical_summary_path = canonical_output / "aggregate_summary.json"
    canonical_summary = _load_json(canonical_summary_path, label="canonical R1R triage aggregate")
    checks = {
        "protocol": canonical_summary.get("protocol") == CANONICAL_PROTOCOL,
        "scope": canonical_summary.get("evaluation_scope") == "development_oof_subset",
        "fold": canonical_summary.get("selected_folds") == [FIXED_FOLD],
        "studies": int(canonical_summary.get("studies", -1)) == 67,
        "promotion": canonical_summary.get("promotion_eligible") is False,
        "truth": canonical_summary.get("sources", {}).get("truth_table", {}).get("sha256") == data["truth_table_sha256"],
        "fold_manifest": canonical_summary.get("sources", {}).get("fold_manifest", {}).get("sha256") == data["fold_manifest_sha256"],
        "frozen": canonical_summary.get("sources", {}).get("frozen_champion_predictions", {}).get("sha256") == data["frozen_champion_predictions_sha256"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"canonical R1R2 triage receipt failed sealed checks: {failed}")
    result = {
        "schema_version": 2,
        "status": "completed",
        "campaign": "mls_reflection_r1r2_replication",
        "evaluation_scope": "development_oof_subset",
        "selected_folds": [FIXED_FOLD],
        "studies": 67,
        "contract": str(contract_path),
        "contract_sha256": validation["contract_sha256"],
        "control_audit": control,
        "candidate_audit": candidate,
        "canonical_aggregate_summary": str(output_dir / "canonical" / "aggregate_summary.json"),
        "canonical_aggregate_summary_sha256": _sha256(canonical_summary_path),
        "canonical_development_gate_passed": canonical_summary["development_gate_passed"],
        "promotion_eligible": False,
        "submission_zip_allowed": False,
        "model_compute": "none_saved_cuda_audits_plus_canonical_triage_cpu_metadata_only",
        "cpu_policy_note": "No model forward pass occurs here; torch checkpoint loading and canonical triage metrics are metadata/CPU-only postprocessing of completed CUDA audits.",
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(staging_dir / "r1r2_development_triage_receipt.json", result)
    os.replace(staging_dir, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--control-private", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-private", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({
        "status": result["status"],
        "evaluation_scope": result["evaluation_scope"],
        "promotion_eligible": result["promotion_eligible"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
