"""Generate a compact, public, decision-oriented MLS campaign index.

The source receipts remain immutable evidence.  This index only projects their
aggregate metrics and MLflow links into one readable view; it never reads
private predictions, checkpoints, DICOMs, or training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/mls_experiments/mls-deploy-aligned-upgrade-20260902"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(name: str) -> dict[str, Any]:
    path = CAMPAIGN / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {name}")
    return value


def _evidence(name: str) -> dict[str, str]:
    path = CAMPAIGN / name
    if "private" in name.lower():
        raise ValueError("Private evidence cannot enter campaign index")
    return {"path": f"reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/{name}", "sha256": _sha256(path)}


def _record_a7() -> dict[str, Any]:
    protocol_name = "A7_PAIRED_TRAINING_PROTOCOL_20260904.json"
    pair_name = "A7_PAIR_AGGREGATE_20260904.json"
    protocol, pair = _read(protocol_name), _read(pair_name)
    receipt = _read("A7_LAUNCH_MLFLOW_RECEIPT_20260904.json")
    return {
        "experiment_key": "A7",
        "hypothesis": protocol["rationale"],
        "state": "rejected_resource_gate",
        "contract": {
            "fold": protocol["fold"], "seed": protocol["seed"],
            "fixed_epoch": protocol["training"]["stop_after_epoch"],
            "steps_per_epoch": protocol["training"]["optimizer_steps_per_epoch"],
            "comparator_kind": "paired_control",
            "training_protocol": _evidence(protocol_name),
        },
        "runs": [{
            "stage": "train_and_historical_resource_screen", "role": "paired_control_and_consistency",
            "mlflow_run_id": receipt["run_id"],
            "state": "historical_receipt_state_not_reinterpreted",
        }],
        "screen": {
            "baseline_metrics": pair["baseline"], "control_metrics": pair["control"],
            "candidate_role": "consistency", "candidate_metrics": pair["consistency"],
            "candidate_minus_control": pair["consistency_minus_control"],
            "resource_gates_passed": pair["consistency_resource_gates_passed"],
        },
        "decision": {
            "promotion_eligible": pair["promotion_eligible"],
            "submission_zip_allowed": pair["submission_zip_allowed"],
            "automatic_replication_allowed": pair["automatic_replication_allowed"],
        },
        "public_evidence": [_evidence(pair_name), _evidence("A7_PAIRED_RESULT_20260904.md")],
        "private_predictions_uploaded": False,
    }


def _record_a8() -> dict[str, Any]:
    protocol_name = "A8_TRAINING_PROTOCOL_20260904.json"
    pair_name = "A8_PAIR_AGGREGATE_20260904.json"
    protocol, pair = _read(protocol_name), _read(pair_name)
    control = _read("A8_CONTROL_TRAINING_SUMMARY_20260904.json")
    refinement = _read("A8_REFINEMENT_TRAINING_SUMMARY_20260904.json")
    return {
        "experiment_key": "A8",
        "hypothesis": "Reference-conditioned refinement versus matched ordinary control.",
        "state": "rejected_resource_gate",
        "contract": {
            "fold": protocol["training"]["fold"], "seed": protocol["training"]["seed"],
            "fixed_epoch": protocol["training"]["stop_epoch"],
            "steps_per_epoch": protocol["training"]["steps_per_epoch"],
            "comparator_kind": "paired_control",
            "training_protocol": _evidence(protocol_name),
        },
        "runs": [
            {"stage": "train", "role": "control", "mlflow_run_id": control["mlflow_run_id"], "state": control["status"]},
            {"stage": "train", "role": "refinement", "mlflow_run_id": refinement["mlflow_run_id"], "state": refinement["status"]},
        ],
        "screen": {
            "baseline_metrics": pair["baseline"], "control_metrics": pair["control"],
            "candidate_role": "refinement", "candidate_metrics": pair["refinement"],
            "candidate_minus_control": pair["refinement_minus_control"],
            "resource_gates_passed": pair["refinement_resource_gates_passed"],
        },
        "decision": {
            "promotion_eligible": pair["promotion_eligible"],
            "submission_zip_allowed": pair["submission_zip_allowed"],
            "automatic_replication_allowed": pair["automatic_replication_allowed"],
        },
        "public_evidence": [_evidence(pair_name), _evidence("A8_PAIRED_RESULT_20260904.md")],
        "private_predictions_uploaded": False,
    }


def _record_a9(audit_run_id: str, audit_state: str) -> dict[str, Any]:
    train_name = "A9_TRAINING_PROTOCOL_20260904.json"
    evaluation_name = "A9_CANONICAL_EVALUATION_PROTOCOL_20260904.json"
    pair_name = "A9_PAIR_AGGREGATE_20260904.json"
    train, evaluation, pair = _read(train_name), _read(evaluation_name), _read(pair_name)
    return {
        "experiment_key": "A9",
        "hypothesis": train["hypothesis"],
        "state": "rejected_resource_gate",
        "contract": {
            "fold": train["training"]["fold"], "seed": train["training"]["seed"],
            "fixed_epoch": train["training"]["fixed_epochs"],
            "steps_per_epoch": train["training"]["steps_per_epoch"],
            "expected_optimizer_steps": train["training"]["total_optimizer_steps"],
            "comparator_kind": "qualified_runtime_baseline",
            "training_protocol": _evidence(train_name),
            "evaluation_protocol": _evidence(evaluation_name),
            "qualified_runtime_reference_sha256": train["runtime_reference_sha256"],
        },
        "runs": [
            {"stage": "train", "role": "candidate", "mlflow_run_id": evaluation["training_provenance"]["mlflow_run_id"], "state": "finished"},
            {
                "stage": "resource_screen", "role": "candidate_vs_qualified_runtime_baseline",
                "source_training_run_id": evaluation["training_provenance"]["mlflow_run_id"],
                "mlflow_run_id": audit_run_id, "state": audit_state,
                "artifact_readback_verified": False,
            },
        ],
        "screen": {
            "scope": {"fold": evaluation["candidate"]["fold"], "seed": evaluation["candidate"]["seed"],
                      "studies": evaluation["candidate"]["expected_studies"], "fixed_epoch": evaluation["candidate"]["fixed_epoch"]},
            "baseline_metrics": pair["baseline"], "candidate_metrics": pair["candidate"],
            "candidate_minus_baseline": pair["candidate_minus_baseline"],
            "gate_results": pair["gate_results"], "resource_gates_passed": pair["resource_gates_passed"],
        },
        "decision": {
            "failed_gates": [name for name, passed in pair["gate_results"].items() if not passed],
            "replication_review_eligible": pair["replication_review_eligible"],
            "promotion_eligible": pair["promotion_eligible"],
            "submission_zip_allowed": pair["submission_zip_allowed"],
        },
        "public_evidence": [_evidence(pair_name), _evidence("A9_PAIRED_RESULT_20260904.md")],
        "private_predictions_uploaded": False,
    }


def _operational_evidence() -> list[dict[str, Any]]:
    """Expose non-candidate infrastructure evidence without mixing it with models."""
    name = "A9_SPEED_EQUIVALENCE_RESULT_20260905.json"
    path = CAMPAIGN / name
    if not path.is_file():
        return []
    result = _read(name)
    if (result.get("status"), result.get("candidate_model"), result.get("checkpoints_written")) != (
        "completed", False, False
    ):
        raise ValueError("Speed-equivalence evidence is not a completed non-candidate run")
    return [{
        "evidence_key": "A9-speed-equivalence-20260905",
        "kind": "non_candidate_trainer_runtime_equivalence",
        "state": "adopted_for_future_candidate_trainers_only" if result["adoption_eligible"] else "not_adopted",
        "candidate_model": False,
        "checkpoint_written": False,
        "speedup_ratio": result["speed"]["speedup_ratio"],
        "equivalence_gates": result["gates"],
        "mlflow": result["mlflow"],
        "public_evidence": [
            _evidence(name),
            _evidence("A9_SPEED_EQUIVALENCE_MLFLOW_RECEIPT_20260905.json"),
        ],
    }]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(CAMPAIGN / "campaign_index.json"))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--a9-audit-run-id", default="ec366b9205a34258b3fc4792d11f4c5c")
    parser.add_argument("--a9-audit-state", default="failed_artifact_readback_unverified")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.replace:
        raise FileExistsError("Refusing to overwrite campaign index without --replace")
    payload = {
        "schema_version": 1,
        "campaign_id": "mls-deploy-aligned-20260902",
        "metric_schema": "mls_resource_v1",
        "purpose": "Public aggregate decision index; not a final triage or leaderboard claim.",
        "records": [_record_a7(), _record_a8(), _record_a9(args.a9_audit_run_id, args.a9_audit_state)],
        "operational_evidence": _operational_evidence(),
    }
    _atomic_json(output, payload)
    print(json.dumps({"status": "completed", "output": str(output), "records": 3}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
