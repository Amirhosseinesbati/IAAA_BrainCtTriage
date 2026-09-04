"""Fail-closed A3 gate using the frozen A2 resource-screen contract.

The common gate implementation is intentionally reused so A3 cannot weaken its
audit, but its decision and MLflow namespace are distinct from rejected A2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_a2_fold0_resource_screen import (
    _atomic_json,
    evaluate as _evaluate_frozen_contract,
)


def evaluate(
    audit_status_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply the unchanged fixed contract and label the resulting A3 decision."""
    result = _evaluate_frozen_contract(
        audit_status_path, metrics_path, checkpoint_path, output_path,
    )
    passed = not result["failed_gates"]
    result.update({
        "status": (
            "passed_for_two_remaining_fold0_seed_replications"
            if passed
            else "rejected_stop_a3_expansion"
        ),
        "screen_scope": "a3_fold0_seed42_resource_screen_only",
        "candidate": "mls-vast-deploy-aligned-a3-study-bag",
    })
    _atomic_json(output_path.resolve(), result)
    return result


def _log_aggregate_to_mlflow(
    run_id: str, result: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Log aggregate A3 gate evidence only; private predictions never enter MLflow."""
    try:
        from mlflow.tracking import MlflowClient

        from src.mlops.tracking import configure_tracking_environment

        configure_tracking_environment()
        client = MlflowClient()
        for key, value in result["observed"].items():
            client.log_metric(run_id, f"a3_resource_{key}", float(value))
        for key, passed in result["gate_results"].items():
            client.log_metric(run_id, f"a3_resource_gate_{key}", float(bool(passed)))
        client.log_metric(
            run_id,
            "a3_resource_all_gates_passed",
            float(not result["failed_gates"]),
        )
        client.set_tag(run_id, "a3_resource_screen_status", str(result["status"]))
        client.set_tag(
            run_id,
            "a3_resource_screen_checkpoint_sha256",
            str(result["checkpoint_sha256"]),
        )
        client.log_artifact(
            run_id,
            str(output_path.resolve()),
            "reports/mls_deploy_aligned_a3/resource_screen",
        )
        return {"status": "logged", "run_id": run_id}
    except Exception as exc:  # The local decision remains authoritative and retryable.
        return {"status": "deferred", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-status", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow-run-id", default="")
    args = parser.parse_args()
    result = evaluate(args.audit_status, args.metrics, args.checkpoint, args.output)
    if args.mlflow_run_id:
        result["mlflow"] = _log_aggregate_to_mlflow(
            args.mlflow_run_id, result, args.output,
        )
        _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "observed": result["observed"],
        "failed_gates": result["failed_gates"],
        "promotion_eligible": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
