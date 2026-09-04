"""Fail-closed resource gate for the signed-geometry MLS A2 experiment.

This script consumes aggregate CUDA-audit output only.  It intentionally does
not read the private per-study prediction CSV, select a checkpoint, or make a
promotion/submission claim.  Passing authorizes exactly the two remaining
pre-registered fold-0 seed replications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_AUDIT_CANDIDATE = "epoch015"
EXPECTED_FOLD = 0
EXPECTED_STUDIES = 70
EXPECTED_POOLING = {
    "selector_threshold": 0.5,
    "top_k": 3,
    "aggregation": "p90",
}
GATES = {
    "mae_mm_lte": 1.4709586392,
    "f1_3mm_gte": 0.8196721311,
    "f1_5mm_gte": 0.7368421053,
    "boundary_f1_gte": 0.7782571182,
    "selection_objective_lte": 1.9044444028,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON contract: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON contract must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite_metric(profile: dict[str, Any], key: str) -> float:
    value = profile.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Fixed profile has no finite {key}")
    return float(value)


def evaluate(
    audit_status_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    audit_status_path = audit_status_path.resolve()
    metrics_path = metrics_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    audit = _read_json(audit_status_path)
    metrics = _read_json(metrics_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if audit.get("state") != "completed":
        raise ValueError("CUDA audit is not completed")
    if audit.get("compute_policy") != "cuda_only_no_cpu_fallback":
        raise ValueError("Unexpected audit compute policy")
    if int(audit.get("fold", -1)) != EXPECTED_FOLD:
        raise ValueError("A2 resource screen requires fold 0")
    if int(audit.get("expected_studies", -1)) != EXPECTED_STUDIES:
        raise ValueError("A2 resource screen requires exactly 70 studies")
    candidates = audit.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {EXPECTED_AUDIT_CANDIDATE}:
        raise ValueError("A2 resource screen must audit exactly epoch015")
    candidate = candidates[EXPECTED_AUDIT_CANDIDATE]
    if candidate.get("state") != "completed" or int(candidate.get("exit_code", -1)) != 0:
        raise ValueError("A2 epoch015 CUDA audit did not complete cleanly")
    if Path(str(candidate.get("checkpoint", ""))).resolve() != checkpoint_path:
        raise ValueError("Audit checkpoint differs from the fixed A2 epoch015 checkpoint")
    if Path(str(metrics.get("checkpoint", ""))).resolve() != checkpoint_path:
        raise ValueError("Metrics checkpoint differs from the fixed A2 epoch015 checkpoint")
    if int(metrics.get("fold", -1)) != EXPECTED_FOLD or int(metrics.get("n_studies", -1)) != EXPECTED_STUDIES:
        raise ValueError("Metrics do not cover the fixed A2 fold-0 contract")
    if int(metrics.get("failures", -1)) != 0:
        raise ValueError("A2 resource screen refuses inference failures")
    fixed = metrics.get("fixed_profile_pre_registered")
    if not isinstance(fixed, dict):
        raise ValueError("Metrics lack the fixed pre-registered profile")
    for key, expected in EXPECTED_POOLING.items():
        if fixed.get(key) != expected:
            raise ValueError(f"Fixed profile changed {key}")

    mae_mm = _finite_metric(fixed, "mae_mm")
    f1_3mm = _finite_metric(fixed, "f1_3mm")
    f1_5mm = _finite_metric(fixed, "f1_5mm")
    boundary_f1 = 0.5 * (f1_3mm + f1_5mm)
    selection_objective = mae_mm + 2.0 * (1.0 - boundary_f1)
    observed = {
        "mae_mm": mae_mm,
        "f1_3mm": f1_3mm,
        "f1_5mm": f1_5mm,
        "boundary_f1": boundary_f1,
        "selection_objective": selection_objective,
    }
    gate_results = {
        "mae_mm_lte": mae_mm <= GATES["mae_mm_lte"],
        "f1_3mm_gte": f1_3mm >= GATES["f1_3mm_gte"],
        "f1_5mm_gte": f1_5mm >= GATES["f1_5mm_gte"],
        "boundary_f1_gte": boundary_f1 >= GATES["boundary_f1_gte"],
        "selection_objective_lte": selection_objective <= GATES["selection_objective_lte"],
    }
    failed_gates = [name for name, passed in gate_results.items() if not passed]
    passed = not failed_gates
    result = {
        "schema_version": 1,
        "status": (
            "passed_for_two_remaining_fold0_seed_replications"
            if passed
            else "rejected_stop_a2_expansion"
        ),
        "screen_scope": "a2_fold0_seed42_resource_screen_only",
        "candidate": "mls-vast-deploy-aligned-a2-signed-geometry",
        "fixed_epoch": 15,
        "fold": EXPECTED_FOLD,
        "studies": EXPECTED_STUDIES,
        "compute_policy": "cuda_only_no_cpu_fallback",
        "audit_status": str(audit_status_path),
        "audit_status_sha256": _sha256(audit_status_path),
        "metrics": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "fixed_pooling": EXPECTED_POOLING,
        "gates": GATES,
        "observed": observed,
        "gate_results": gate_results,
        "failed_gates": failed_gates,
        "can_start_only_seeds_2026_and_3407_on_fold0": passed,
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    _atomic_json(output_path.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-status", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.audit_status, args.metrics, args.checkpoint, args.output)
    print(json.dumps({
        "status": result["status"],
        "observed": result["observed"],
        "failed_gates": result["failed_gates"],
        "promotion_eligible": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
