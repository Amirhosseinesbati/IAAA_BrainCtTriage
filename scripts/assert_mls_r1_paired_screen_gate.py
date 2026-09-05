"""Fail-closed comparison receipt for the pre-registered R1 paired MLS screen.

This script deliberately consumes only aggregate raw-DICOM evaluation receipts.
It never loads a model or exposes private per-study predictions.  Passing this
screen authorizes only the next, separately frozen triage gate; it is not a
promotion or submission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


GATES = {
    "study_f1_3mm_noninferior": ("f1_3mm", "higher"),
    "study_f1_5mm_noninferior": ("f1_5mm", "higher"),
    "study_boundary_f1_noninferior": ("boundary_f1", "higher"),
    "study_mae_noninferior": ("mae_mm", "lower"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_receipt(path: Path, *, arm: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{arm} receipt must be a JSON object")
    required = {
        "status", "campaign", "scope", "arm", "fold", "seed", "studies",
        "fixed_epoch", "checkpoint_sha256", "preregistration_sha256",
        "cache_validation_receipt_sha256", "truth_table_sha256", "metrics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{arm} receipt lacks required fields: {missing}")
    if payload["status"] != "completed" or payload["arm"] != arm:
        raise ValueError(f"{arm} receipt is not a completed matching R1 evaluation")
    if payload["campaign"] != "mls_reflection_paired" or payload["scope"] != "raw_dicom_single_fold_mls_screen_only":
        raise ValueError(f"{arm} receipt has an incompatible evaluation scope")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError(f"{arm} receipt metrics must be a mapping")
    for metric, _ in GATES.values():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{arm} receipt has invalid {metric}")
    return payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite R1 screen receipt: {args.output}")
    preregistration = args.preregistration.resolve()
    if _sha256(preregistration) != args.preregistration_sha256:
        raise ValueError("R1 preregistration checksum differs")
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    declared = set(prereg.get("screen_gates", []))
    if not set(GATES).issubset(declared):
        raise ValueError("R1 preregistration does not declare every required paired screen gate")

    control_path = args.control.resolve()
    candidate_path = args.candidate.resolve()
    control = _load_receipt(control_path, arm="control")
    candidate = _load_receipt(candidate_path, arm="candidate")
    shared = (
        "fold", "seed", "studies", "fixed_epoch", "preregistration_sha256",
        "cache_validation_receipt_sha256", "truth_table_sha256",
    )
    for key in shared:
        if control[key] != candidate[key]:
            raise ValueError(f"control/candidate receipt mismatch for {key}")
    if control["preregistration_sha256"] != args.preregistration_sha256:
        raise ValueError("evaluation receipts do not bind the requested preregistration")

    outcomes: dict[str, dict[str, Any]] = {}
    for gate, (metric, direction) in GATES.items():
        control_value = float(control["metrics"][metric])
        candidate_value = float(candidate["metrics"][metric])
        delta = candidate_value - control_value
        passed = delta >= 0.0 if direction == "higher" else delta <= 0.0
        outcomes[gate] = {
            "metric": metric,
            "direction": direction,
            "control": control_value,
            "candidate": candidate_value,
            "candidate_minus_control": delta,
            "noninferiority_margin": 0.0,
            "passed": passed,
        }
    passed = all(item["passed"] for item in outcomes.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "campaign": "mls_reflection_r1_paired_screen_gate",
        "preregistration_sha256": args.preregistration_sha256,
        "control_receipt_sha256": _sha256(control_path),
        "candidate_receipt_sha256": _sha256(candidate_path),
        "control_checkpoint_sha256": control["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "fold": control["fold"],
        "seed": control["seed"],
        "studies": control["studies"],
        "fixed_epoch": control["fixed_epoch"],
        "gates": outcomes,
        "next_gate_authorized": passed,
        "promotion_eligible": False,
        "submission_zip_allowed": False,
        "private_predictions_persisted": False,
        "model_compute": "none_aggregate_receipt_only",
    }
    _atomic_json(args.output, result)
    if not passed:
        raise RuntimeError("R1 paired screen non-inferiority did not pass")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "next_gate_authorized": result["next_gate_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
