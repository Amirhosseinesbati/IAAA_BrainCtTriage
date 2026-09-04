"""Fail-closed A4 triage gate for the fixed three-seed fold-0 audit.

This gate is intentionally self-contained.  The deployment worktree carries
only the source needed for the active A4 campaign, so importing validation
helpers from historical A1/A3 scripts would make the supposedly fail-closed
decision depend on files that are not part of the candidate's audit contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = "deploy_aligned_fixed_three_seed_median_canonical_triage"
EXPECTED_GATES = (
    "frozen_context_macro_f1_strictly_improved",
    "frozen_context_urgent_f1_strictly_improved",
    "frozen_context_accuracy_noninferior",
    "normal_f1_not_below_control_minus_0p01",
    "critical_f1_not_below_control_minus_0p01",
    "f1_3mm_noninferior",
    "f1_5mm_noninferior",
    "oracle_macro_and_urgent_directions_nonnegative",
)


EXPECTED_PREREGISTRATION_STATUS = "locked_before_any_a4_audit_or_triage_outcome"
EXPECTED_STAGE = "a4_pair_rank"


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
    temporary.replace(path)


def _one_fold_source(payload: dict[str, Any], name: str) -> dict[str, Any]:
    rows = payload.get("sources", {}).get(name)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{name} must contain exactly the fold-0 audit source")
    source = rows[0]
    if int(source.get("fold", -1)) != 0:
        raise ValueError(f"{name} must contain exactly the fold-0 audit source")
    if int(source.get("studies", -1)) != 70:
        raise ValueError(f"{name} must bind exactly 70 held-out studies")
    for field in ("sha256", "audit_summary_sha256"):
        digest = source.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{name}.{field} is not a lowercase SHA-256 digest")
    return source


def _valid_three_member_source(source: dict[str, Any]) -> bool:
    checkpoints = source.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or len(checkpoints) != 3:
        return False
    values = list(checkpoints.values())
    return (
        len(set(values)) == 3
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in values
        )
    )


def evaluate(
    aggregate_summary: Path,
    preregistration: Path,
    output: Path,
) -> dict[str, Any]:
    aggregate_summary = aggregate_summary.resolve()
    preregistration = preregistration.resolve()
    summary = _read_json(aggregate_summary)
    prereg = _read_json(preregistration)
    if prereg.get("status") != EXPECTED_PREREGISTRATION_STATUS:
        raise ValueError("A4 triage preregistration is not locked")
    if prereg.get("candidate", {}).get("stage") != EXPECTED_STAGE:
        raise ValueError("A4 triage preregistration names an unexpected candidate")
    if tuple(prereg.get("fixed_screen_gates", ())) != EXPECTED_GATES:
        raise ValueError("A4 triage gates differ from the executable contract")
    if int(summary.get("schema_version", -1)) != 1 or summary.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected canonical triage evaluation protocol")
    if summary.get("selected_folds") != [0] or int(summary.get("studies", -1)) != 70:
        raise ValueError("A4 early-rejection screen must cover exactly fold 0 / 70 studies")
    if bool(summary.get("full_fold_coverage")) or bool(summary.get("promotion_eligible")):
        raise ValueError("A fold-0 screen cannot claim full coverage or promotion eligibility")
    if summary.get("evaluation_scope") != "development_oof_subset":
        raise ValueError("Fold-0 screen must be labelled as a development OOF subset")

    baseline_source = _one_fold_source(summary, "baseline_folds")
    candidate_source = _one_fold_source(summary, "candidate_folds")
    control = prereg.get("control", {})
    control_checks = {
        "fold": int(control.get("fold", -1)) == 0,
        "studies": int(control.get("studies", -1)) == 70,
        "fixed_epoch": int(control.get("fixed_epoch", -1)) == 15,
        "seeds": control.get("seeds") == [42, 2026, 3407],
        "aggregation": control.get("aggregation") == "median",
        "audit_summary_sha256": baseline_source.get("audit_summary_sha256") == control.get("audit_summary_sha256"),
        "private_predictions_sha256": baseline_source.get("sha256") == control.get("private_predictions_sha256"),
        "candidate_distinct": baseline_source.get("sha256") != candidate_source.get("sha256"),
        "candidate_three_distinct_members": _valid_three_member_source(candidate_source),
    }
    failed_control = [name for name, passed in control_checks.items() if not passed]
    if failed_control:
        raise ValueError(f"A4 triage source contract failed: {failed_control}")

    contexts = summary.get("contexts", {})
    if set(contexts) != {"frozen_champion", "oracle"}:
        raise ValueError("Both frozen Champion and oracle contexts are required")
    frozen = contexts["frozen_champion"]
    oracle = contexts["oracle"]
    frozen_baseline = frozen["baseline"]
    frozen_candidate = frozen["candidate"]
    thresholds = summary["threshold_metrics"]
    gates = {
        "frozen_context_macro_f1_strictly_improved": frozen["delta"]["macro_f1"] > 0.0,
        "frozen_context_urgent_f1_strictly_improved": frozen["delta"]["urgent_f1"] > 0.0,
        "frozen_context_accuracy_noninferior": frozen_candidate["accuracy"] >= frozen_baseline["accuracy"],
        "normal_f1_not_below_control_minus_0p01": frozen_candidate["per_class"]["Normal"]["f1"] >= frozen_baseline["per_class"]["Normal"]["f1"] - 0.01,
        "critical_f1_not_below_control_minus_0p01": frozen_candidate["per_class"]["Critical"]["f1"] >= frozen_baseline["per_class"]["Critical"]["f1"] - 0.01,
        "f1_3mm_noninferior": thresholds["candidate"]["f1_3mm"] >= thresholds["baseline"]["f1_3mm"],
        "f1_5mm_noninferior": thresholds["candidate"]["f1_5mm"] >= thresholds["baseline"]["f1_5mm"],
        "oracle_macro_and_urgent_directions_nonnegative": oracle["delta"]["macro_f1"] >= 0.0 and oracle["delta"]["urgent_f1"] >= 0.0,
    }
    failed = [name for name in EXPECTED_GATES if not gates[name]]
    passed = not failed
    result = {
        "schema_version": 1,
        "status": "passed_for_a4_folds_1_2_development_expansion" if passed else "rejected_stop_a4_cross_fold_expansion",
        "screen_scope": "fold0_three_seed_early_rejection_only",
        "candidate": EXPECTED_STAGE,
        "fold": 0,
        "studies": 70,
        "aggregate_summary": str(aggregate_summary),
        "aggregate_summary_sha256": _sha256(aggregate_summary),
        "preregistration": str(preregistration),
        "preregistration_sha256": _sha256(preregistration),
        "baseline_source": baseline_source,
        "candidate_source": candidate_source,
        "source_contract": control_checks,
        "gates": gates,
        "failed_gates": failed,
        "can_expand_to_folds_1_2": passed,
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    _atomic_json(output.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.aggregate_summary, args.preregistration, args.output)
    print(json.dumps({"status": result["status"], "failed_gates": result["failed_gates"], "can_expand_to_folds_1_2": result["can_expand_to_folds_1_2"], "promotion_eligible": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
