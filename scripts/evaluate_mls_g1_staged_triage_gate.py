"""Fail-closed staged triage gate for the preregistered G1 C0-vs-2.5D test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = "deploy_aligned_fixed_three_seed_median_canonical_triage"
GATES = (
    "frozen_macro_f1_strictly_improved",
    "frozen_urgent_f1_strictly_improved",
    "frozen_accuracy_noninferior",
    "normal_f1_not_below_minus_0p01",
    "critical_f1_not_below_minus_0p01",
    "f1_3mm_noninferior",
    "f1_5mm_noninferior",
    "catastrophic_errors_not_worse",
    "oracle_macro_and_urgent_directions_nonnegative",
)
STAGES = {
    "fold3_screen": {"fold": 3, "studies": 66},
    "fold4_confirmation": {"fold": 4, "studies": 68},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON contract: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON contract must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source(summary: dict[str, Any], name: str, *, fold: int, studies: int) -> dict[str, Any]:
    rows = summary.get("sources", {}).get(name)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"G1 triage summary must have exactly one {name} source")
    source = rows[0]
    if int(source.get("fold", -1)) != fold or int(source.get("studies", -1)) != studies:
        raise ValueError(f"G1 {name} source does not match the preregistered held-out fold")
    for key in ("sha256", "audit_summary_sha256"):
        digest = source.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"G1 {name}.{key} is not a lowercase SHA-256 digest")
    return source


def _load_g1_arm(source: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(source.get("audit_summary_path", ""))).resolve()
    if not summary_path.is_file() or _sha256(summary_path) != source["audit_summary_sha256"]:
        raise ValueError("G1 per-fold evaluator summary is missing or checksum-mismatched")
    payload = _read_json(summary_path)
    prediction_path = Path(str(source.get("path", ""))).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"G1 private prediction artifact is missing: {prediction_path}")
    prediction_sha256 = _sha256(prediction_path)
    if (
        prediction_sha256 != source["sha256"]
        or payload.get("private_predictions_sha256") != prediction_sha256
    ):
        raise ValueError(
            "G1 aggregate/private predictions are not bound to the audited evaluator output"
        )
    required = {
        "status": "completed",
        "protocol": "heldout_fold_fixed_epoch15_three_distinct_seed_median",
        "campaign": "g1_2p5d_deploy_aligned",
        "arm": expected["arm"],
        "dataset_variant": "multitask_2p5d_v1",
        "input_channels": expected["input_channels"],
        "cache_manifest_sha256": expected["cache_manifest_sha256"],
        "model_config_signature_sha256": expected["model_config_signature_sha256"],
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"G1 per-fold evaluator {key} does not match preregistration")
    checkpoints = payload.get("checkpoint_manifest")
    if not isinstance(checkpoints, dict) or len(checkpoints) != 3:
        raise ValueError("G1 per-fold evaluator does not bind three checkpoints")
    hashes = [item.get("sha256") for item in checkpoints.values() if isinstance(item, dict)]
    if len(hashes) != 3 or len(set(hashes)) != 3:
        raise ValueError("G1 per-fold evaluator checkpoints are not three distinct members")
    if payload.get("config_differences") != ["seed"]:
        raise ValueError("G1 seed ensemble differs in a field other than seed")
    return payload


def _validate_preregistration(prereg: dict[str, Any], stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if prereg.get("status") != "locked_before_any_g1_cuda_outcome":
        raise ValueError("G1 preregistration is not locked")
    if prereg.get("campaign") != "g1_2p5d_deploy_aligned":
        raise ValueError("Unexpected G1 campaign preregistration")
    if tuple(prereg.get("fixed_screen_gates", ())) != GATES:
        raise ValueError("G1 staged gates differ from the executable contract")
    expected_stage = STAGES[stage]
    declared = prereg.get("stages", {}).get(stage, {})
    if declared != expected_stage:
        raise ValueError("G1 preregistration has an unexpected fold/study stage contract")
    arms = prereg.get("arms", {})
    control = arms.get("control")
    candidate = arms.get("candidate")
    if not isinstance(control, dict) or not isinstance(candidate, dict):
        raise ValueError("G1 preregistration must describe control and candidate arms")
    expected_common = {
        "dataset_variant": "multitask_2p5d_v1",
        "cache_manifest_sha256": prereg.get("cache_manifest_sha256"),
        "fixed_epoch": 15,
        "seeds": [42, 2026, 3407],
    }
    for arm, expected_arm, channels in (
        (control, "g1_c0_3ch", 3),
        (candidate, "g1_a_9ch", 9),
    ):
        if arm.get("arm") != expected_arm or int(arm.get("input_channels", -1)) != channels:
            raise ValueError("G1 preregistration arm/channel identity is invalid")
        for key, value in expected_common.items():
            if arm.get(key) != value:
                raise ValueError(f"G1 preregistration arm differs in {key}")
        digest = arm.get("model_config_signature_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("G1 preregistration lacks a config signature digest")
    control_signature = control.get("model_config_signature")
    candidate_signature = candidate.get("model_config_signature")
    if not isinstance(control_signature, dict) or not isinstance(candidate_signature, dict):
        raise ValueError("G1 preregistration lacks seed-free config payloads")
    different = {
        key for key in control_signature.keys() | candidate_signature.keys()
        if control_signature.get(key) != candidate_signature.get(key)
    }
    if different != {"input_channels"}:
        raise ValueError(
            "G1 control/candidate must differ only in input_channels, got "
            f"{sorted(different)}"
        )
    return control, candidate


def evaluate(
    *,
    aggregate_summary: Path,
    preregistration: Path,
    stage: str,
    output: Path,
    prior_fold3_gate: Path | None,
) -> dict[str, Any]:
    summary = _read_json(aggregate_summary.resolve())
    prereg = _read_json(preregistration.resolve())
    stage_contract = STAGES[stage]
    control, candidate = _validate_preregistration(prereg, stage)
    if int(summary.get("schema_version", -1)) != 1 or summary.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected canonical deploy-aligned triage protocol")
    if summary.get("selected_folds") != [stage_contract["fold"]] or int(summary.get("studies", -1)) != stage_contract["studies"]:
        raise ValueError("G1 staged triage summary does not exactly cover its preregistered fold")
    if summary.get("evaluation_scope") != "development_oof_subset" or bool(summary.get("full_fold_coverage")):
        raise ValueError("A G1 staged screen may not claim full OOF coverage")
    if bool(summary.get("promotion_eligible")):
        raise ValueError("A G1 staged screen may not claim promotion eligibility")
    if stage == "fold4_confirmation":
        if prior_fold3_gate is None:
            raise ValueError("G1 fold4 confirmation requires the passed fold3 gate receipt")
        prior_fold3_gate = prior_fold3_gate.resolve()
        earlier = _read_json(prior_fold3_gate)
        required_prior = {
            "status": "passed_for_g1_fold4_confirmation",
            "campaign": "g1_2p5d_deploy_aligned",
            "stage": "fold3_screen",
            "fold": STAGES["fold3_screen"]["fold"],
            "studies": STAGES["fold3_screen"]["studies"],
            "can_start_fold4": True,
            "preregistration_sha256": _sha256(preregistration.resolve()),
        }
        for key, value in required_prior.items():
            if earlier.get(key) != value:
                raise ValueError(
                    "G1 fold4 confirmation requires the matching passed fold3 gate "
                    f"receipt; mismatch in {key}"
                )
        prior_fold3_gate_binding: dict[str, str] | None = {
            "path": str(prior_fold3_gate),
            "sha256": _sha256(prior_fold3_gate),
        }
    else:
        prior_fold3_gate_binding = None

    baseline_source = _source(summary, "baseline_folds", **stage_contract)
    candidate_source = _source(summary, "candidate_folds", **stage_contract)
    baseline_eval = _load_g1_arm(baseline_source, control)
    candidate_eval = _load_g1_arm(candidate_source, candidate)
    if baseline_source["sha256"] == candidate_source["sha256"]:
        raise ValueError("G1 control and candidate point to the same private predictions")
    contexts = summary.get("contexts", {})
    if set(contexts) != {"frozen_champion", "oracle"}:
        raise ValueError("G1 staged triage requires frozen and oracle contexts")
    frozen = contexts["frozen_champion"]
    oracle = contexts["oracle"]
    frozen_baseline = frozen["baseline"]
    frozen_candidate = frozen["candidate"]
    thresholds = summary["threshold_metrics"]
    gates = {
        "frozen_macro_f1_strictly_improved": frozen["delta"]["macro_f1"] > 0.0,
        "frozen_urgent_f1_strictly_improved": frozen["delta"]["urgent_f1"] > 0.0,
        "frozen_accuracy_noninferior": frozen_candidate["accuracy"] >= frozen_baseline["accuracy"],
        "normal_f1_not_below_minus_0p01": frozen_candidate["per_class"]["Normal"]["f1"] >= frozen_baseline["per_class"]["Normal"]["f1"] - 0.01,
        "critical_f1_not_below_minus_0p01": frozen_candidate["per_class"]["Critical"]["f1"] >= frozen_baseline["per_class"]["Critical"]["f1"] - 0.01,
        "f1_3mm_noninferior": thresholds["candidate"]["f1_3mm"] >= thresholds["baseline"]["f1_3mm"],
        "f1_5mm_noninferior": thresholds["candidate"]["f1_5mm"] >= thresholds["baseline"]["f1_5mm"],
        "catastrophic_errors_not_worse": (
            frozen_candidate["catastrophic_errors"]["normal_to_critical"]
            <= frozen_baseline["catastrophic_errors"]["normal_to_critical"]
            and frozen_candidate["catastrophic_errors"]["critical_to_normal"]
            <= frozen_baseline["catastrophic_errors"]["critical_to_normal"]
        ),
        "oracle_macro_and_urgent_directions_nonnegative": (
            oracle["delta"]["macro_f1"] >= 0.0 and oracle["delta"]["urgent_f1"] >= 0.0
        ),
    }
    failed = [name for name in GATES if not gates[name]]
    passed = not failed
    status = (
        "passed_for_g1_fold4_confirmation" if stage == "fold3_screen" and passed
        else "passed_g1_crossfold_confirmation" if stage == "fold4_confirmation" and passed
        else "rejected_stop_g1_crossfold_expansion"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "campaign": "g1_2p5d_deploy_aligned",
        "stage": stage,
        "fold": stage_contract["fold"],
        "studies": stage_contract["studies"],
        "aggregate_summary": str(aggregate_summary.resolve()),
        "aggregate_summary_sha256": _sha256(aggregate_summary.resolve()),
        "preregistration": str(preregistration.resolve()),
        "preregistration_sha256": _sha256(preregistration.resolve()),
        "prior_fold3_gate": prior_fold3_gate_binding,
        "baseline_source": baseline_source,
        "candidate_source": candidate_source,
        "baseline_evaluator_sha256": _sha256(Path(str(baseline_source["audit_summary_path"]))),
        "candidate_evaluator_sha256": _sha256(Path(str(candidate_source["audit_summary_path"]))),
        "baseline_arm_config_signature_sha256": baseline_eval["model_config_signature_sha256"],
        "candidate_arm_config_signature_sha256": candidate_eval["model_config_signature_sha256"],
        "gates": gates,
        "failed_gates": failed,
        "can_start_fold4": bool(stage == "fold3_screen" and passed),
        "crossfold_confirmed": bool(stage == "fold4_confirmation" and passed),
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    _atomic_json(output.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--prior-fold3-gate", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        aggregate_summary=args.aggregate_summary,
        preregistration=args.preregistration,
        stage=args.stage,
        prior_fold3_gate=args.prior_fold3_gate,
        output=args.output,
    )
    print(json.dumps({
        "status": result["status"], "stage": result["stage"],
        "failed_gates": result["failed_gates"], "promotion_eligible": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
