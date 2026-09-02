"""Log a deploy-aligned MLS aggregate evaluation as a separate MLflow run.

Only the canonical aggregate JSON is uploaded.  Per-study predictions remain
private on the evaluation host and are never accepted as artifacts here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.tracking import ExperimentContext, experiment_run, log_run_summary


PUBLIC_ARTIFACT_NAME = "aggregate_summary.json"
PRIVATE_ARTIFACT_NAMES = {
    "per_study_private.csv",
    "study_member_predictions_private.csv",
    "study_member_predictions.csv",
    "study_slice_predictions.csv",
}


def _load_and_validate(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.name in PRIVATE_ARTIFACT_NAMES or path.name != PUBLIC_ARTIFACT_NAME:
        raise ValueError(
            f"Only {PUBLIC_ARTIFACT_NAME!r} may be logged; got {path.name!r}"
        )
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Aggregate summary is unexpectedly large; refusing upload")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Aggregate summary must be a JSON object")
    required = {
        "schema_version",
        "protocol",
        "evaluation_scope",
        "development_gate_passed",
        "promotion_eligible",
        "selected_folds",
        "available_folds",
        "full_fold_coverage",
        "sources",
        "studies",
        "threshold_metrics",
        "contexts",
        "promotion_gates",
        "failed_hard_gates",
    }
    if set(payload) != required:
        raise ValueError(
            "Aggregate summary fields differ from the canonical schema: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    if int(payload["schema_version"]) != 1:
        raise ValueError("Unsupported aggregate summary schema_version")
    if payload["protocol"] != "deploy_aligned_fixed_three_seed_median_canonical_triage":
        raise ValueError("Unexpected MLS evaluation protocol")
    selected = [int(value) for value in payload["selected_folds"]]
    available = [int(value) for value in payload["available_folds"]]
    if selected != sorted(set(selected)) or available != sorted(set(available)):
        raise ValueError("Fold lists must be sorted and unique")
    contexts = payload["contexts"]
    if not isinstance(contexts, dict) or set(contexts) != {"frozen_champion", "oracle"}:
        raise ValueError("Both frozen_champion and oracle contexts are required")
    gates = payload["promotion_gates"]
    if not isinstance(gates, dict) or not gates or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("promotion_gates must be a non-empty boolean mapping")
    failed = payload["failed_hard_gates"]
    if not isinstance(failed, list) or any(name not in gates for name in failed):
        raise ValueError("failed_hard_gates must reference known gates")
    if bool(payload["promotion_eligible"]):
        if not bool(payload["full_fold_coverage"]):
            raise ValueError("Promotion eligibility requires full fold coverage")
        if payload["evaluation_scope"] != "full_oof" or selected != available:
            raise ValueError("Promotion eligibility requires the complete OOF scope")
        if int(payload["studies"]) != 338:
            raise ValueError("Promotion eligibility requires all 338 immutable studies")
        if failed:
            raise ValueError("A promotion-eligible summary cannot contain failed hard gates")
        if not all(gates.values()):
            raise ValueError("A promotion-eligible summary requires every promotion gate")
    return payload


def _context_metrics(prefix: str, context: dict[str, Any]) -> dict[str, float]:
    baseline = context["baseline"]
    candidate = context["candidate"]
    delta = context["delta"]
    bootstrap = context["paired_patient_bootstrap"]
    result = {
        f"{prefix}_baseline_macro_f1": float(baseline["macro_f1"]),
        f"{prefix}_candidate_macro_f1": float(candidate["macro_f1"]),
        f"{prefix}_delta_macro_f1": float(delta["macro_f1"]),
        f"{prefix}_baseline_accuracy": float(baseline["accuracy"]),
        f"{prefix}_candidate_accuracy": float(candidate["accuracy"]),
        f"{prefix}_delta_accuracy": float(delta["accuracy"]),
        f"{prefix}_baseline_urgent_f1": float(baseline["per_class"]["Urgent"]["f1"]),
        f"{prefix}_candidate_urgent_f1": float(candidate["per_class"]["Urgent"]["f1"]),
        f"{prefix}_delta_urgent_f1": float(delta["urgent_f1"]),
        f"{prefix}_bootstrap_probability_improvement": float(
            bootstrap["probability_of_improvement"]
        ),
        f"{prefix}_bootstrap_ci95_low": float(bootstrap["ci95_low"]),
        f"{prefix}_bootstrap_ci95_high": float(bootstrap["ci95_high"]),
        f"{prefix}_changed_triage_decisions": float(context["changed_triage_decisions"]),
    }
    for class_name in ("Normal", "Urgent", "Critical"):
        key = class_name.lower()
        result[f"{prefix}_baseline_{key}_f1"] = float(
            baseline["per_class"][class_name]["f1"]
        )
        result[f"{prefix}_candidate_{key}_f1"] = float(
            candidate["per_class"][class_name]["f1"]
        )
    return result


def _metric_payload(payload: dict[str, Any]) -> dict[str, float]:
    thresholds = payload["threshold_metrics"]
    metrics = {
        "evaluated_studies": float(payload["studies"]),
        **_context_metrics("frozen", payload["contexts"]["frozen_champion"]),
        **_context_metrics("oracle", payload["contexts"]["oracle"]),
    }
    for branch in ("baseline", "candidate", "delta"):
        for threshold in (1, 3, 5):
            name = f"f1_{threshold}mm"
            metrics[f"mls_{branch}_{name}"] = float(thresholds[branch][name])
    return metrics


def _source_hashes(payload: dict[str, Any]) -> dict[str, str]:
    sources = payload["sources"]
    result: dict[str, str] = {}
    for branch in ("baseline_folds", "candidate_folds"):
        rows = sources[branch]
        if not isinstance(rows, list):
            raise TypeError(f"sources.{branch} must be a list")
        for row in rows:
            fold = int(row["fold"])
            result[f"{branch}_fold{fold}_sha256"] = str(row["sha256"])
    for name in ("frozen_champion_predictions", "truth_table", "fold_manifest"):
        result[f"{name}_sha256"] = str(sources[name]["sha256"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="mls-deploy-aligned-meta-evaluation")
    args = parser.parse_args()

    payload = _load_and_validate(args.aggregate_summary)
    metrics = _metric_payload(payload)
    source_hashes = _source_hashes(payload)
    context = ExperimentContext(
        task_key="mls",
        run_name=args.run_name,
        run_config={
            "protocol": payload["protocol"],
            "evaluation_scope": payload["evaluation_scope"],
            "selected_folds": payload["selected_folds"],
            "available_folds": payload["available_folds"],
            "source_hashes": source_hashes,
        },
        strategy="deploy-aligned-fixed-three-seed-median",
        tags={
            "stage": "meta-evaluation",
            "development_gate_passed": str(bool(payload["development_gate_passed"])).lower(),
            "promotion_eligible": str(bool(payload["promotion_eligible"])).lower(),
            "full_fold_coverage": str(bool(payload["full_fold_coverage"])).lower(),
            "raw_medical_predictions_uploaded": "false",
        },
        notes=(
            "Leak-free deploy-aligned comparison of fixed three-seed MLS medians. "
            "Only aggregate metrics and provenance hashes are uploaded."
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with experiment_run(context) as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            f"gate_{name}": str(bool(value)).lower()
            for name, value in payload["promotion_gates"].items()
        })
        mlflow.log_artifact(
            str(args.aggregate_summary.resolve()),
            artifact_path="reports/meta_evaluation",
        )
        summary = {
            "run_id": run.info.run_id,
            "evaluation_scope": payload["evaluation_scope"],
            "selected_folds": payload["selected_folds"],
            "studies": payload["studies"],
            "development_gate_passed": payload["development_gate_passed"],
            "promotion_eligible": payload["promotion_eligible"],
            "failed_hard_gates": payload["failed_hard_gates"],
            "source_hashes": source_hashes,
            "metrics": metrics,
            "uploaded_artifacts": [PUBLIC_ARTIFACT_NAME],
            "raw_medical_predictions_uploaded": False,
        }
        log_run_summary(summary, "mls_deploy_aligned_meta_evaluation.json")
        output = args.output_dir / "mlflow_meta_run.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
