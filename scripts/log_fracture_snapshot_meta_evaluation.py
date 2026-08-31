"""Log aggregate-only fracture snapshot evidence to remote MLflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.mlops.tracking import ExperimentContext, experiment_run, log_run_summary


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-summary", type=Path, required=True)
    parser.add_argument("--decision-summary", type=Path, required=True)
    parser.add_argument("--threshold-summary", type=Path, required=True)
    parser.add_argument("--incumbent-meta-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-name", default="fracture-snapshot-decision-preserving-fixed040-v1"
    )
    args = parser.parse_args()

    snapshot = _load(args.snapshot_summary)
    decision = _load(args.decision_summary)
    threshold = _load(args.threshold_summary)
    snapshot_bootstrap = snapshot["bootstrap"]
    decision_bootstrap = decision["bootstrap"]
    if not isinstance(snapshot_bootstrap, dict) or not isinstance(
        decision_bootstrap, dict
    ):
        raise TypeError("Bootstrap summaries must be objects")

    context = ExperimentContext(
        task_key="fracture",
        run_name=args.run_name,
        run_config={
            "validation": "five_fold_patient_disjoint_oof",
            "snapshot_epochs": [10, 15],
            "snapshot_slice_weight": 0.5,
            "snapshot_pooling": "top5_mean",
            "snapshot_fusion_weight": float(snapshot["fixed_weight"]),
            "calibration": "outer_train_empirical_cdf",
            "decision_mapping": "incumbent_decision_preserving_snapshot_ranking",
            "bootstrap_iterations": int(decision_bootstrap["iterations"]),
            "incumbent_meta_run_id": args.incumbent_meta_run_id,
        },
        strategy="yolov8s-epoch10-15-snapshot-plus-sa-mil",
        tags={
            "stage": "snapshot-meta-evaluation",
            "candidate_status": "validated_ranking_research",
            "aggregate_only": "true",
            "private_predictions_logged": "false",
        },
        notes=(
            "Leakage-controlled snapshot fusion evaluation. Per-study predictions and "
            "outer-train calibration arrays remain private on the Vast workspace."
        ),
    )
    decision_interval = decision_bootstrap["difference_95"]
    snapshot_interval = snapshot_bootstrap["difference_95"]
    metrics = {
        "incumbent_macro_auc": float(decision["reference_macro_auc"]),
        "snapshot_fusion_macro_auc": float(snapshot["fusion_macro_auc"]),
        "snapshot_fusion_worst_fold_auc": float(snapshot["fusion_worst_fold_auc"]),
        "decision_preserving_crossfit_macro_auc": float(
            decision["crossfit_candidate_macro_auc"]
        ),
        "decision_preserving_crossfit_difference": float(
            decision["crossfit_macro_difference"]
        ),
        "decision_preserving_worst_fold_auc": float(
            decision["crossfit_candidate_worst_fold_auc"]
        ),
        "decision_preserving_deployment_macro_auc": float(
            decision["deployment_candidate_macro_auc"]
        ),
        "decision_preserving_crossfit_f1": float(
            decision["crossfit_classification"]["f1"]
        ),
        "decision_preserving_ci95_lower": float(decision_interval[0]),
        "decision_preserving_ci95_median": float(decision_interval[1]),
        "decision_preserving_ci95_upper": float(decision_interval[2]),
        "decision_preserving_probability_not_better": float(
            decision_bootstrap["probability_candidate_not_better"]
        ),
        "snapshot_fusion_ci95_lower": float(snapshot_interval[0]),
        "snapshot_fusion_ci95_upper": float(snapshot_interval[2]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with experiment_run(context) as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "incumbent_meta_run_id": args.incumbent_meta_run_id,
                "selection_gate": "no_direct_triage_gain_decisions_preserved",
                "incumbent_decisions_preserved_crossfit": str(
                    decision["incumbent_decisions_preserved"]
                ).lower(),
            }
        )
        for path in (
            args.snapshot_summary,
            args.decision_summary,
            args.threshold_summary,
        ):
            mlflow.log_artifact(str(path), artifact_path="aggregate_evaluation")
        summary = {
            "run_id": run.info.run_id,
            "incumbent_meta_run_id": args.incumbent_meta_run_id,
            "metrics": metrics,
            "private_predictions_logged": False,
            "decision": "validated_ranking_research_no_direct_triage_gain",
        }
        log_run_summary(summary, "fracture_snapshot_meta_evaluation.json")
        (args.output / "mlflow_run.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
