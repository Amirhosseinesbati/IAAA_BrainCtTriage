"""Log the five-fold fracture MIL meta-evaluation to remote MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from src.mlops.tracking import ExperimentContext, experiment_run, log_run_summary


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _metric_payload(
    prefix: str, payload: dict[str, object]
) -> dict[str, float]:
    result = {
        f"{prefix}_macro_auc": float(payload["blend_macro_auc"]),
        f"{prefix}_macro_difference": float(payload["macro_difference"]),
        f"{prefix}_worst_fold_auc": float(payload["blend_worst_fold_auc"]),
    }
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise TypeError("bootstrap must be a JSON object")
    interval = bootstrap["difference_95"]
    if not isinstance(interval, list) or len(interval) != 3:
        raise ValueError("difference_95 must contain lower, median and upper")
    result.update(
        {
            f"{prefix}_difference_ci95_lower": float(interval[0]),
            f"{prefix}_difference_ci95_median": float(interval[1]),
            f"{prefix}_difference_ci95_upper": float(interval[2]),
            f"{prefix}_probability_not_better": float(
                bootstrap["probability_blend_not_better"]
            ),
        }
    )
    per_fold = payload["per_fold"]
    if not isinstance(per_fold, list):
        raise TypeError("per_fold must be a list")
    for row in per_fold:
        if not isinstance(row, dict):
            raise TypeError("per_fold entries must be objects")
        result[f"{prefix}_fold{int(row['fold'])}_auc"] = float(row["blend_auc"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--rank-crossfit", type=Path, required=True)
    parser.add_argument("--deployable-crossfit", type=Path, required=True)
    parser.add_argument("--deployable-fixed", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="fracture-sa-mil-five-fold-meta-v2")
    args = parser.parse_args()

    standalone = _load_json(args.standalone)
    rank_crossfit = _load_json(args.rank_crossfit)
    deployable_crossfit = _load_json(args.deployable_crossfit)
    deployable_fixed = _load_json(args.deployable_fixed)

    source_run_ids: dict[int, str] = {}
    for fold in range(5):
        identity = _load_json(args.model_root / f"fold_{fold}_v2" / "mlflow_run.json")
        source_run_ids[fold] = str(identity["run_id"])

    context = ExperimentContext(
        task_key="fracture",
        run_name=args.run_name,
        run_config={
            "validation": "five_fold_patient_disjoint_oof",
            "bootstrap_iterations": int(deployable_fixed["bootstrap"]["iterations"]),
            "fixed_candidate_weight": 0.45,
            "calibration": "outer_train_empirical_cdf",
            "source_run_ids": source_run_ids,
        },
        strategy="yolov8s-adjacent-plus-smooth-attention-mil",
        tags={"stage": "five-fold-meta-evaluation", "candidate_status": "promising"},
        notes=(
            "Leakage-controlled five-fold comparison of standalone SA-MIL, cohort-rank "
            "diagnostic blends, and per-study deployable outer-train CDF blends."
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with experiment_run(context) as run:
        client = mlflow.MlflowClient()
        source_status = {
            fold: client.get_run(run_id).info.status
            for fold, run_id in source_run_ids.items()
        }
        if any(status != "FINISHED" for status in source_status.values()):
            raise RuntimeError(f"Non-finished source runs: {source_status}")
        metrics = {
            "reference_macro_auc": float(deployable_fixed["reference_macro_auc"]),
            "reference_worst_fold_auc": float(
                deployable_fixed["reference_worst_fold_auc"]
            ),
            "standalone_mil_macro_auc": float(standalone["candidate_macro_auc"]),
            "rank_crossfit_macro_auc": float(rank_crossfit["blend_macro_auc"]),
            **_metric_payload("deployable_crossfit", deployable_crossfit),
            **_metric_payload("deployable_fixed045", deployable_fixed),
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "source_runs_all_finished": "true",
                "selection_gate": "pending_packaged_inference_and_runtime_validation",
                **{
                    f"source_fold_{fold}_run_id": run_id
                    for fold, run_id in source_run_ids.items()
                },
            }
        )
        for path in (
            args.standalone.parent,
            args.rank_crossfit.parent,
            args.deployable_crossfit.parent,
            args.deployable_fixed.parent,
        ):
            mlflow.log_artifacts(str(path), artifact_path=f"meta_evaluation/{path.name}")
        summary = {
            "run_id": run.info.run_id,
            "source_run_ids": source_run_ids,
            "source_status": source_status,
            "metrics": metrics,
            "decision": "promising_pending_packaged_inference_and_runtime_validation",
        }
        log_run_summary(summary, "fracture_mil_meta_evaluation.json")
        (args.output / "mlflow_meta_run.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
