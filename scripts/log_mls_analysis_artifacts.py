"""Upload aggregate MLS reports to an existing MLflow run.

Raw per-study predictions and medical image data are deliberately rejected.
This script performs no model computation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_section
from src.mlops.tracking import configure_tracking_environment


ALLOWED_NAMES = {
    "report.md",
    "epoch_metrics.jsonl",
    "metrics.json",
    "decomposition.json",
    "decomposition_report.md",
    "postprocessing_search.json",
    "crossfold_pooling_summary.json",
    "checkpoint_pooling_summary.json",
    "checkpoint_audit_report.md",
    "repro_gate_epoch15.json",
    "promotion_gate.json",
    "audit_status.json",
    "PREREGISTERED_PLAN.md",
    "POSTFAILURE_DIAGNOSTIC_PLAN.md",
    "e2e_aggregate_metrics.json",
    "target_analysis.json",
    "crossrun_component_blend_summary.json",
    "crossrun_component_blend_grid.csv",
    "CROSSRUN_COMPONENT_BLEND_SCREEN.md",
    "LOCKED_AUDIT_AND_FAILURE_ANALYSIS.md",
    "LOCKED_THIRDFOLD_FAILURE_ANALYSIS.md",
    "POSTFAILURE_NAMED_BEST_PLAN.md",
    "CONSERVATIVE_THREEFOLD_OOF_PLAN.md",
    "fixed_component_transfer_summary.json",
    "FROZEN_COMPONENT_TRANSFER_REPORT.md",
    "conservative_threefold_oof_summary.json",
    "CONSERVATIVE_THREEFOLD_OOF_REPORT.md",
    "CONSERVATIVE_PACKAGE_INTEGRATION_PLAN.md",
    "build.json",
    "full_package_smoke.json",
    "package_oof_audit_summary.json",
    "PACKAGE_OOF_AUDIT_REPORT.md",
    "status.json",
}

DENIED_NAMES = {
    "study_slice_predictions.csv",
    "selector_measurement_decomposition.csv",
    "screen_selected_predictions.csv",
    "study_member_predictions.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.experiment_dir.resolve()
    all_files = [path for path in root.rglob("*") if path.is_file()]
    artifacts = sorted({path for path in all_files if path.name in ALLOWED_NAMES})
    rejected = [path for path in all_files if path.name in DENIED_NAMES]
    if not artifacts:
        raise FileNotFoundError(f"No aggregate MLS artifacts found under {root}")

    configure_tracking_environment()
    client = MlflowClient()
    report_path = config_section("mlflow", "artifact_paths", "reports")
    for path in artifacts:
        if path.name not in ALLOWED_NAMES:
            raise ValueError(f"Artifact is not allowlisted: {path.name}")
        relative_parent = path.parent.relative_to(root).as_posix()
        destination = report_path if relative_parent == "." else f"{report_path}/{relative_parent}"
        client.log_artifact(args.run_id, str(path), destination)
    metrics_logged = {}
    root_metrics = next(
        (
            candidate
            for candidate in (
                root / "metrics.json",
                root / "e2e_aggregate_metrics.json",
            )
            if candidate in artifacts
        ),
        None,
    )
    if root_metrics is not None:
        payload = json.loads(root_metrics.read_text(encoding="utf-8"))
        metrics_logged = {
            f"analysis_{key}"[:250]: float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if metrics_logged:
            for key, value in metrics_logged.items():
                client.log_metric(args.run_id, key, value)
    oof_summary = next(
        (
            path
            for path in artifacts
            if path.name == "conservative_threefold_oof_summary.json"
        ),
        None,
    )
    if oof_summary is not None:
        payload = json.loads(oof_summary.read_text(encoding="utf-8"))
        baseline = payload["micro_oof"]["baseline"]
        candidate = payload["micro_oof"]["candidate"]
        delta = payload["micro_oof"]["delta"]
        oof_metrics = {
            "analysis_oof_baseline_mae_mm": float(baseline["mae_mm"]),
            "analysis_oof_candidate_mae_mm": float(candidate["mae_mm"]),
            "analysis_oof_delta_mae_mm": float(delta["mae_mm"]),
            "analysis_oof_baseline_boundary_f1": float(baseline["boundary_f1"]),
            "analysis_oof_candidate_boundary_f1": float(candidate["boundary_f1"]),
            "analysis_oof_delta_boundary_f1": float(delta["boundary_f1"]),
            "analysis_oof_baseline_objective": float(
                baseline["selection_objective"]
            ),
            "analysis_oof_candidate_objective": float(
                candidate["selection_objective"]
            ),
            "analysis_oof_delta_objective": float(
                delta["selection_objective"]
            ),
        }
        for key, value in oof_metrics.items():
            client.log_metric(args.run_id, key, value)
        metrics_logged.update(oof_metrics)
        client.set_tag(
            args.run_id,
            "conservative_threefold_oof_passed",
            str(bool(payload["passed"])).lower(),
        )
    package_summary = next(
        (
            path
            for path in artifacts
            if path.name == "package_oof_audit_summary.json"
        ),
        None,
    )
    if package_summary is not None:
        payload = json.loads(package_summary.read_text(encoding="utf-8"))
        baseline = payload["micro_oof"]["baseline"]
        candidate = payload["micro_oof"]["packaged_candidate"]
        package_metrics = {
            "analysis_package_oof_baseline_mae_mm": float(baseline["mae_mm"]),
            "analysis_package_oof_candidate_mae_mm": float(candidate["mae_mm"]),
            "analysis_package_oof_baseline_boundary_f1": float(
                baseline["boundary_f1"]
            ),
            "analysis_package_oof_candidate_boundary_f1": float(
                candidate["boundary_f1"]
            ),
            "analysis_package_oof_baseline_objective": float(
                baseline["selection_objective"]
            ),
            "analysis_package_oof_candidate_objective": float(
                candidate["selection_objective"]
            ),
            "analysis_package_runtime_total_s": float(payload["runtime_total_s"]),
            "analysis_package_peak_vram_gb": float(payload["peak_vram_gb"]),
            "analysis_package_max_member_delta_mm": float(
                payload["parity_maxima"]["max_abs_member_value_mm"]
            ),
        }
        for key, value in package_metrics.items():
            client.log_metric(args.run_id, key, value)
        metrics_logged.update(package_metrics)
        client.set_tag(
            args.run_id,
            "conservative_package_oof_audit_passed",
            str(bool(payload["passed"])).lower(),
        )
    client.set_tag(args.run_id, "aggregate_analysis_artifacts_uploaded", "true")
    print(json.dumps({
        "run_id": args.run_id,
        "uploaded": [str(path.relative_to(root)) for path in artifacts],
        "metrics_logged": sorted(metrics_logged),
        "raw_artifacts_excluded": [str(path.relative_to(root)) for path in rejected],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
