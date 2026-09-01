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
    "e2e_aggregate_metrics.json",
    "target_analysis.json",
}

DENIED_NAMES = {
    "study_slice_predictions.csv",
    "selector_measurement_decomposition.csv",
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
