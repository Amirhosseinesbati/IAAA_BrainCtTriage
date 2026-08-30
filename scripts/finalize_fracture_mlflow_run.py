"""Recover a completed fracture experiment after a tracking-network failure."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlflow


def _study_metrics(payload: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    pooling = payload.get("pooling", {})
    if not isinstance(pooling, dict):
        return result
    for method, values in pooling.items():
        if not isinstance(values, dict):
            continue
        auc = values.get("auc")
        threshold = values.get("threshold_0_5", {})
        if isinstance(auc, (int, float)) and math.isfinite(float(auc)):
            result[f"study_{method}_auc"] = float(auc)
        if isinstance(threshold, dict):
            f1 = threshold.get("f1")
            if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
                result[f"study_{method}_f1_at_0_5"] = float(f1)
    return result


def _log_with_retry(path: Path, artifact_path: str, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            delay_s = min(5 * (2 ** (attempt - 1)), 30)
            print(
                f"Upload failed for {path.name} ({attempt}/{attempts}); "
                f"retrying in {delay_s}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument(
        "--defer-model-artifacts",
        action="store_true",
        help="Recover metrics/diagnostics now and leave large model uploads deferred.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Recover metrics/tags only; defer all artifacts to a direct uploader.",
    )
    args = parser.parse_args()

    study_dir = args.run_dir / "study_evaluation"
    weights_dir = args.run_dir / "weights"
    metrics_path = study_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    for name in ("best.pt", "last.pt"):
        if not (weights_dir / name).is_file():
            raise FileNotFoundError(weights_dir / name)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = _study_metrics(payload)
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("recovered_after_tracking_failure", "true")
        if args.metrics_only:
            mlflow.set_tags({
                "artifact_status": "deferred",
                "artifact_reason": "remote_object_store_unreachable",
            })
        else:
            for path in sorted(study_dir.iterdir()):
                if path.is_file():
                    _log_with_retry(path, "study_evaluation", args.attempts)
            for path in sorted(args.run_dir.glob("*.png")):
                _log_with_retry(path, "plots", args.attempts)
        if args.metrics_only or args.defer_model_artifacts:
            mlflow.set_tags({
                "model_artifact_status": "deferred",
                "model_artifact_reason": (
                    "remote_object_store_unreachable"
                    if args.metrics_only
                    else "bandwidth_constrained_tracking_transport"
                ),
            })
        else:
            for name in ("best.pt", "last.pt"):
                _log_with_retry(weights_dir / name, "models", args.attempts)
        if not args.metrics_only:
            mlflow.log_dict(
                {"run_dir": str(args.run_dir), "study_metrics": metrics},
                "recovery_summary.json",
            )
    print(json.dumps({"run_id": args.run_id, "study_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
