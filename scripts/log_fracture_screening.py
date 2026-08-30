"""Attach checkpoint-screening evidence and decisions to an MLflow run."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow


def _pairs(values: list[str], kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid {kind} {value!r}; expected KEY=VALUE")
        key, item = value.split("=", 1)
        result[key] = item
    return result


def _log_with_retry(path: Path, artifact_path: str, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(5 * (2 ** (attempt - 1)), 30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--screening-dir", type=Path, required=True)
    parser.add_argument("--artifact-path", default="checkpoint_screening")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()

    if not args.screening_dir.is_dir():
        raise FileNotFoundError(args.screening_dir)
    tags = _pairs(args.tag, "tag")
    raw_metrics = _pairs(args.metric, "metric")
    metrics = {key: float(value) for key, value in raw_metrics.items()}

    with mlflow.start_run(run_id=args.run_id):
        if tags:
            mlflow.set_tags(tags)
        if metrics:
            mlflow.log_metrics(metrics)
        for path in sorted(args.screening_dir.rglob("*")):
            if not path.is_file() or path.name == "DONE":
                continue
            relative_parent = path.parent.relative_to(args.screening_dir)
            destination = str(Path(args.artifact_path) / relative_parent)
            _log_with_retry(path, destination, args.attempts)

    print(
        {
            "run_id": args.run_id,
            "screening_dir": str(args.screening_dir),
            "tags": tags,
            "metrics": metrics,
        }
    )


if __name__ == "__main__":
    main()
