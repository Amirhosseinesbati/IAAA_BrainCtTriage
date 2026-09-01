"""Replay MLflow operations deferred by a transient network outage."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

PROJECT_ROOT_HINT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

from src.config import PROJECT_ROOT
from src.mlops import configure_tracking_environment
from src.mlops.tracking import build_source_snapshot


def _default_queue() -> Path:
    configured = os.getenv("IAAA_MLFLOW_PENDING_QUEUE", "").strip()
    return Path(configured) if configured else PROJECT_ROOT / "reports" / "mlflow_pending_events.jsonl"


def _replay_payload(payload: dict[str, Any]) -> None:
    filename = str(payload["filename"])
    artifact_path = str(payload["artifact_path"])
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / filename
        if payload.get("yaml_format"):
            target.write_text(
                yaml.safe_dump(payload["content"], sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            target.write_text(
                json.dumps(payload["content"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        mlflow.log_artifact(str(target), artifact_path=artifact_path)


def _replay_event(event: dict[str, Any]) -> None:
    operation = event["operation"]
    payload = event.get("payload", {})
    if operation == "log_metrics":
        mlflow.log_metrics(payload["metrics"], step=int(payload["step"]))
    elif operation == "log_params":
        mlflow.log_params(payload["params"])
    elif operation == "set_tags":
        mlflow.set_tags(payload["tags"])
    elif operation == "log_artifact":
        local_path = Path(payload["local_path"])
        if not local_path.is_file():
            raise FileNotFoundError(f"Deferred artifact is missing: {local_path}")
        mlflow.log_artifact(
            str(local_path), artifact_path=str(payload["artifact_path"]),
        )
    elif operation == "log_payload":
        _replay_payload(payload)
    elif operation == "log_source_snapshot":
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "project_source.zip"
            build_source_snapshot(target)
            mlflow.log_artifact(str(target), artifact_path="code")
    elif operation == "end_run":
        return
    else:
        raise ValueError(f"Unsupported deferred MLflow operation: {operation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=_default_queue())
    args = parser.parse_args()
    queue = args.queue.resolve()
    if not queue.is_file():
        print(f"No deferred MLflow queue found: {queue}")
        return

    configure_tracking_environment()
    events = [
        json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remaining: list[dict[str, Any]] = []
    replayed = 0
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            event["replay_error"] = "missing run_id"
            remaining.append(event)
            continue
        try:
            with mlflow.start_run(run_id=str(run_id)):
                _replay_event(event)
            replayed += 1
        except Exception as exc:
            event["replay_error"] = f"{type(exc).__name__}: {exc}"[:2000]
            remaining.append(event)

    temporary = queue.with_suffix(queue.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining),
        encoding="utf-8",
    )
    os.replace(temporary, queue)
    print(f"replayed={replayed} remaining={len(remaining)} queue={queue}")
    if remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
