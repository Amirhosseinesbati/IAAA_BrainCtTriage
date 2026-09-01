"""Recover a completed MLS run from a console-encoding teardown failure.

This script performs no model inference or training.  It verifies the existing
local artifacts, marks the existing MLflow run FINISHED, and synchronizes the
completed report without creating a duplicate run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.config import PROJECT_ROOT, config_section
from src.mlops.tracking import configure_tracking_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Required artifact missing or empty: {resolved}")
    return resolved


def completed_report(text: str) -> str:
    text = text.replace("- Status: `failed`", "- Status: `completed`")
    text = text.replace("- Status: `running`", "- Status: `completed`")
    lines = [line for line in text.splitlines() if not line.startswith("- Error: `")]
    for index, line in enumerate(lines):
        if line.startswith("- Updated UTC:"):
            lines[index] = f"- Updated UTC: `{datetime.now(timezone.utc).isoformat()}`"
            break
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    configure_tracking_environment()
    report = require_file(args.report)
    history = require_file(args.history)
    best = require_file(args.best_checkpoint)
    final = require_file(args.final_checkpoint)

    client = MlflowClient()
    before = client.get_run(args.run_id).info.status
    report.write_text(completed_report(report.read_text(encoding="utf-8")), encoding="utf-8")

    artifact_paths = config_section("mlflow", "artifact_paths")
    client.log_artifact(args.run_id, str(best), artifact_paths["models"])
    client.log_artifact(args.run_id, str(final), artifact_paths["models"])
    client.log_artifact(args.run_id, str(report), artifact_paths["reports"])
    client.log_artifact(args.run_id, str(history), artifact_paths["reports"])
    client.set_tag(args.run_id, "recovered_from_console_encoding_error", "true")
    client.set_terminated(args.run_id, status="FINISHED")
    after = client.get_run(args.run_id).info.status

    ledger = PROJECT_ROOT / "reports" / "mls_experiments" / "experiment_log.md"
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(
            f"| {datetime.now(timezone.utc).isoformat()} | {args.run_name} | completed | "
            f"Recovered MLflow teardown encoding error; existing run {args.run_id} finalized. |\n"
        )
    print(json.dumps({"run_id": args.run_id, "before": before, "after": after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
