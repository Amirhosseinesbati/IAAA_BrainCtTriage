"""Evaluate the preregistered MLS Vast reproducibility gate without a model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from src.strategies.config_models import MLSHeatmapConfig


METRIC_GUARDS = {
    "study_mls_mae_mm": 0.35,
    "study_boundary_f1": 0.06,
    "selector_auc": 0.03,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return rows


def _epoch_row(rows: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    matches = [row for row in rows if int(float(row.get("epoch", -1))) == epoch]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one epoch {epoch} row, found {len(matches)}")
    return matches[0]


def _parse_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    status_match = re.search(r"^- Status: `([^`]+)`", text, flags=re.MULTILINE)
    config_match = re.search(r"^- Config: `(\{.*\})`\s*$", text, flags=re.MULTILINE)
    run_match = re.search(r"^- MLflow run id: `([^`]+)`", text, flags=re.MULTILINE)
    if status_match is None or config_match is None:
        raise ValueError(f"Report is missing status or serialized config: {path}")
    return {
        "status": status_match.group(1),
        "config": json.loads(config_match.group(1)),
        "mlflow_run_id": run_match.group(1) if run_match else None,
    }


def _resolved_manifest_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("task") != "mls" or payload.get("strategy") != "mls_heatmap":
        raise ValueError("Expected an MLS heatmap experiment manifest")
    return MLSHeatmapConfig.model_validate(payload["training_config"]).model_dump()


def evaluate_repro_gate(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    report: dict[str, Any],
    expected_config: dict[str, Any],
    *,
    epoch: int = 15,
    guards: dict[str, float] | None = None,
) -> dict[str, Any]:
    guards = dict(guards or METRIC_GUARDS)
    reference = _epoch_row(reference_rows, epoch)
    candidate = _epoch_row(candidate_rows, epoch)
    observed_config = MLSHeatmapConfig.model_validate(report["config"]).model_dump()
    normalized_expected_config = MLSHeatmapConfig.model_validate(expected_config).model_dump()
    checks: dict[str, dict[str, Any]] = {
        "terminal_completed": {
            "passed": report["status"] == "completed",
            "observed": report["status"],
            "expected": "completed",
        },
        "manifest_config_exact": {
            "passed": observed_config == normalized_expected_config,
        },
        "mlflow_run_identified": {
            "passed": bool(report.get("mlflow_run_id")),
            "observed": report.get("mlflow_run_id"),
        },
    }
    for metric, tolerance in guards.items():
        reference_value = float(reference[metric])
        candidate_value = float(candidate[metric])
        finite = math.isfinite(reference_value) and math.isfinite(candidate_value)
        absolute_delta = abs(candidate_value - reference_value) if finite else math.inf
        checks[metric] = {
            "passed": finite and absolute_delta <= tolerance,
            "reference": reference_value,
            "candidate": candidate_value,
            "signed_delta": candidate_value - reference_value,
            "absolute_delta": absolute_delta,
            "tolerance": tolerance,
        }
    return {
        "schema_version": 1,
        "gate": "mls_vast_exp14r2_reproducibility",
        "epoch": epoch,
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_repro_gate(
        _load_jsonl(args.reference_metrics),
        _load_jsonl(args.candidate_metrics),
        _parse_report(args.candidate_report),
        _resolved_manifest_config(args.candidate_manifest),
        epoch=args.epoch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
