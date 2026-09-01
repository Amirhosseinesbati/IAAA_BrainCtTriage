"""Select ICH model sources from calibration only, then build the hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_ich_channel_hybrid import evaluate_channel_hybrid
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


SUBTYPE_DIRECTIONS = {
    "dice_known_pixels": 1,
    "study_auc": 1,
    "mae_ml": -1,
}


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing metric: {name}")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"Non-finite metric: {name}")
    return result


def select_channel_sources(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Choose candidate only when all core subtype metrics are non-inferior."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    baseline = _unwrap(baseline_payload)
    candidate = _unwrap(candidate_payload)
    decisions: dict[str, dict[str, Any]] = {}
    reference_labels: list[str] = []
    candidate_labels: list[str] = []
    for label in OUTPUT_LABELS[1:]:
        deltas: dict[str, float | None] = {}
        checks: dict[str, bool] = {}
        unavailable_metrics: list[str] = []
        for metric, direction in SUBTYPE_DIRECTIONS.items():
            name = f"{label}.{metric}"
            try:
                delta = _finite(
                    candidate["subtypes"][label][metric], f"candidate.{name}"
                ) - _finite(
                    baseline["subtypes"][label][metric], f"baseline.{name}"
                )
            except ValueError:
                deltas[metric] = None
                checks[metric] = False
                unavailable_metrics.append(metric)
                continue
            deltas[metric] = delta
            checks[metric] = direction * delta >= -tolerance
        strictly_better = any(
            deltas[metric] is not None
            and direction * float(deltas[metric]) > tolerance
            for metric, direction in SUBTYPE_DIRECTIONS.items()
        )
        use_candidate = (
            not unavailable_metrics and all(checks.values()) and strictly_better
        )
        (candidate_labels if use_candidate else reference_labels).append(label)
        decisions[label] = {
            "source": "candidate" if use_candidate else "reference",
            "candidate_minus_reference": deltas,
            "all_core_metrics_non_inferior": all(checks.values()),
            "at_least_one_core_metric_strictly_better": strictly_better,
            "unavailable_metrics": unavailable_metrics,
            "selection_reason": (
                "insufficient_calibration_support"
                if unavailable_metrics
                else "core_metrics_gate"
            ),
        }

    any_delta = _finite(
        candidate["any_ich_study_auc"], "candidate.any_ich_study_auc"
    ) - _finite(baseline["any_ich_study_auc"], "baseline.any_ich_study_auc")
    any_source = "candidate" if any_delta > tolerance else "reference"
    return {
        "schema_version": 1,
        "selection_split": "calibration_only_no_outer",
        "policy": (
            "Per subtype, use candidate iff Dice and AUC are no lower, MAE is no "
            "higher, and at least one is strictly better. Use candidate Any score "
            "iff Any-AUC is strictly higher."
        ),
        "tolerance": tolerance,
        "reference_labels": reference_labels,
        "candidate_labels": candidate_labels,
        "any_ich_source": any_source,
        "any_ich_auc_candidate_minus_reference": any_delta,
        "subtypes": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--baseline-run-summary", required=True, type=Path)
    parser.add_argument("--candidate-run-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    selection = select_channel_sources(
        baseline_payload, candidate_payload, tolerance=args.tolerance
    )
    hybrid = evaluate_channel_hybrid(
        args.baseline_predictions,
        args.candidate_predictions,
        reference_run_summary=args.baseline_run_summary,
        candidate_run_summary=args.candidate_run_summary,
        output_dir=args.output_dir,
        reference_labels=tuple(selection["reference_labels"]),
        any_source=str(selection["any_ich_source"]),
        evaluation_split="calibration_only_no_outer",
    )
    hybrid["channel_selection"] = selection
    selection_path = args.output_dir / "channel_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_path = args.output_dir / "hybrid_summary.json"
    summary_path.write_text(
        json.dumps(hybrid, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(hybrid, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
