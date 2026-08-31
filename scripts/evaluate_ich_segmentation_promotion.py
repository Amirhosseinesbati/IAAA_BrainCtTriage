"""Apply pre-registered single-fold promotion gates to an ICH candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_CONFIG_DIFFERENCES = {
    "checkpoint_selection_strategy",
    "empty_foreground_top_fraction",
    "empty_foreground_weight",
    "output_dir",
    "run_name",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _normalized_config(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "resolved_config.json")
    config.setdefault("empty_foreground_weight", 0.0)
    config.setdefault("empty_foreground_top_fraction", 1.0)
    config.setdefault("checkpoint_selection_strategy", "legacy")
    return config


def _metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name)
    if value is None:
        raise ValueError(f"Outer summary lacks a finite {name}")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"Outer summary has a non-finite {name}")
    return result


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_config = _normalized_config(baseline_dir)
    candidate_config = _normalized_config(candidate_dir)
    differing = sorted(
        key
        for key in baseline_config.keys() | candidate_config.keys()
        if baseline_config.get(key) != candidate_config.get(key)
    )
    unexpected = sorted(set(differing) - ALLOWED_CONFIG_DIFFERENCES)
    if unexpected:
        raise ValueError(
            "Candidate is not an approved-method comparison; unexpected config "
            f"differences: {unexpected}"
        )
    baseline_run = _read_json(baseline_dir / "run_summary.json")
    candidate_run = _read_json(candidate_dir / "run_summary.json")
    if baseline_run.get("manifest_sha256") != candidate_run.get("manifest_sha256"):
        raise ValueError("Baseline and candidate use different manifests")
    baseline = _read_json(baseline_dir / "outer_summary.json")
    candidate = _read_json(candidate_dir / "outer_summary.json")

    names = (
        "selection_score",
        "mean_foreground_dice",
        "any_ich_study_auc",
        "macro_subtype_study_auc",
        "presence_f1_at_0_1ml",
        "normal_false_positive_rate_at_0_1ml",
        "total_volume_mae_ml",
    )
    baseline_metrics = {name: _metric(baseline, name) for name in names}
    candidate_metrics = {name: _metric(candidate, name) for name in names}
    deltas = {
        name: candidate_metrics[name] - baseline_metrics[name] for name in names
    }
    gates = {
        "fpr_reduction_at_least_0_05": {
            "passed": deltas["normal_false_positive_rate_at_0_1ml"] <= -0.05,
            "delta": deltas["normal_false_positive_rate_at_0_1ml"],
        },
        "presence_f1_not_lower": {
            "passed": deltas["presence_f1_at_0_1ml"] >= 0.0,
            "delta": deltas["presence_f1_at_0_1ml"],
        },
        "dice_drop_no_more_than_0_01": {
            "passed": deltas["mean_foreground_dice"] >= -0.01,
            "delta": deltas["mean_foreground_dice"],
        },
        "selection_drop_no_more_than_0_005": {
            "passed": deltas["selection_score"] >= -0.005,
            "delta": deltas["selection_score"],
        },
        "any_auc_drop_no_more_than_0_01": {
            "passed": deltas["any_ich_study_auc"] >= -0.01,
            "delta": deltas["any_ich_study_auc"],
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "schema_version": 1,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "manifest_sha256": baseline_run["manifest_sha256"],
        "config_differences": {
            key: {
                "baseline": baseline_config.get(key),
                "candidate": candidate_config.get(key),
            }
            for key in differing
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_baseline": deltas,
        "gates": gates,
        "all_primary_gates_passed": passed,
        "mae_warning": (
            "candidate_mae_worse"
            if deltas["total_volume_mae_ml"] > 0
            else "candidate_mae_not_worse"
        ),
        "decision": (
            "advance_to_independent_fold"
            if passed
            else "reject_or_redesign_before_independent_fold"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = compare_runs(args.baseline_dir, args.candidate_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.require_pass and not result["all_primary_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
