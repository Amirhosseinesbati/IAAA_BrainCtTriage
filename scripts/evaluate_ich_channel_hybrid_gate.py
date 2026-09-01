"""Gate a channel-wise ICH hybrid against its calibration reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS


GLOBAL_DIRECTIONS = {
    "selection_score": 1,
    "mean_foreground_dice": 1,
    "any_ich_study_auc": 1,
    "macro_subtype_study_auc": 1,
    "presence_f1_at_0_1ml": 1,
    "normal_false_positive_rate_at_0_1ml": -1,
    "total_volume_mae_ml": -1,
}
SUBTYPE_DIRECTIONS = {
    "dice_known_pixels": 1,
    "study_auc": 1,
    "mae_ml": -1,
}
STRATUM_FIELDS = (
    "positive_studies",
    "dice_known_pixels",
    "presence_sensitivity_at_0_1ml",
    "mae_ml",
)


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


def _same_optional_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def compare_channel_hybrid(
    baseline_payload: dict[str, Any],
    hybrid_payload: dict[str, Any],
    *,
    minimum_selection_gain: float = 0.005,
) -> dict[str, Any]:
    baseline = _unwrap(baseline_payload)
    hybrid = _unwrap(hybrid_payload)
    evaluation_split = str(
        hybrid_payload.get("evaluation_split", "calibration_only_no_outer")
    )
    if evaluation_split not in {"calibration_only_no_outer", "outer_fold"}:
        raise ValueError(f"Unsupported hybrid evaluation split: {evaluation_split}")
    reference_labels = tuple(hybrid_payload.get("reference_labels", ()))
    candidate_labels = tuple(hybrid_payload.get("candidate_labels", ()))
    if set(reference_labels) | set(candidate_labels) != set(OUTPUT_LABELS[1:]):
        raise ValueError("Hybrid channel partition must cover every ICH subtype once")
    if set(reference_labels) & set(candidate_labels):
        raise ValueError("Reference and candidate channel partitions overlap")

    global_deltas: dict[str, float] = {}
    global_gates: dict[str, dict[str, bool]] = {}
    for metric, direction in GLOBAL_DIRECTIONS.items():
        delta = _finite(hybrid[metric], metric) - _finite(baseline[metric], metric)
        global_deltas[metric] = delta
        global_gates[f"{metric}_not_worse"] = {
            "passed": bool(direction * delta >= -1e-12)
        }
    global_gates["selection_gain_is_material"] = {
        "passed": global_deltas["selection_score"] >= minimum_selection_gain
    }

    subtype_deltas: dict[str, dict[str, float | None]] = {}
    subtype_gates: dict[str, dict[str, dict[str, bool]]] = {}
    for label in OUTPUT_LABELS[1:]:
        subtype_deltas[label] = {}
        subtype_gates[label] = {}
        for metric, direction in SUBTYPE_DIRECTIONS.items():
            name = f"{label}.{metric}"
            baseline_value = baseline["subtypes"][label][metric]
            hybrid_value = hybrid["subtypes"][label][metric]
            if baseline_value is None or hybrid_value is None:
                both_missing = baseline_value is None and hybrid_value is None
                delta = None
                passed = both_missing and label in reference_labels
                subtype_deltas[label][metric] = delta
                subtype_gates[label][f"{metric}_not_worse"] = {
                    "passed": passed,
                    "unsupported_in_both": both_missing,
                }
                continue
            delta = _finite(hybrid_value, name) - _finite(baseline_value, name)
            subtype_deltas[label][metric] = delta
            subtype_gates[label][f"{metric}_not_worse"] = {
                "passed": bool(direction * delta >= -1e-12)
            }

    reference_preservation: dict[str, dict[str, bool]] = {}
    for label in reference_labels:
        checks = {
            metric: _same_optional_number(
                baseline["subtypes"][label][metric],
                hybrid["subtypes"][label][metric],
            )
            for metric in SUBTYPE_DIRECTIONS
        }
        for stratum, baseline_values in baseline["subtypes"][label][
            "volume_strata"
        ].items():
            hybrid_values = hybrid["subtypes"][label]["volume_strata"][stratum]
            for field in STRATUM_FIELDS:
                checks[f"{stratum}.{field}"] = _same_optional_number(
                    baseline_values[field], hybrid_values[field]
                )
        reference_preservation[label] = checks

    all_global = all(item["passed"] for item in global_gates.values())
    all_subtypes = all(
        item["passed"]
        for label_gates in subtype_gates.values()
        for item in label_gates.values()
    )
    all_reference_exact = all(
        passed
        for label_checks in reference_preservation.values()
        for passed in label_checks.values()
    )
    all_passed = all_global and all_subtypes and all_reference_exact
    return {
        "schema_version": 1,
        "evaluation_split": evaluation_split,
        "reference_labels": list(reference_labels),
        "candidate_labels": list(candidate_labels),
        "minimum_selection_gain": minimum_selection_gain,
        "global_deltas": global_deltas,
        "global_gates": global_gates,
        "subtype_deltas": subtype_deltas,
        "subtype_gates": subtype_gates,
        "reference_channel_exact_preservation": reference_preservation,
        "all_gates_passed": all_passed,
        "decision": (
            (
                "outer_fold_supports_locked_channel_strategy"
                if all_passed
                else "outer_fold_does_not_support_locked_channel_strategy"
            )
            if evaluation_split == "outer_fold"
            else (
                "advance_to_cross_fitted_five_fold_oof"
                if all_passed
                else "reject_before_outer"
            )
        ),
        "protocol_note": (
            "This outer result must not change the locked channel mapping; final "
            "evidence requires the complete patient-disjoint five-fold OOF."
            if evaluation_split == "outer_fold"
            else "Outer was not read. Channel choice must be repeated using each "
            "fold's calibration partition before that fold's outer inference."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--hybrid-summary", required=True, type=Path)
    parser.add_argument("--minimum-selection-gain", type=float, default=0.005)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    hybrid_payload = json.loads(args.hybrid_summary.read_text(encoding="utf-8"))
    result = compare_channel_hybrid(
        baseline_payload,
        hybrid_payload,
        minimum_selection_gain=args.minimum_selection_gain,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
