"""Evaluate a single-change ICH calibration screen without tuning on outer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_CONFIG_DIFFERENCES = {
    "evaluate_outer",
    "output_dir",
    "run_name",
    "sampler_study_balance_power",
    "ivh_center_loss_weight",
    "ivh_center_square_size",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    nested = payload.get("summary")
    if isinstance(nested, dict):
        return nested
    return payload


def _config(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "resolved_config.json")
    payload.setdefault("sampler_study_balance_power", 0.0)
    payload.setdefault("ivh_center_loss_weight", 0.0)
    payload.setdefault("ivh_center_square_size", 11)
    payload.setdefault("evaluate_outer", True)
    return payload


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing metric: {name}")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"Non-finite metric: {name}")
    return result


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    ivh = summary["subtypes"]["IVH"]
    small = ivh["volume_strata"]["small_le_2ml"]
    return {
        "selection_score": _finite(summary["selection_score"], "selection_score"),
        "mean_foreground_dice": _finite(
            summary["mean_foreground_dice"], "mean_foreground_dice"
        ),
        "any_ich_study_auc": _finite(
            summary["any_ich_study_auc"], "any_ich_study_auc"
        ),
        "macro_subtype_study_auc": _finite(
            summary["macro_subtype_study_auc"], "macro_subtype_study_auc"
        ),
        "presence_f1_at_0_1ml": _finite(
            summary["presence_f1_at_0_1ml"], "presence_f1_at_0_1ml"
        ),
        "normal_false_positive_rate_at_0_1ml": _finite(
            summary["normal_false_positive_rate_at_0_1ml"],
            "normal_false_positive_rate_at_0_1ml",
        ),
        "total_volume_mae_ml": _finite(
            summary["total_volume_mae_ml"], "total_volume_mae_ml"
        ),
        "ivh_dice_known_pixels": _finite(
            ivh["dice_known_pixels"], "ivh_dice_known_pixels"
        ),
        "ivh_study_auc": _finite(ivh["study_auc"], "ivh_study_auc"),
        "ivh_mae_ml": _finite(ivh["mae_ml"], "ivh_mae_ml"),
        "small_ivh_positive_studies": _finite(
            small["positive_studies"], "small_ivh_positive_studies"
        ),
        "small_ivh_dice_known_pixels": _finite(
            small["dice_known_pixels"], "small_ivh_dice_known_pixels"
        ),
        "small_ivh_presence_sensitivity_at_0_1ml": _finite(
            small["presence_sensitivity_at_0_1ml"],
            "small_ivh_presence_sensitivity_at_0_1ml",
        ),
        "small_ivh_mae_ml": _finite(small["mae_ml"], "small_ivh_mae_ml"),
    }


def _compare_scope(
    baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]
) -> dict[str, Any]:
    baseline = _metrics(baseline_summary)
    candidate = _metrics(candidate_summary)
    delta = {name: candidate[name] - baseline[name] for name in baseline}
    small_count_equal = delta["small_ivh_positive_studies"] == 0.0
    small_signal = small_count_equal and (
        (
            delta["small_ivh_dice_known_pixels"] > 0.0
            and delta["small_ivh_presence_sensitivity_at_0_1ml"] >= -0.10
        )
        or (
            delta["small_ivh_presence_sensitivity_at_0_1ml"] > 0.0
            and delta["small_ivh_dice_known_pixels"] >= -0.05
        )
    ) and delta["small_ivh_mae_ml"] <= 0.0
    gates = {
        "selection_drop_no_more_than_0_005": delta["selection_score"] >= -0.005,
        "normal_fpr_not_worse": delta[
            "normal_false_positive_rate_at_0_1ml"
        ] <= 0.0,
        "dice_drop_no_more_than_0_01": delta["mean_foreground_dice"] >= -0.01,
        "any_auc_drop_no_more_than_0_01": delta["any_ich_study_auc"] >= -0.01,
        "macro_auc_drop_no_more_than_0_01": delta[
            "macro_subtype_study_auc"
        ] >= -0.01,
        "presence_f1_drop_no_more_than_0_01": delta[
            "presence_f1_at_0_1ml"
        ] >= -0.01,
        "total_volume_mae_not_worse": delta["total_volume_mae_ml"] <= 0.0,
        "ivh_dice_drop_no_more_than_0_01": delta[
            "ivh_dice_known_pixels"
        ] >= -0.01,
        "ivh_auc_drop_no_more_than_0_01": delta["ivh_study_auc"] >= -0.01,
        "ivh_mae_not_worse": delta["ivh_mae_ml"] <= 0.0,
        "small_ivh_signal_improves": small_signal,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": delta,
        "gates": {
            name: {"passed": bool(passed)} for name, passed in gates.items()
        },
        "all_gates_passed": all(gates.values()),
    }


def compare_sampler_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    baseline_calibration_summary: Path,
    candidate_calibration_summary: Path,
    baseline_outer_summary: Path | None = None,
    candidate_outer_summary: Path | None = None,
) -> dict[str, Any]:
    baseline_config = _config(baseline_dir)
    candidate_config = _config(candidate_dir)
    differing = sorted(
        key
        for key in baseline_config.keys() | candidate_config.keys()
        if baseline_config.get(key) != candidate_config.get(key)
    )
    unexpected = sorted(set(differing) - ALLOWED_CONFIG_DIFFERENCES)
    if unexpected:
        raise ValueError(
            "ICH screen must be a single-method comparison; unexpected config "
            f"differences: {unexpected}"
        )
    baseline_run = _read_json(baseline_dir / "run_summary.json")
    candidate_run = _read_json(candidate_dir / "run_summary.json")
    if baseline_run.get("manifest_sha256") != candidate_run.get("manifest_sha256"):
        raise ValueError("Baseline and candidate use different manifests")
    calibration = _compare_scope(
        _summary(baseline_calibration_summary),
        _summary(candidate_calibration_summary),
    )
    if (baseline_outer_summary is None) != (candidate_outer_summary is None):
        raise ValueError("Provide both outer summaries or neither")
    outer = None
    if baseline_outer_summary is not None and candidate_outer_summary is not None:
        outer = _compare_scope(
            _summary(baseline_outer_summary),
            _summary(candidate_outer_summary),
        )
    calibration_passed = bool(calibration["all_gates_passed"])
    outer_passed = None if outer is None else bool(outer["all_gates_passed"])
    if not calibration_passed:
        decision = "reject_before_outer"
    elif outer is None:
        decision = "advance_to_five_fold_oof_without_single_outer_tuning"
    elif outer_passed:
        decision = "advance_to_five_fold_oof"
    else:
        decision = "reject_after_exploratory_outer"
    protocol_note = (
        "Outer result was observed despite a failed calibration gate; treat it as "
        "exploratory and never as an independent confirmation."
        if outer is not None and not calibration_passed
        else "Outer was not used for calibration-only screening."
        if outer is None
        else "Outer comparison is reported but repeated development use limits its "
        "independence; final evidence requires five-fold OOF and leaderboard testing."
    )
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
        "calibration": calibration,
        "outer": outer,
        "decision": decision,
        "protocol_note": protocol_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-calibration-summary", required=True, type=Path)
    parser.add_argument("--candidate-calibration-summary", required=True, type=Path)
    parser.add_argument("--baseline-outer-summary", type=Path)
    parser.add_argument("--candidate-outer-summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_sampler_runs(
        args.baseline_dir,
        args.candidate_dir,
        baseline_calibration_summary=args.baseline_calibration_summary,
        candidate_calibration_summary=args.candidate_calibration_summary,
        baseline_outer_summary=args.baseline_outer_summary,
        candidate_outer_summary=args.candidate_outer_summary,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
