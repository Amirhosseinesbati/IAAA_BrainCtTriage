"""Gate the preregistered exp67 background/IPH-to-SAH calibration screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_ich_sah_residual_gate import (
    ALLOWED_CONFIG_DIFFERENCES,
    INVARIANT_METRICS,
    _finite,
    _normalise_config,
    _read_json,
    _summary_metrics,
)
from src.strategies.ich_2p5d.segmentation_train import checkpoint_selection_score
from src.strategies.ich_v2.operations import notify_campaign


INVARIANT_SUBTYPES = ("EDH", "IVH", "SDH")
ALLOWED_DIFFERENCES = ALLOWED_CONFIG_DIFFERENCES | {
    "sah_include_incumbent_iph",
    "sah_positive_pixel_loss_weight",
}
EXPECTED_RECIPE = {
    "classification_loss_weight": 0.0,
    "diffuse_tversky_loss_weight": 0.0,
    "epochs": 6,
    "evaluate_outer": False,
    "five_slice_context_adapter": False,
    "freeze_base_model": True,
    "horizontal_symmetry_adapter": False,
    "learning_rate": 5e-4,
    "max_train_steps": None,
    "patience": 3,
    "physical_volume_loss_weight": 0.0,
    "sampler_study_balance_power": 0.0,
    "sah_include_incumbent_iph": True,
    "sah_maximum_logit_residual": 8.0,
    "sah_positive_pixel_loss_weight": 0.03,
    "sah_residual_adapter": True,
    "sah_residual_hidden_channels": 16,
    "sah_tversky_loss_weight": 0.0,
}
THRESHOLDS = {
    "minimum_sah_dice_gain": 0.01,
    "minimum_sah_mae_improvement_ml": 0.10,
    "minimum_checkpoint_score_gain": 0.001,
    "maximum_iph_dice_loss": 0.005,
    "maximum_iph_mae_worsening_ml": 0.10,
    "maximum_iph_absolute_bias_worsening_ml": 0.10,
    "maximum_iph_auc_loss": 0.005,
    "invariant_absolute_tolerance": 1e-10,
}


def evaluate_exp67_gate(
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    baseline_config = _normalise_config(baseline_config)
    candidate_config = _normalise_config(candidate_config)
    differing = sorted(
        key
        for key in baseline_config.keys() | candidate_config.keys()
        if baseline_config.get(key) != candidate_config.get(key)
    )
    unexpected = sorted(set(differing) - ALLOWED_DIFFERENCES)
    recipe_checks = {
        key: candidate_config.get(key) == expected
        for key, expected in EXPECTED_RECIPE.items()
    }

    baseline_summary = baseline_run["calibration_summary"]
    candidate_summary = candidate_run["calibration_summary"]
    baseline = _summary_metrics(baseline_summary)
    candidate = _summary_metrics(candidate_summary)
    deltas = {key: candidate[key] - baseline[key] for key in baseline}
    tolerance = THRESHOLDS["invariant_absolute_tolerance"]

    invariant_checks = {
        f"{subtype.lower()}_{metric}_exact": abs(
            deltas[f"{subtype.lower()}_{metric}"]
        )
        <= tolerance
        for subtype in INVARIANT_SUBTYPES
        for metric in INVARIANT_METRICS
    }
    computed_score = checkpoint_selection_score(
        candidate_summary, "fpr_volume_penalized"
    )
    baseline_score = checkpoint_selection_score(
        baseline_summary, "fpr_volume_penalized"
    )
    reported_score = _finite(
        candidate_run["best_calibration_checkpoint_score"],
        "best_calibration_checkpoint_score",
    )
    provenance_checks = {
        "baseline_outer_not_evaluated": not bool(
            baseline_run.get("outer_evaluation_performed")
        ),
        "candidate_outer_not_evaluated": not bool(
            candidate_run.get("outer_evaluation_performed")
        ),
        "manifest_sha_matches": baseline_run.get("manifest_sha256")
        == candidate_run.get("manifest_sha256"),
        "warm_start_sha_matches_baseline_checkpoint": candidate_run.get(
            "initial_checkpoint_sha256"
        )
        == baseline_run.get("checkpoint_sha256"),
        "checkpoint_score_recomputed_exactly": abs(computed_score - reported_score)
        <= 1e-9,
        "candidate_recipe_exact": all(recipe_checks.values()),
        "no_unregistered_config_differences": not unexpected,
    }
    quality_checks = {
        "trained_candidate_epoch_at_least_one": int(candidate_run["best_epoch"]) >= 1,
        "sah_dice_gain_at_least_0_01": deltas["sah_dice_known_pixels"]
        >= THRESHOLDS["minimum_sah_dice_gain"],
        "sah_mae_improves_at_least_0_10ml": -deltas["sah_mae_ml"]
        >= THRESHOLDS["minimum_sah_mae_improvement_ml"],
        "checkpoint_score_gain_at_least_0_001": computed_score - baseline_score
        >= THRESHOLDS["minimum_checkpoint_score_gain"],
        "iph_dice_loss_at_most_0_005": deltas["iph_dice_known_pixels"]
        >= -THRESHOLDS["maximum_iph_dice_loss"],
        "iph_mae_worsening_at_most_0_10ml": deltas["iph_mae_ml"]
        <= THRESHOLDS["maximum_iph_mae_worsening_ml"],
        "iph_absolute_bias_worsening_at_most_0_10ml": (
            abs(candidate["iph_bias_ml"]) - abs(baseline["iph_bias_ml"])
        )
        <= THRESHOLDS["maximum_iph_absolute_bias_worsening_ml"],
        "iph_auc_loss_at_most_0_005": deltas["iph_study_auc"]
        >= -THRESHOLDS["maximum_iph_auc_loss"],
        "sah_auc_not_worse": deltas["sah_study_auc"] >= -tolerance,
        "any_auc_not_worse": deltas["any_ich_study_auc"] >= -tolerance,
        "macro_auc_not_worse": deltas["macro_subtype_study_auc"] >= -tolerance,
        "normal_fpr_not_worse": deltas[
            "normal_false_positive_rate_at_0_1ml"
        ]
        <= tolerance,
        "presence_f1_not_worse": deltas["presence_f1_at_0_1ml"] >= -tolerance,
        "total_volume_mae_not_worse": deltas["total_volume_mae_ml"] <= tolerance,
        "absolute_total_volume_bias_not_worse": (
            abs(candidate["total_volume_bias_ml"])
            <= abs(baseline["total_volume_bias_ml"]) + tolerance
        ),
        **invariant_checks,
    }
    passed = all(provenance_checks.values()) and all(quality_checks.values())
    return {
        "schema_version": 1,
        "experiment": "exp67_sah_background_or_iph_selective_residual",
        "evaluation_scope": "calibration_only_no_outer",
        "thresholds": THRESHOLDS,
        "candidate_recipe_checks": recipe_checks,
        "config_differences": {
            key: {
                "baseline": baseline_config.get(key),
                "candidate": candidate_config.get(key),
            }
            for key in differing
        },
        "unexpected_config_differences": unexpected,
        "provenance_checks": {
            key: {"passed": bool(value)} for key, value in provenance_checks.items()
        },
        "quality_checks": {
            key: {"passed": bool(value)} for key, value in quality_checks.items()
        },
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "candidate_minus_baseline": deltas,
        "baseline_checkpoint_score": baseline_score,
        "computed_candidate_checkpoint_score": computed_score,
        "reported_candidate_checkpoint_score": reported_score,
        "all_gates_passed": bool(passed),
        "decision": (
            "advance_locked_recipe_to_patient_disjoint_oof"
            if passed
            else "reject_before_outer_close_frozen_relabel_branch"
        ),
        "protocol_note": (
            "The outer fold was not read. Passing authorizes only the locked recipe "
            "for patient-disjoint OOF; it is not a promotion or leaderboard claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    result = evaluate_exp67_gate(
        _read_json(args.baseline_dir / "run_summary.json"),
        _read_json(args.candidate_dir / "run_summary.json"),
        _read_json(args.baseline_dir / "resolved_config.json"),
        _read_json(args.candidate_dir / "resolved_config.json"),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.notify:
        failed = [
            name
            for group in ("provenance_checks", "quality_checks")
            for name, item in result[group].items()
            if not item["passed"]
        ]
        delta = result["candidate_minus_baseline"]
        passed = bool(result["all_gates_passed"])
        notify_campaign(
            "success" if passed else "warning",
            (
                "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
                "🏆 exp67 همهٔ گیت‌های SAH gain و IPH safety را پاس کرد. تحلیل کوتاه: "
                "این فقط مجوز OOF بیمارمحور recipe قفل‌شده است و هنوز مدل پذیرفته‌شده "
                "یا نتیجهٔ لیدربرد محسوب نمی‌شود."
                if passed
                else "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
                "🧪 exp67 پیش از outer رد شد. تحلیل کوتاه: انتخاب‌پذیری train به بهبود "
                "هم‌زمان SAH و ایمنی IPH روی calibration تعمیم نیافت؛ مسیر frozen "
                "relabel بسته و معماری چندبرچسبی/دو‌مرحله‌ای در اولویت قرار می‌گیرد."
            ),
            experiment="exp67_sah_background_or_iph_selective_residual",
            decision=result["decision"],
            failed_gates=", ".join(failed) if failed else "none",
            checkpoint_score_delta=(
                f"{result['computed_candidate_checkpoint_score'] - result['baseline_checkpoint_score']:+.5f}"
            ),
            sah_dice_delta=f"{delta['sah_dice_known_pixels']:+.5f}",
            sah_mae_delta_ml=f"{delta['sah_mae_ml']:+.3f}",
            iph_dice_delta=f"{delta['iph_dice_known_pixels']:+.5f}",
            iph_mae_delta_ml=f"{delta['iph_mae_ml']:+.3f}",
            normal_fpr_delta=(
                f"{delta['normal_false_positive_rate_at_0_1ml']:+.5f}"
            ),
        )


if __name__ == "__main__":
    main()
