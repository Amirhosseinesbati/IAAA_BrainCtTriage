"""Gate the preregistered exp65 background-to-SAH residual screen.

Only aggregate run artifacts are consumed.  The gate requires an epoch greater
than zero, exact preservation of every non-target subtype, a meaningful SAH
gain, and no degradation in false positives or volume before any OOF work is
authorized.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    checkpoint_selection_score,
)
from src.strategies.ich_v2.operations import notify_campaign


NON_TARGET_SUBTYPES = ("EDH", "IPH", "IVH", "SDH")
INVARIANT_METRICS = ("dice_known_pixels", "study_auc", "mae_ml", "bias_ml")
ALLOWED_CONFIG_DIFFERENCES = {
    "classification_loss_weight",
    "epochs",
    "freeze_base_model",
    "initial_checkpoint",
    "learning_rate",
    "output_dir",
    "patience",
    "run_name",
    "sah_maximum_logit_residual",
    "sah_residual_adapter",
    "sah_residual_hidden_channels",
    "sah_tversky_loss_weight",
}
EXPECTED_CANDIDATE_RECIPE = {
    "classification_loss_weight": 0.0,
    "diffuse_tversky_loss_weight": 0.0,
    "epochs": 6,
    "evaluate_outer": False,
    "five_slice_context_adapter": False,
    "freeze_base_model": True,
    "horizontal_symmetry_adapter": False,
    "learning_rate": 5e-4,
    "max_train_steps": None,
    "patience": 2,
    "physical_volume_loss_weight": 0.0,
    "sampler_study_balance_power": 0.0,
    "sah_maximum_logit_residual": 8.0,
    "sah_residual_adapter": True,
    "sah_residual_hidden_channels": 16,
    "sah_tversky_loss_weight": 0.03,
    "sah_positive_pixel_loss_weight": 0.0,
}
THRESHOLDS = {
    "minimum_sah_dice_gain": 0.01,
    "minimum_sah_mae_improvement_ml": 0.10,
    "minimum_checkpoint_score_gain": 0.001,
    "invariant_absolute_tolerance": 1e-10,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing metric: {name}")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"Non-finite metric: {name}")
    return result


def _normalise_config(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = asdict(
        ICH25DSegmentationTrainConfig(run_name="defaults", output_dir="defaults")
    )
    defaults.update(payload)
    return defaults


def _summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    result = {
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
        "normal_false_positive_rate_at_0_1ml": _finite(
            summary["normal_false_positive_rate_at_0_1ml"], "normal_fpr"
        ),
        "presence_f1_at_0_1ml": _finite(
            summary["presence_f1_at_0_1ml"], "presence_f1"
        ),
        "total_volume_mae_ml": _finite(
            summary["total_volume_mae_ml"], "total_volume_mae_ml"
        ),
        "total_volume_bias_ml": _finite(
            summary["total_volume_bias_ml"], "total_volume_bias_ml"
        ),
    }
    for subtype, values in summary["subtypes"].items():
        for metric in INVARIANT_METRICS:
            result[f"{subtype.lower()}_{metric}"] = _finite(
                values[metric], f"{subtype}.{metric}"
            )
    return result


def evaluate_sah_residual_gate(
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
    *,
    allowed_config_differences: set[str] | None = None,
    expected_candidate_recipe: dict[str, Any] | None = None,
    experiment_name: str = "exp65_sah_background_expansion_residual",
) -> dict[str, Any]:
    allowed_config_differences = (
        ALLOWED_CONFIG_DIFFERENCES
        if allowed_config_differences is None
        else allowed_config_differences
    )
    expected_candidate_recipe = (
        EXPECTED_CANDIDATE_RECIPE
        if expected_candidate_recipe is None
        else expected_candidate_recipe
    )
    baseline_config = _normalise_config(baseline_config)
    candidate_config = _normalise_config(candidate_config)
    differing = sorted(
        key
        for key in baseline_config.keys() | candidate_config.keys()
        if baseline_config.get(key) != candidate_config.get(key)
    )
    unexpected = sorted(set(differing) - allowed_config_differences)
    recipe_checks = {
        key: candidate_config.get(key) == expected
        for key, expected in expected_candidate_recipe.items()
    }

    baseline_summary = baseline_run["calibration_summary"]
    candidate_summary = candidate_run["calibration_summary"]
    baseline = _summary_metrics(baseline_summary)
    candidate = _summary_metrics(candidate_summary)
    deltas = {key: candidate[key] - baseline[key] for key in baseline}
    tolerance = THRESHOLDS["invariant_absolute_tolerance"]
    non_target_checks = {
        f"{subtype.lower()}_{metric}_exact": abs(
            deltas[f"{subtype.lower()}_{metric}"]
        )
        <= tolerance
        for subtype in NON_TARGET_SUBTYPES
        for metric in INVARIANT_METRICS
    }
    ranking_checks = {
        "any_auc_exact": abs(deltas["any_ich_study_auc"]) <= tolerance,
        "macro_auc_exact": abs(deltas["macro_subtype_study_auc"]) <= tolerance,
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
        "normal_fpr_not_worse": candidate[
            "normal_false_positive_rate_at_0_1ml"
        ]
        <= baseline["normal_false_positive_rate_at_0_1ml"] + tolerance,
        "presence_f1_not_worse": candidate["presence_f1_at_0_1ml"]
        >= baseline["presence_f1_at_0_1ml"] - tolerance,
        "total_volume_mae_not_worse": candidate["total_volume_mae_ml"]
        <= baseline["total_volume_mae_ml"] + tolerance,
        "absolute_total_volume_bias_not_worse": abs(candidate["total_volume_bias_ml"])
        <= abs(baseline["total_volume_bias_ml"]) + tolerance,
        **non_target_checks,
        **ranking_checks,
    }
    passed = all(provenance_checks.values()) and all(quality_checks.values())
    return {
        "schema_version": 1,
        "experiment": experiment_name,
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
            "advance_to_patient_disjoint_five_fold_oof"
            if passed
            else "reject_before_outer"
        ),
        "protocol_note": (
            "The outer fold was not read. A pass authorizes the locked adapter "
            "recipe for patient-disjoint OOF, not final promotion or leaderboard claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    result = evaluate_sah_residual_gate(
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
                "🏆 تصمیم exp65 مسابقه IAAA: residual ایزولهٔ SAH همهٔ گیت‌ها را "
                "پاس کرد و اجازهٔ OOF بیمارمحور گرفت. تحلیل کوتاه: چهار زیرنوع دیگر "
                "دقیقاً ثابت ماندند و بهبود SAH با هزینهٔ FPR یا حجم خریداری نشده است."
                if passed
                else "🧪 تصمیم exp65 مسابقه IAAA: کاندید پیش از outer رد شد. تحلیل "
                "کوتاه: معماری ایزوله جلوی آسیب جانبی را می‌گیرد، اما بدون افزایش "
                "معنادار SAH و بهبود حجم، صرف پیچیده‌تر شدن مدل ارزش OOF ندارد."
            ),
            experiment="exp65_sah_background_expansion_residual",
            decision=result["decision"],
            failed_gates=", ".join(failed) if failed else "none",
            checkpoint_score_delta=f"{result['computed_candidate_checkpoint_score'] - result['baseline_checkpoint_score']:+.5f}",
            sah_dice_delta=f"{delta['sah_dice_known_pixels']:+.5f}",
            sah_mae_delta_ml=f"{delta['sah_mae_ml']:+.3f}",
            total_mae_delta_ml=f"{delta['total_volume_mae_ml']:+.3f}",
            normal_fpr_delta=f"{delta['normal_false_positive_rate_at_0_1ml']:+.5f}",
        )


if __name__ == "__main__":
    main()
