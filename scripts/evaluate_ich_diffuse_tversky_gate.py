"""Evaluate the preregistered exp63 diffuse-Tversky calibration gate.

The evaluator consumes aggregate run artifacts only. It never reads or uploads
row-level medical predictions and it refuses to advance a candidate that used
the outer fold, changed an unregistered part of the recipe, or did not improve
the warm-start checkpoint at an epoch greater than zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.strategies.ich_2p5d.segmentation_train import checkpoint_selection_score
from src.strategies.ich_v2.operations import notify_campaign


ALLOWED_CONFIG_DIFFERENCES = {
    "diffuse_tversky_loss_weight",
    "epochs",
    "initial_checkpoint",
    "learning_rate",
    "output_dir",
    "patience",
    "run_name",
}

EXPECTED_CANDIDATE_RECIPE = {
    "diffuse_tversky_loss_weight": 0.03,
    "epochs": 4,
    "learning_rate": 2e-5,
    "patience": 2,
    "physical_volume_loss_weight": 0.0,
    "evaluate_outer": False,
    "max_train_steps": None,
}

THRESHOLDS = {
    "minimum_checkpoint_score": 0.58767,
    "minimum_diffuse_mean_dice": 0.22234,
    "minimum_sah_dice": 0.04302,
    "minimum_sdh_dice": 0.37166,
    "minimum_edh_dice": 0.52798,
    "minimum_iph_dice": 0.66484,
    "minimum_ivh_dice": 0.63802,
    "minimum_selection_score": 0.66495,
    "minimum_mean_foreground_dice": 0.45808,
    "maximum_normal_fpr": 0.19444 + 1e-5,
    "minimum_presence_f1": 0.87235,
    "minimum_any_auc": 0.91025,
    "minimum_macro_auc": 0.88789,
    "maximum_total_volume_mae_ml": 10.26777,
    "maximum_absolute_total_volume_bias_ml": 6.06151,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _normalise_config(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.setdefault("diffuse_tversky_loss_weight", 0.0)
    result.setdefault("physical_volume_loss_weight", 0.0)
    result.setdefault("evaluate_outer", True)
    result.setdefault("max_train_steps", None)
    return result


def _finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing metric: {name}")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise ValueError(f"Non-finite metric: {name}")
    return result


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    subtypes = summary["subtypes"]
    subtype_dice = {
        label: _finite(
            subtypes[label]["dice_known_pixels"],
            f"{label}.dice_known_pixels",
        )
        for label in ("EDH", "IPH", "IVH", "SAH", "SDH")
    }
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
        "normal_false_positive_rate_at_0_1ml": _finite(
            summary["normal_false_positive_rate_at_0_1ml"],
            "normal_false_positive_rate_at_0_1ml",
        ),
        "presence_f1_at_0_1ml": _finite(
            summary["presence_f1_at_0_1ml"], "presence_f1_at_0_1ml"
        ),
        "total_volume_mae_ml": _finite(
            summary["total_volume_mae_ml"], "total_volume_mae_ml"
        ),
        "total_volume_bias_ml": _finite(
            summary["total_volume_bias_ml"], "total_volume_bias_ml"
        ),
        **{f"{label.lower()}_dice": value for label, value in subtype_dice.items()},
        "diffuse_mean_dice": (subtype_dice["SAH"] + subtype_dice["SDH"]) / 2.0,
    }


def evaluate_diffuse_tversky_gate(
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
    unexpected = sorted(set(differing) - ALLOWED_CONFIG_DIFFERENCES)

    baseline_summary = baseline_run["calibration_summary"]
    candidate_summary = candidate_run["calibration_summary"]
    baseline_metrics = _metrics(baseline_summary)
    candidate_metrics = _metrics(candidate_summary)
    deltas = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in baseline_metrics
    }
    computed_checkpoint_score = checkpoint_selection_score(
        candidate_summary, "fpr_volume_penalized"
    )
    reported_checkpoint_score = _finite(
        candidate_run["best_calibration_checkpoint_score"],
        "best_calibration_checkpoint_score",
    )

    recipe_checks = {
        key: candidate_config.get(key) == expected
        for key, expected in EXPECTED_CANDIDATE_RECIPE.items()
    }
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
        "checkpoint_score_recomputed_exactly": abs(
            computed_checkpoint_score - reported_checkpoint_score
        )
        <= 1e-9,
        "no_unregistered_config_differences": not unexpected,
        "candidate_recipe_exact": all(recipe_checks.values()),
    }

    quality_checks = {
        "trained_candidate_epoch_at_least_one": int(candidate_run["best_epoch"]) >= 1,
        "checkpoint_score_at_least_0_58767": computed_checkpoint_score
        >= THRESHOLDS["minimum_checkpoint_score"],
        "diffuse_mean_dice_at_least_0_22234": candidate_metrics[
            "diffuse_mean_dice"
        ]
        >= THRESHOLDS["minimum_diffuse_mean_dice"],
        "sah_dice_at_least_0_04302": candidate_metrics["sah_dice"]
        >= THRESHOLDS["minimum_sah_dice"],
        "sdh_dice_at_least_0_37166": candidate_metrics["sdh_dice"]
        >= THRESHOLDS["minimum_sdh_dice"],
        "edh_dice_at_least_0_52798": candidate_metrics["edh_dice"]
        >= THRESHOLDS["minimum_edh_dice"],
        "iph_dice_at_least_0_66484": candidate_metrics["iph_dice"]
        >= THRESHOLDS["minimum_iph_dice"],
        "ivh_dice_at_least_0_63802": candidate_metrics["ivh_dice"]
        >= THRESHOLDS["minimum_ivh_dice"],
        "selection_or_dice_global_gate": (
            candidate_metrics["selection_score"]
            >= THRESHOLDS["minimum_selection_score"]
            or candidate_metrics["mean_foreground_dice"]
            >= THRESHOLDS["minimum_mean_foreground_dice"]
        ),
        "normal_fpr_at_most_0_19444": candidate_metrics[
            "normal_false_positive_rate_at_0_1ml"
        ]
        <= THRESHOLDS["maximum_normal_fpr"],
        "presence_f1_at_least_0_87235": candidate_metrics[
            "presence_f1_at_0_1ml"
        ]
        >= THRESHOLDS["minimum_presence_f1"],
        "any_auc_at_least_0_91025": candidate_metrics["any_ich_study_auc"]
        >= THRESHOLDS["minimum_any_auc"],
        "macro_auc_at_least_0_88789": candidate_metrics[
            "macro_subtype_study_auc"
        ]
        >= THRESHOLDS["minimum_macro_auc"],
        "total_volume_mae_at_most_10_26777ml": candidate_metrics[
            "total_volume_mae_ml"
        ]
        <= THRESHOLDS["maximum_total_volume_mae_ml"],
        "absolute_total_volume_bias_at_most_6_06151ml": abs(
            candidate_metrics["total_volume_bias_ml"]
        )
        <= THRESHOLDS["maximum_absolute_total_volume_bias_ml"],
    }
    all_passed = all(provenance_checks.values()) and all(quality_checks.values())
    return {
        "schema_version": 1,
        "experiment": "exp63_diffuse_tversky_positive_only",
        "evaluation_scope": "calibration_only_no_outer",
        "thresholds": THRESHOLDS,
        "config_differences": {
            key: {
                "baseline": baseline_config.get(key),
                "candidate": candidate_config.get(key),
            }
            for key in differing
        },
        "unexpected_config_differences": unexpected,
        "candidate_recipe_checks": recipe_checks,
        "provenance_checks": {
            name: {"passed": bool(passed)}
            for name, passed in provenance_checks.items()
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline": deltas,
        "computed_candidate_checkpoint_score": computed_checkpoint_score,
        "reported_candidate_checkpoint_score": reported_checkpoint_score,
        "quality_checks": {
            name: {"passed": bool(passed)} for name, passed in quality_checks.items()
        },
        "all_gates_passed": bool(all_passed),
        "decision": (
            "advance_to_patient_disjoint_five_fold_oof"
            if all_passed
            else "reject_before_outer"
        ),
        "protocol_note": (
            "Outer fold was not read. Passing authorizes the locked recipe for "
            "patient-disjoint five-fold OOF; it is not final promotion or "
            "leaderboard evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    baseline_run = _read_json(args.baseline_dir / "run_summary.json")
    candidate_run = _read_json(args.candidate_dir / "run_summary.json")
    baseline_config = _read_json(args.baseline_dir / "resolved_config.json")
    candidate_config = _read_json(args.candidate_dir / "resolved_config.json")
    result = evaluate_diffuse_tversky_gate(
        baseline_run, candidate_run, baseline_config, candidate_config
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.notify:
        metrics = result["candidate_metrics"]
        passed = bool(result["all_gates_passed"])
        failed = [
            name
            for group in ("provenance_checks", "quality_checks")
            for name, item in result[group].items()
            if not item["passed"]
        ]
        notify_campaign(
            "success" if passed else "warning",
            (
                "تصمیم ماشینی exp63 ثبت شد: همهٔ گیت‌ها پاس شدند و recipe قفل‌شده "
                "می‌تواند وارد OOF پنج‌fold بیمارمحور شود. تحلیل کوتاه: این هنوز "
                "promotion نهایی یا نتیجهٔ لیدربرد نیست و outer برای تنظیم استفاده نشده است."
                if passed
                else "تصمیم ماشینی exp63 ثبت شد: کاندید پیش از outer رد شد. تحلیل "
                "کوتاه: بهبود یک معیار برای جبران شکست provenance، حجم، FPR یا حفظ "
                "زیرنوع‌ها کافی نیست؛ مسیر بعدی باید از الگوی گیت‌های شکست‌خورده استخراج شود."
            ),
            experiment="exp63_diffuse_tversky_positive_only",
            decision=result["decision"],
            failed_gates=", ".join(failed) if failed else "none",
            checkpoint_score=f"{result['computed_candidate_checkpoint_score']:.5f}",
            diffuse_mean_dice=f"{metrics['diffuse_mean_dice']:.5f}",
            normal_fpr=f"{metrics['normal_false_positive_rate_at_0_1ml']:.5f}",
            presence_f1=f"{metrics['presence_f1_at_0_1ml']:.5f}",
            volume_mae_ml=f"{metrics['total_volume_mae_ml']:.3f}",
            volume_bias_ml=f"{metrics['total_volume_bias_ml']:.3f}",
        )


if __name__ == "__main__":
    main()
