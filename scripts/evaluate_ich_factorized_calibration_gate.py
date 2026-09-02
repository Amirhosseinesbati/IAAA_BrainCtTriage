"""Evaluate the preregistered Exp80 factorized ICH calibration gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlflow

from src.strategies.ich_v2.operations import configure_remote_mlflow, notify_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-summary", required=True, type=Path)
    parser.add_argument("--incumbent-run-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--candidate-run-summary", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent_summary.read_text(encoding="utf-8"))
    incumbent_run = json.loads(
        args.incumbent_run_summary.read_text(encoding="utf-8")
    )
    candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    candidate_run = json.loads(
        args.candidate_run_summary.read_text(encoding="utf-8")
    )
    config = json.loads(args.resolved_config.read_text(encoding="utf-8"))

    def subtype_dice(summary: dict[str, object], name: str) -> float:
        return float(summary["subtypes"][name]["dice_known_pixels"])

    checkpoint_score = float(candidate_run["best_calibration_checkpoint_score"])
    metrics = {
        "checkpoint_score": checkpoint_score,
        "selection_score": float(candidate["selection_score"]),
        "mean_foreground_dice": float(candidate["mean_foreground_dice"]),
        "normal_fpr": float(candidate["normal_false_positive_rate_at_0_1ml"]),
        "presence_f1": float(candidate["presence_f1_at_0_1ml"]),
        "total_volume_mae_ml": float(candidate["total_volume_mae_ml"]),
        "total_volume_bias_ml": float(candidate["total_volume_bias_ml"]),
        "any_ich_auc": float(candidate["any_ich_study_auc"]),
        "macro_subtype_auc": float(candidate["macro_subtype_study_auc"]),
        **{
            f"{name.lower()}_dice": subtype_dice(candidate, name)
            for name in ("IVH", "IPH", "SDH", "EDH", "SAH")
        },
    }
    deltas = {
        "checkpoint_score": checkpoint_score
        - float(incumbent_run["best_calibration_checkpoint_score"]),
        "selection_score": metrics["selection_score"]
        - float(incumbent["selection_score"]),
        "mean_foreground_dice": metrics["mean_foreground_dice"]
        - float(incumbent["mean_foreground_dice"]),
        "normal_fpr": metrics["normal_fpr"]
        - float(incumbent["normal_false_positive_rate_at_0_1ml"]),
        "presence_f1": metrics["presence_f1"]
        - float(incumbent["presence_f1_at_0_1ml"]),
        "total_volume_mae_ml": metrics["total_volume_mae_ml"]
        - float(incumbent["total_volume_mae_ml"]),
        "total_volume_bias_ml": metrics["total_volume_bias_ml"]
        - float(incumbent["total_volume_bias_ml"]),
        "any_ich_auc": metrics["any_ich_auc"]
        - float(incumbent["any_ich_study_auc"]),
        "macro_subtype_auc": metrics["macro_subtype_auc"]
        - float(incumbent["macro_subtype_study_auc"]),
        **{
            f"{name.lower()}_dice": metrics[f"{name.lower()}_dice"]
            - subtype_dice(incumbent, name)
            for name in ("IVH", "IPH", "SDH", "EDH", "SAH")
        },
    }
    checkpoint = Path(str(candidate_run["checkpoint"]))
    gates = {
        "checkpoint_score_gain_at_least_0_003": checkpoint_score
        >= 0.5896680239,
        "selection_gain_at_least_0_003": metrics["selection_score"]
        >= 0.6691624032,
        "dice_gain_at_least_0_005": metrics["mean_foreground_dice"]
        >= 0.4641058994,
        "normal_fpr_noninferior": metrics["normal_fpr"] <= 0.1944444444,
        "presence_f1_noninferior": metrics["presence_f1"] >= 0.8823529412,
        "volume_mae_noninferior": metrics["total_volume_mae_ml"]
        <= 10.7627157621,
        "absolute_volume_bias_noninferior": abs(metrics["total_volume_bias_ml"])
        <= 6.2363560641,
        "any_auc_noninferior": metrics["any_ich_auc"] >= 0.9233860968,
        "macro_subtype_auc_noninferior": metrics["macro_subtype_auc"]
        >= 0.9109191965,
        "sah_dice_gain_at_least_0_01": metrics["sah_dice"] >= 0.0630242235,
        "sdh_dice_gain_at_least_0_005": metrics["sdh_dice"] >= 0.3866645469,
        "ivh_dice_drop_at_most_0_005": deltas["ivh_dice"] >= -0.005,
        "iph_dice_drop_at_most_0_005": deltas["iph_dice"] >= -0.005,
        "edh_dice_drop_at_most_0_005": deltas["edh_dice"] >= -0.005,
        "all_required_aggregates_finite": all(
            math.isfinite(value) for value in (*metrics.values(), *deltas.values())
        ),
        "locked_config_and_outer_policy": (
            bool(config.get("factorized_output_head", False))
            and config.get("max_train_steps") is None
            and int(config.get("epochs", -1)) == 3
            and math.isclose(float(config.get("learning_rate", -1)), 5e-5)
            and config.get("evaluate_outer") is False
            and candidate_run.get("run_kind") == "calibration_screen"
            and candidate_run.get("outer_evaluation_performed") is False
            and candidate_run.get("outer_summary") is None
        ),
        "checkpoint_written": checkpoint.is_file(),
    }
    gates["all_passed"] = all(gates.values())
    result = {
        "analysis_kind": "factorized_ich_preregistered_calibration_gate",
        "decision": (
            "authorize_same_family_replication"
            if gates["all_passed"]
            else "reject_before_outer_or_oof"
        ),
        "aggregate_only_no_row_level_medical_predictions": True,
        "run_id": candidate_run["run_id"],
        "run_name": candidate_run["run_name"],
        "git_commit": candidate_run["git_commit"],
        "best_epoch": candidate_run["best_epoch"],
        "metrics": metrics,
        "deltas_vs_exp61": deltas,
        "preregistered_gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    configure_remote_mlflow()
    with mlflow.start_run(run_id=str(candidate_run["run_id"])):
        mlflow.log_artifact(str(args.output), artifact_path="preregistered_gate")
        mlflow.set_tag("calibration_gate_decision", result["decision"])
    notify_campaign(
        "completion",
        (
            "گیت نهایی calibration معماری فاکتورگیری‌شده ارزیابی شد. "
            f"نتیجه: {'عبور' if gates['all_passed'] else 'رد'}؛ "
            f"Δscore={deltas['checkpoint_score']:+.4f}، "
            f"ΔDice={deltas['mean_foreground_dice']:+.4f}، "
            f"ΔSAH={deltas['sah_dice']:+.4f}، ΔSDH={deltas['sdh_dice']:+.4f}، "
            f"ΔMAE={deltas['total_volume_mae_ml']:+.3f}ml. "
            "تحلیل کوتاه: فقط عبور هم‌زمان تمام گیت‌های کیفیت، FPR و حجم مجوز "
            "replication می‌دهد؛ در غیر این صورت outer دست‌نخورده می‌ماند."
        ),
        run=result["run_name"],
        decision=result["decision"],
        mlflow=result["run_id"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gates["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
