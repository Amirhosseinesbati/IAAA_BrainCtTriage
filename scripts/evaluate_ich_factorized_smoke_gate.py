"""Evaluate the preregistered aggregate-only factorized ICH smoke gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlflow
import pandas as pd

from src.strategies.ich_v2.operations import configure_remote_mlflow, notify_campaign


METRIC_COLUMNS = {
    "selection_score": "calibration_selection_score",
    "mean_foreground_dice": "calibration_mean_foreground_dice",
    "any_ich_study_auc": "calibration_any_ich_study_auc",
    "macro_subtype_study_auc": "calibration_macro_subtype_study_auc",
    "total_volume_mae_ml": "calibration_total_volume_mae_ml",
    "total_volume_bias_ml": "calibration_total_volume_bias_ml",
    "normal_false_positive_rate_at_0_1ml": "calibration_normal_fpr_at_0_1ml",
    "presence_f1_at_0_1ml": "calibration_presence_f1_at_0_1ml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-summary", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent_summary.read_text(encoding="utf-8"))
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    config = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    history = pd.read_csv(args.history)
    epoch_zero_rows = history.loc[history["epoch"] == 0]
    if len(epoch_zero_rows) != 1:
        raise ValueError("Smoke history must contain exactly one epoch-zero row")
    epoch_zero = epoch_zero_rows.iloc[0]
    differences = {
        metric: abs(float(epoch_zero[column]) - float(incumbent[metric]))
        for metric, column in METRIC_COLUMNS.items()
    }
    numeric_history = history.select_dtypes(include="number").to_numpy(dtype=float)
    checkpoint = Path(str(run_summary["checkpoint"]))
    gates = {
        "epoch_zero_continuous_metrics_within_1e_6": all(
            differences[metric] <= 1e-6
            for metric in (
                "selection_score",
                "mean_foreground_dice",
                "any_ich_study_auc",
                "macro_subtype_study_auc",
                "total_volume_mae_ml",
                "total_volume_bias_ml",
            )
        ),
        "epoch_zero_fpr_and_f1_exact": (
            differences["normal_false_positive_rate_at_0_1ml"] == 0.0
            and differences["presence_f1_at_0_1ml"] == 0.0
        ),
        "all_history_numeric_values_finite": bool(
            all(math.isfinite(float(value)) for value in numeric_history.flat)
        ),
        "partial_epoch_completed": bool((history["epoch"] == 1).any()),
        "outer_evaluation_not_performed": (
            run_summary.get("outer_evaluation_performed") is False
            and run_summary.get("outer_summary") is None
        ),
        "peak_vram_below_20_gib": float(run_summary["peak_vram_gb"]) < 20.0,
        "smoke_scope_locked": (
            run_summary.get("run_kind") == "smoke"
            and int(config.get("max_train_steps", -1)) == 4
            and bool(config.get("factorized_output_head", False))
            and config.get("evaluate_outer") is False
        ),
        "checkpoint_written": checkpoint.is_file(),
    }
    gates["all_passed"] = all(gates.values())
    result = {
        "analysis_kind": "factorized_ich_preregistered_smoke_gate",
        "decision": (
            "authorize_locked_three_epoch_calibration_screen"
            if gates["all_passed"]
            else "reject_before_full_calibration_or_outer"
        ),
        "aggregate_only_no_row_level_medical_predictions": True,
        "run_id": run_summary.get("run_id"),
        "run_name": run_summary.get("run_name"),
        "git_commit": run_summary.get("git_commit"),
        "epoch_zero_absolute_differences": differences,
        "peak_vram_gb": float(run_summary["peak_vram_gb"]),
        "history_rows": len(history),
        "preregistered_gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    configure_remote_mlflow()
    with mlflow.start_run(run_id=str(run_summary["run_id"])):
        mlflow.log_artifact(str(args.output), artifact_path="preregistered_gate")
        mlflow.set_tag("smoke_gate_decision", result["decision"])
    notify_campaign(
        "completion",
        (
            "گیت smoke معماری فاکتورگیری‌شده ارزیابی شد. "
            f"نتیجه: {'عبور' if gates['all_passed'] else 'رد'}؛ "
            f"اختلاف Dice در epoch صفر={differences['mean_foreground_dice']:.3e}، "
            f"اختلاف MAE={differences['total_volume_mae_ml']:.3e} و "
            f"peak VRAM={float(run_summary['peak_vram_gb']):.2f}GiB. "
            "تحلیل کوتاه: رد این گیت اجازهٔ اجرای کامل را نمی‌دهد؛ ابتدا باید "
            "هویت عددی همان مسیر inference اصلاح و دوباره از ابتدا اثبات شود."
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
