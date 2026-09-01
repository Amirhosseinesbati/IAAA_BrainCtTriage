"""Gate the preregistered exp66 positive-pixel SAH recovery screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_ich_sah_residual_gate import (
    ALLOWED_CONFIG_DIFFERENCES,
    _read_json,
    evaluate_sah_residual_gate,
)
from src.strategies.ich_v2.operations import notify_campaign


EXP66_ALLOWED_CONFIG_DIFFERENCES = ALLOWED_CONFIG_DIFFERENCES | {
    "sah_positive_pixel_loss_weight",
}
EXP66_EXPECTED_RECIPE = {
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
    "sah_maximum_logit_residual": 12.0,
    "sah_positive_pixel_loss_weight": 0.03,
    "sah_residual_adapter": True,
    "sah_residual_hidden_channels": 16,
    "sah_tversky_loss_weight": 0.0,
}


def evaluate_exp66_gate(
    baseline_run: dict,
    candidate_run: dict,
    baseline_config: dict,
    candidate_config: dict,
) -> dict:
    return evaluate_sah_residual_gate(
        baseline_run,
        candidate_run,
        baseline_config,
        candidate_config,
        allowed_config_differences=EXP66_ALLOWED_CONFIG_DIFFERENCES,
        expected_candidate_recipe=EXP66_EXPECTED_RECIPE,
        experiment_name="exp66_sah_positive_pixel_recovery",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    result = evaluate_exp66_gate(
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
                "🏆 تصمیم exp66 خونریزی مسابقه IAAA: loss مثبت‌پیکسلی همهٔ گیت‌ها را "
                "پاس کرد و مجوز OOF بیمارمحور گرفت. تحلیل کوتاه: recovery واقعی SAH "
                "بدون خریدن آن با FPR یا افت حجم/زیرنوع‌های دیگر تعمیم یافته است."
                if passed
                else "🧪 تصمیم exp66 خونریزی مسابقه IAAA: کاندید پیش از outer رد شد. "
                "تحلیل کوتاه: fit انتخابی train به بهبود ایمن calibration تبدیل نشده؛ "
                "پس افزایش epoch یا وزن بدون تغییر representation توجیه ندارد."
            ),
            experiment="exp66_sah_positive_pixel_recovery",
            decision=result["decision"],
            failed_gates=", ".join(failed) if failed else "none",
            checkpoint_score_delta=(
                f"{result['computed_candidate_checkpoint_score'] - result['baseline_checkpoint_score']:+.5f}"
            ),
            sah_dice_delta=f"{delta['sah_dice_known_pixels']:+.5f}",
            sah_mae_delta_ml=f"{delta['sah_mae_ml']:+.3f}",
            total_mae_delta_ml=f"{delta['total_volume_mae_ml']:+.3f}",
            normal_fpr_delta=(
                f"{delta['normal_false_positive_rate_at_0_1ml']:+.5f}"
            ),
        )


if __name__ == "__main__":
    main()
