from __future__ import annotations

from dataclasses import asdict

from scripts.evaluate_ich_sah_iph_residual_gate import evaluate_exp67_gate
from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    checkpoint_selection_score,
)


def _summary(*, candidate: bool) -> dict:
    subtypes = {
        "EDH": {
            "dice_known_pixels": 0.50,
            "study_auc": 0.90,
            "mae_ml": 1.0,
            "bias_ml": -0.2,
        },
        "IVH": {
            "dice_known_pixels": 0.60,
            "study_auc": 0.96,
            "mae_ml": 0.6,
            "bias_ml": -0.1,
        },
        "SDH": {
            "dice_known_pixels": 0.40,
            "study_auc": 0.85,
            "mae_ml": 7.0,
            "bias_ml": -4.0,
        },
        "IPH": {
            "dice_known_pixels": 0.698 if candidate else 0.700,
            "study_auc": 0.949 if candidate else 0.950,
            "mae_ml": 1.05 if candidate else 1.0,
            "bias_ml": -0.25 if candidate else -0.2,
        },
        "SAH": {
            "dice_known_pixels": 0.07 if candidate else 0.05,
            "study_auc": 0.90,
            "mae_ml": 1.8 if candidate else 2.0,
            "bias_ml": -1.0,
        },
    }
    return {
        "selection_score": 0.605 if candidate else 0.600,
        "mean_foreground_dice": 0.454 if candidate else 0.450,
        "any_ich_study_auc": 0.92,
        "macro_subtype_study_auc": 0.90,
        "normal_false_positive_rate_at_0_1ml": 0.20,
        "presence_f1_at_0_1ml": 0.82,
        "total_volume_mae_ml": 9.8 if candidate else 10.0,
        "total_volume_bias_ml": -4.8 if candidate else -5.0,
        "subtypes": subtypes,
    }


def _inputs() -> tuple[dict, dict, dict, dict]:
    baseline_summary = _summary(candidate=False)
    candidate_summary = _summary(candidate=True)
    baseline_run = {
        "calibration_summary": baseline_summary,
        "best_calibration_checkpoint_score": checkpoint_selection_score(
            baseline_summary, "fpr_volume_penalized"
        ),
        "best_epoch": 9,
        "checkpoint_sha256": "baseline-sha",
        "manifest_sha256": "manifest-sha",
        "outer_evaluation_performed": False,
    }
    candidate_run = {
        "calibration_summary": candidate_summary,
        "best_calibration_checkpoint_score": checkpoint_selection_score(
            candidate_summary, "fpr_volume_penalized"
        ),
        "best_epoch": 2,
        "initial_checkpoint_sha256": "baseline-sha",
        "manifest_sha256": "manifest-sha",
        "outer_evaluation_performed": False,
    }
    baseline_config = asdict(
        ICH25DSegmentationTrainConfig(
            run_name="baseline",
            output_dir="baseline",
            epochs=10,
            evaluate_outer=False,
        )
    )
    candidate_config = dict(baseline_config)
    candidate_config.update(
        {
            "run_name": "candidate",
            "output_dir": "candidate",
            "epochs": 6,
            "learning_rate": 5e-4,
            "classification_loss_weight": 0.0,
            "initial_checkpoint": "baseline/best.pth",
            "freeze_base_model": True,
            "sah_residual_adapter": True,
            "sah_residual_hidden_channels": 16,
            "sah_maximum_logit_residual": 8.0,
            "sah_include_incumbent_iph": True,
            "sah_positive_pixel_loss_weight": 0.03,
        }
    )
    return baseline_run, candidate_run, baseline_config, candidate_config


def test_exp67_gate_accepts_sah_gain_with_bounded_iph_cost() -> None:
    result = evaluate_exp67_gate(*_inputs())

    assert result["all_gates_passed"]
    assert result["decision"] == "advance_locked_recipe_to_patient_disjoint_oof"


def test_exp67_gate_rejects_excessive_iph_dice_loss() -> None:
    baseline_run, candidate_run, baseline_config, candidate_config = _inputs()
    candidate_run["calibration_summary"]["subtypes"]["IPH"][
        "dice_known_pixels"
    ] = 0.69

    result = evaluate_exp67_gate(
        baseline_run,
        candidate_run,
        baseline_config,
        candidate_config,
    )

    assert not result["all_gates_passed"]
    assert not result["quality_checks"]["iph_dice_loss_at_most_0_005"]["passed"]


def test_exp67_gate_rejects_missing_iph_support_recipe() -> None:
    baseline_run, candidate_run, baseline_config, candidate_config = _inputs()
    candidate_config["sah_include_incumbent_iph"] = False

    result = evaluate_exp67_gate(
        baseline_run,
        candidate_run,
        baseline_config,
        candidate_config,
    )

    assert not result["all_gates_passed"]
    assert not result["candidate_recipe_checks"]["sah_include_incumbent_iph"]
