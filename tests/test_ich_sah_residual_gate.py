from __future__ import annotations

from dataclasses import asdict

from scripts.evaluate_ich_sah_residual_gate import evaluate_sah_residual_gate
from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    checkpoint_selection_score,
)


def _summary(*, candidate: bool) -> dict:
    subtypes = {
        label: {
            "dice_known_pixels": 0.40,
            "study_auc": 0.90,
            "mae_ml": 1.0,
            "bias_ml": -0.2,
        }
        for label in ("EDH", "IPH", "IVH", "SDH")
    }
    subtypes["SAH"] = {
        "dice_known_pixels": 0.07 if candidate else 0.05,
        "study_auc": 0.90,
        "mae_ml": 1.8 if candidate else 2.0,
        "bias_ml": -1.0,
    }
    return {
        "selection_score": 0.605 if candidate else 0.600,
        "mean_foreground_dice": 0.404 if candidate else 0.400,
        "any_ich_study_auc": 0.90,
        "macro_subtype_study_auc": 0.90,
        "normal_false_positive_rate_at_0_1ml": 0.20,
        "presence_f1_at_0_1ml": 0.80,
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
            "patience": 2,
            "learning_rate": 5e-4,
            "classification_loss_weight": 0.0,
            "initial_checkpoint": "baseline/best.pth",
            "freeze_base_model": True,
            "sah_residual_adapter": True,
            "sah_residual_hidden_channels": 16,
            "sah_maximum_logit_residual": 8.0,
            "sah_tversky_loss_weight": 0.03,
        }
    )
    return baseline_run, candidate_run, baseline_config, candidate_config


def test_sah_residual_gate_accepts_isolated_meaningful_gain() -> None:
    result = evaluate_sah_residual_gate(*_inputs())
    assert result["all_gates_passed"]
    assert result["decision"] == "advance_to_patient_disjoint_five_fold_oof"


def test_sah_residual_gate_rejects_non_target_change() -> None:
    baseline_run, candidate_run, baseline_config, candidate_config = _inputs()
    candidate_run["calibration_summary"]["subtypes"]["SDH"][
        "dice_known_pixels"
    ] -= 1e-4
    result = evaluate_sah_residual_gate(
        baseline_run, candidate_run, baseline_config, candidate_config
    )
    assert not result["all_gates_passed"]
    assert not result["quality_checks"]["sdh_dice_known_pixels_exact"]["passed"]


def test_sah_residual_gate_rejects_epoch_zero_identity() -> None:
    baseline_run, candidate_run, baseline_config, candidate_config = _inputs()
    candidate_run["best_epoch"] = 0
    result = evaluate_sah_residual_gate(
        baseline_run, candidate_run, baseline_config, candidate_config
    )
    assert not result["all_gates_passed"]
    assert not result["quality_checks"][
        "trained_candidate_epoch_at_least_one"
    ]["passed"]
