from copy import deepcopy

from scripts.evaluate_ich_diffuse_tversky_gate import (
    evaluate_diffuse_tversky_gate,
)
from src.strategies.ich_2p5d.segmentation_train import checkpoint_selection_score


def _summary() -> dict:
    return {
        "selection_score": 0.670,
        "mean_foreground_dice": 0.470,
        "any_ich_study_auc": 0.925,
        "macro_subtype_study_auc": 0.905,
        "normal_false_positive_rate_at_0_1ml": 0.18,
        "presence_f1_at_0_1ml": 0.89,
        "total_volume_mae_ml": 9.9,
        "total_volume_bias_ml": -5.8,
        "subtypes": {
            "EDH": {"dice_known_pixels": 0.54},
            "IPH": {"dice_known_pixels": 0.68},
            "IVH": {"dice_known_pixels": 0.65},
            "SAH": {"dice_known_pixels": 0.07},
            "SDH": {"dice_known_pixels": 0.39},
        },
    }


def _config() -> dict:
    return {
        "run_name": "baseline",
        "output_dir": "baseline",
        "epochs": 10,
        "learning_rate": 2e-4,
        "patience": 3,
        "initial_checkpoint": None,
        "evaluate_outer": False,
        "max_train_steps": None,
        "physical_volume_loss_weight": 0.0,
        "diffuse_tversky_loss_weight": 0.0,
        "seed": 42,
    }


def _inputs() -> tuple[dict, dict, dict, dict]:
    baseline_summary = _summary()
    baseline = {
        "best_epoch": 9,
        "best_calibration_checkpoint_score": checkpoint_selection_score(
            baseline_summary, "fpr_volume_penalized"
        ),
        "calibration_summary": baseline_summary,
        "checkpoint_sha256": "baseline-sha",
        "initial_checkpoint_sha256": None,
        "manifest_sha256": "manifest-sha",
        "outer_evaluation_performed": False,
    }
    candidate_summary = deepcopy(baseline_summary)
    candidate = {
        "best_epoch": 2,
        "best_calibration_checkpoint_score": checkpoint_selection_score(
            candidate_summary, "fpr_volume_penalized"
        ),
        "calibration_summary": candidate_summary,
        "checkpoint_sha256": "candidate-sha",
        "initial_checkpoint_sha256": "baseline-sha",
        "manifest_sha256": "manifest-sha",
        "outer_evaluation_performed": False,
    }
    baseline_config = _config()
    candidate_config = {
        **baseline_config,
        "run_name": "candidate",
        "output_dir": "candidate",
        "epochs": 4,
        "learning_rate": 2e-5,
        "patience": 2,
        "initial_checkpoint": "baseline/best.pth",
        "diffuse_tversky_loss_weight": 0.03,
    }
    return baseline, candidate, baseline_config, candidate_config


def test_passing_candidate_advances_only_to_five_fold_oof() -> None:
    result = evaluate_diffuse_tversky_gate(*_inputs())
    assert result["all_gates_passed"]
    assert result["decision"] == "advance_to_patient_disjoint_five_fold_oof"


def test_outer_observation_or_epoch_zero_forces_rejection() -> None:
    baseline, candidate, baseline_config, candidate_config = _inputs()
    candidate["outer_evaluation_performed"] = True
    candidate["best_epoch"] = 0
    result = evaluate_diffuse_tversky_gate(
        baseline, candidate, baseline_config, candidate_config
    )
    assert not result["all_gates_passed"]
    assert not result["provenance_checks"]["candidate_outer_not_evaluated"]["passed"]
    assert not result["quality_checks"][
        "trained_candidate_epoch_at_least_one"
    ]["passed"]


def test_volume_failure_cannot_be_hidden_by_better_diffuse_dice() -> None:
    baseline, candidate, baseline_config, candidate_config = _inputs()
    candidate["calibration_summary"]["subtypes"]["SAH"]["dice_known_pixels"] = 0.20
    candidate["calibration_summary"]["subtypes"]["SDH"]["dice_known_pixels"] = 0.50
    candidate["calibration_summary"]["total_volume_mae_ml"] = 11.0
    candidate["best_calibration_checkpoint_score"] = checkpoint_selection_score(
        candidate["calibration_summary"], "fpr_volume_penalized"
    )
    result = evaluate_diffuse_tversky_gate(
        baseline, candidate, baseline_config, candidate_config
    )
    assert not result["all_gates_passed"]
    assert not result["quality_checks"][
        "total_volume_mae_at_most_10_26777ml"
    ]["passed"]


def test_unregistered_recipe_change_forces_rejection() -> None:
    baseline, candidate, baseline_config, candidate_config = _inputs()
    candidate_config["batch_size"] = 8
    baseline_config["batch_size"] = 16
    result = evaluate_diffuse_tversky_gate(
        baseline, candidate, baseline_config, candidate_config
    )
    assert not result["all_gates_passed"]
    assert result["unexpected_config_differences"] == ["batch_size"]
