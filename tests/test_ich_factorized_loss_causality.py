from __future__ import annotations

from scripts.diagnose_ich_factorized_loss_causality import causal_decision
from scripts.diagnose_ich_factorized_calibration_attribution import (
    calibration_attribution_decision,
)
from scripts.diagnose_ich_factorized_residual_heads_calibration import (
    residual_head_gate,
)
from scripts.train_ich_factorized_residual_heads_local import (
    residual_head_screen_gate,
)


def _deltas(full: float, no_dice: float, no_focal: float, foreground: float):
    values = {
        "full_exp80": full,
        "without_conditional_dice": no_dice,
        "without_conditional_focal": no_focal,
        "foreground_only": foreground,
    }
    return {
        variant: {
            "subtypes": {
                "SDH": {"soft_dice": value},
                "SAH": {"soft_dice": value},
            }
        }
        for variant, value in values.items()
    }


def test_causal_decision_identifies_conditional_dice() -> None:
    result = causal_decision(_deltas(-0.02, -0.005, -0.018, 0.001))
    assert result["decision"] == "conditional_dice_primary_suspect"


def test_causal_decision_identifies_interaction() -> None:
    result = causal_decision(_deltas(-0.02, -0.01, -0.011, 0.001))
    assert result["decision"] == "conditional_loss_interaction_suspected"


def test_causal_decision_requires_reproduced_short_horizon_drift() -> None:
    result = causal_decision(_deltas(-0.0005, 0.01, 0.01, 0.0))
    assert result["decision"] == "short_horizon_drift_not_reproduced"


def _calibration_deltas(
    full: float,
    no_dice: float,
    no_focal: float,
    foreground: float,
    *,
    diffuse_offset: float = 0.0,
):
    values = {
        "full_exp80": full,
        "without_conditional_dice": no_dice,
        "without_conditional_focal": no_focal,
        "foreground_only": foreground,
    }
    return {
        variant: {
            "mean_foreground_dice": value,
            "subtypes": {
                "SDH": {"dice_known_pixels": value + diffuse_offset},
                "SAH": {"dice_known_pixels": value + diffuse_offset},
            },
        }
        for variant, value in values.items()
    }


def test_calibration_attribution_identifies_conditional_dice() -> None:
    result = calibration_attribution_decision(
        _calibration_deltas(-0.04, -0.005, -0.035, 0.0)
    )
    assert result["decision"] == "conditional_dice_primary_suspect"


def test_calibration_attribution_identifies_interaction() -> None:
    result = calibration_attribution_decision(
        _calibration_deltas(-0.04, -0.01, -0.015, 0.0)
    )
    assert result["decision"] == "conditional_loss_interaction_primary_suspect"


def _gate_summary(*, dice: float, sdh: float, edh: float, sah: float):
    return {
        "mean_foreground_dice": dice,
        "normal_false_positive_rate_at_0_1ml": 0.1,
        "presence_f1_at_0_1ml": 0.9,
        "total_volume_mae_ml": 10.0,
        "total_volume_bias_ml": -5.0,
        "subtypes": {
            "SDH": {"dice_known_pixels": sdh},
            "EDH": {"dice_known_pixels": edh},
            "SAH": {"dice_known_pixels": sah},
        },
    }


def test_residual_head_gate_requires_every_safety_metric() -> None:
    baseline = _gate_summary(dice=0.45, sdh=0.38, edh=0.53, sah=0.05)
    candidate = _gate_summary(dice=0.446, sdh=0.376, edh=0.526, sah=0.046)
    assert residual_head_gate(baseline, candidate)["all_passed"]
    candidate["subtypes"]["SDH"]["dice_known_pixels"] = 0.37
    assert not residual_head_gate(baseline, candidate)["all_passed"]


def _screen_summary(
    *,
    score: float,
    selection: float,
    dice: float,
    sdh: float,
    sah: float,
    mae: float = 10.0,
    bias: float = -5.0,
):
    return {
        "checkpoint_score": score,
        "selection_score": selection,
        "mean_foreground_dice": dice,
        "any_ich_study_auc": 0.92,
        "macro_subtype_study_auc": 0.91,
        "presence_f1_at_0_1ml": 0.88,
        "normal_false_positive_rate_at_0_1ml": 0.19,
        "total_volume_mae_ml": mae,
        "total_volume_bias_ml": bias,
        "subtypes": {
            "IVH": {"dice_known_pixels": 0.64},
            "IPH": {"dice_known_pixels": 0.67},
            "SDH": {"dice_known_pixels": sdh},
            "EDH": {"dice_known_pixels": 0.53},
            "SAH": {"dice_known_pixels": sah},
        },
    }


def test_residual_head_screen_gate_requires_material_joint_gain() -> None:
    baseline = _screen_summary(
        score=0.58, selection=0.66, dice=0.45, sdh=0.38, sah=0.05
    )
    candidate = _screen_summary(
        score=0.583, selection=0.663, dice=0.455, sdh=0.385, sah=0.06,
        mae=9.9, bias=-4.9,
    )
    gates = residual_head_screen_gate(baseline, candidate)
    assert gates["all_passed"]
    candidate["subtypes"]["SAH"]["dice_known_pixels"] = 0.059
    assert not residual_head_screen_gate(baseline, candidate)["all_passed"]


def test_residual_head_screen_gate_rejects_score_gain_with_worse_volume() -> None:
    baseline = _screen_summary(
        score=0.58, selection=0.66, dice=0.45, sdh=0.38, sah=0.05
    )
    candidate = _screen_summary(
        score=0.583, selection=0.663, dice=0.455, sdh=0.385, sah=0.06,
        mae=10.01, bias=-4.9,
    )
    gates = residual_head_screen_gate(baseline, candidate)
    assert not gates["volume_mae_noninferior"]
    assert not gates["all_passed"]
