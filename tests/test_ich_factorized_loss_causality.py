from __future__ import annotations

from scripts.diagnose_ich_factorized_loss_causality import causal_decision
from scripts.diagnose_ich_factorized_calibration_attribution import (
    calibration_attribution_decision,
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
