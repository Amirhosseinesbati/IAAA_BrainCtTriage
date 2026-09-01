import numpy as np
import pytest

from scripts.calibrate_ich_2p5d_mask_thresholds import (
    _threshold_dice_and_selection,
)


def test_threshold_selection_uses_conservative_fallback_without_observed_pixels() -> None:
    thresholds = np.asarray([0.1, 0.25, 0.5], dtype=np.float32)

    dice, selected, reason = _threshold_dice_and_selection(
        intersections=np.zeros(3),
        predicted=np.asarray([100, 20, 0]),
        observed_pixels=0,
        thresholds=thresholds,
    )

    assert np.isnan(dice).all()
    assert selected == 2
    assert reason == "fallback_no_observed_pixels"


def test_threshold_selection_maximizes_dice_and_breaks_ties_conservatively() -> None:
    thresholds = np.asarray([0.1, 0.25, 0.5], dtype=np.float32)

    dice, selected, reason = _threshold_dice_and_selection(
        intersections=np.asarray([8, 8, 4]),
        predicted=np.asarray([12, 12, 4]),
        observed_pixels=8,
        thresholds=thresholds,
    )

    np.testing.assert_allclose(dice, np.asarray([0.8, 0.8, 2 / 3]))
    assert selected == 1
    assert reason == "max_calibration_pixel_dice"


def test_threshold_selection_rejects_fallback_outside_grid() -> None:
    with pytest.raises(ValueError, match="fallback must occur exactly once"):
        _threshold_dice_and_selection(
            intersections=np.zeros(2),
            predicted=np.zeros(2),
            observed_pixels=0,
            thresholds=np.asarray([0.1, 0.25]),
        )
