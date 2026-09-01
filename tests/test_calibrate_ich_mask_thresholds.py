import unittest

import numpy as np

from scripts.calibrate_ich_2p5d_mask_thresholds import (
    MISSING_CLASS_THRESHOLD,
    _threshold_dice_and_selection,
)


class TestMaskThresholdSelection(unittest.TestCase):
    def test_missing_class_policy_is_fixed_at_half_probability(self) -> None:
        self.assertEqual(MISSING_CLASS_THRESHOLD, 0.5)

    def test_uses_conservative_fallback_without_observed_pixels(self) -> None:
        thresholds = np.asarray([0.1, 0.25, 0.5], dtype=np.float32)

        dice, selected, reason = _threshold_dice_and_selection(
            intersections=np.zeros(3),
            predicted=np.asarray([100, 20, 0]),
            observed_pixels=0,
            thresholds=thresholds,
        )

        self.assertTrue(np.isnan(dice).all())
        self.assertEqual(selected, 2)
        self.assertEqual(reason, "fallback_no_observed_pixels")

    def test_maximizes_dice_and_breaks_ties_conservatively(self) -> None:
        thresholds = np.asarray([0.1, 0.25, 0.5], dtype=np.float32)

        dice, selected, reason = _threshold_dice_and_selection(
            intersections=np.asarray([8, 8, 4]),
            predicted=np.asarray([12, 12, 4]),
            observed_pixels=8,
            thresholds=thresholds,
        )

        np.testing.assert_allclose(dice, np.asarray([0.8, 0.8, 2 / 3]))
        self.assertEqual(selected, 1)
        self.assertEqual(reason, "max_calibration_pixel_dice")

    def test_rejects_fallback_outside_grid(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "fallback must occur exactly once"
        ):
            _threshold_dice_and_selection(
                intersections=np.zeros(2),
                predicted=np.zeros(2),
                observed_pixels=0,
                thresholds=np.asarray([0.1, 0.25]),
            )


if __name__ == "__main__":
    unittest.main()
