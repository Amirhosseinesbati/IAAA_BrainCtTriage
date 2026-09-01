from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import scheduled_heatmap_sigma


class MLSHeatmapSigmaScheduleTests(unittest.TestCase):
    def test_none_preserves_historical_fixed_target(self) -> None:
        values = [scheduled_heatmap_sigma(3.0, None, epoch, 23) for epoch in (1, 12, 23)]
        self.assertEqual(values, [3.0, 3.0, 3.0])

    def test_symmetric_schedule_preserves_midpoint_and_mean(self) -> None:
        values = [scheduled_heatmap_sigma(3.0, 2.0, epoch, 23) for epoch in range(1, 24)]
        self.assertAlmostEqual(values[0], 4.0)
        self.assertAlmostEqual(values[11], 3.0)
        self.assertAlmostEqual(values[-1], 2.0)
        self.assertAlmostEqual(sum(values) / len(values), 3.0)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_resume_epoch_maps_to_same_width(self) -> None:
        uninterrupted = scheduled_heatmap_sigma(3.0, 2.0, 15, 23)
        resumed = scheduled_heatmap_sigma(3.0, 2.0, 15, 23)
        self.assertEqual(uninterrupted, resumed)

    def test_config_rejects_fine_to_coarse_schedule(self) -> None:
        with self.assertRaises(ValidationError):
            MLSHeatmapConfig(heatmap_sigma=3.0, heatmap_sigma_anneal_end=4.0)

    def test_config_rejects_schedule_start_above_supported_range(self) -> None:
        with self.assertRaises(ValidationError):
            MLSHeatmapConfig(heatmap_sigma=6.0, heatmap_sigma_anneal_end=2.0)


if __name__ == "__main__":
    unittest.main()
