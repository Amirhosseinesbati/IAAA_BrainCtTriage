from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_ich_ivh_components import (
    component_measurements,
    numeric_summary,
)


class AnalyzeICHIVHComponentsTests(unittest.TestCase):
    def test_component_measurements_use_eight_connectivity(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 1
        mask[4, 4] = 1
        areas, volumes = component_measurements(mask, voxel_volume_ml=0.02)
        self.assertEqual(areas, [2, 1])
        np.testing.assert_allclose(volumes, [0.04, 0.02])

    def test_empty_mask_and_numeric_summary_are_explicit(self):
        areas, volumes = component_measurements(
            np.zeros((3, 3), dtype=np.uint8), voxel_volume_ml=0.01
        )
        self.assertEqual(areas, [])
        self.assertEqual(volumes, [])
        self.assertEqual(numeric_summary([])["count"], 0)
        summary = numeric_summary([1, 2, 9])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["maximum"], 9.0)

    def test_nonpositive_voxel_volume_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "voxel_volume_ml"):
            component_measurements(np.ones((2, 2)), voxel_volume_ml=0.0)


if __name__ == "__main__":
    unittest.main()
