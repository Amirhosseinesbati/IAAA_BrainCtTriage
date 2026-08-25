from __future__ import annotations

import unittest

import numpy as np

from submission.model import _finalize_intermediates, _remove_small_components


def values():
    return {
        "V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.09,
        "V_SAH": 0.0, "V_IVH": 0.0,
        "fracture_prob": 1.3, "MLS_mm": -1.0,
    }


class TestSubmissionPostprocessing(unittest.TestCase):
    def test_sanitize_without_calibration(self):
        result = _finalize_intermediates(values(), None)
        self.assertEqual(result["V_IPH"], 0.0)
        self.assertEqual(result["fracture_prob"], 1.0)
        self.assertEqual(result["MLS_mm"], 0.0)

    def test_json_calibration_mapping(self):
        mapping = {key: {"x": [0.0, 2.0], "y": [0.0, 1.0]} for key in values()}
        result = _finalize_intermediates({key: 1.0 for key in values()}, {"mappings": mapping})
        self.assertAlmostEqual(result["V_IPH"], 0.5)

    def test_component_filter(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[0, 0, 0] = 2
        mask[2:5, 2:5, 2:5] = 2
        cleaned = _remove_small_components(mask, voxel_vol_ml=0.01)
        self.assertEqual(cleaned[0, 0, 0], 0)
        self.assertEqual(int((cleaned == 2).sum()), 27)


if __name__ == "__main__":
    unittest.main()
