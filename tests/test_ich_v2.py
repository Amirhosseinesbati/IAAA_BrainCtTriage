from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.strategies.ich_v2.geometry import (
    dicom_affine_ras,
    volumes_from_labelmap,
    voxel_volume_ml,
)
from src.strategies.ich_v2.supervision import (
    ICH_AREA_COLUMNS,
    clean_negative_study_ids,
    stack_partial_targets,
)


class TestICHV2Geometry(unittest.TestCase):
    def test_affine_preserves_physical_voxel_volume(self):
        slices = [
            SimpleNamespace(
                ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                PixelSpacing=[0.5, 0.75],
                ImagePositionPatient=[10, 20, 30 + 2 * index],
            )
            for index in range(3)
        ]
        affine = dicom_affine_ras(slices)
        self.assertAlmostEqual(voxel_volume_ml(affine), 0.00075)
        np.testing.assert_allclose(affine[:3, 3], [-10, -20, 30])

    def test_volume_is_measured_in_prediction_space(self):
        labels = np.zeros((2, 2, 2), dtype=np.uint8)
        labels.flat[:3] = 1
        labels.flat[3:5] = 4
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        volumes = volumes_from_labelmap(labels, affine)
        self.assertAlmostEqual(volumes["V_IVH"], 0.006)
        self.assertAlmostEqual(volumes["V_EDH"], 0.004)


class TestICHV2Supervision(unittest.TestCase):
    def test_unknown_json_slice_is_not_background_supervision(self):
        parsed = [
            {"has_label": True, "mask_2d": np.ones((2, 3), dtype=np.uint8)},
            {"has_label": False, "mask_2d": np.zeros((2, 3), dtype=np.uint8)},
        ]
        labels, supervision = stack_partial_targets(parsed, shape=(2, 3))
        self.assertEqual(labels.shape, (2, 2, 3))
        self.assertTrue(np.all(supervision[0] == 1))
        self.assertTrue(np.all(supervision[1] == 0))

    def test_clean_negative_gate_ignores_broken_anyich(self):
        rows = []
        for study, triage, any_ich, ivh in [
            ("10", 0, False, 0.0),
            ("11", 1, False, 0.0),
            ("12", 0, False, 7.0),
        ]:
            row = {"dicom_series.id": study, "triage_class": triage, "AnyICH": any_ich}
            row.update({column: 0.0 for column in ICH_AREA_COLUMNS})
            row["IntraventricularHemorrhage_Area"] = ivh
            rows.append(row)
        self.assertEqual(clean_negative_study_ids(pd.DataFrame(rows)), {"10"})


if __name__ == "__main__":
    unittest.main()
