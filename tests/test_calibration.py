from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.calibration import (
    INTERMEDIATE_KEYS, TriageCalibrator, assess_calibration_candidate,
    cross_validate_calibration,
)
from src.inference.postprocessing import remove_small_components, sanitize_intermediates


class TestPostprocessing(unittest.TestCase):
    def test_component_filter_uses_physical_volume(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[2:4, 2:4, 2:4] = 1
        cleaned = remove_small_components(mask, voxel_volume_ml=0.02, min_component_ml={1: 0.1})
        self.assertEqual(cleaned[0, 0, 0], 0)
        self.assertEqual(int((cleaned == 1).sum()), 8)

    def test_sanitize_suppresses_sub_noise_volume(self):
        values = {key: 0.0 for key in INTERMEDIATE_KEYS}
        values.update({"V_IPH": 0.09, "fracture_prob": 1.2, "MLS_mm": -2})
        cleaned = sanitize_intermediates(values)
        self.assertEqual(cleaned["V_IPH"], 0.0)
        self.assertEqual(cleaned["fracture_prob"], 1.0)
        self.assertEqual(cleaned["MLS_mm"], 0.0)


class TestCalibration(unittest.TestCase):
    def make_frame(self):
        rows = []
        for index in range(30):
            label = index % 3
            row = {"fold": index % 5, "patient_id": str(index), "triage_class": label}
            for key in INTERMEDIATE_KEYS:
                truth = 0.0
                if key == "V_IPH" and label > 0:
                    truth = 10.0 if label == 1 else 80.0
                if key == "MLS_mm":
                    truth = [0.1, 3.5, 6.0][label]
                row[f"gt_{key}"] = truth
                row[f"pred_{key}"] = truth * 1.5 + index * 0.001
            rows.append(row)
        return pd.DataFrame(rows)

    def test_round_trip_bundle(self):
        frame = self.make_frame()
        calibrator = TriageCalibrator.fit(frame)
        with tempfile.TemporaryDirectory() as tmp:
            path = calibrator.save(Path(tmp) / "cal.json")
            loaded = TriageCalibrator.load(path)
            values = {key: frame.iloc[0][f"pred_{key}"] for key in INTERMEDIATE_KEYS}
            self.assertEqual(calibrator.transform(values), loaded.transform(values))

    def test_nested_oof(self):
        calibrated, metrics = cross_validate_calibration(self.make_frame())
        self.assertEqual(len(calibrated), 30)
        self.assertIn("macro_f1", metrics)

    def test_candidate_assessment_is_paired_and_auditable(self):
        frame = self.make_frame()
        calibrated, _ = cross_validate_calibration(frame)
        assessment = assess_calibration_candidate(frame, calibrated)
        self.assertIn("accepted", assessment)
        self.assertIn("raw", assessment)
        self.assertIn("candidate", assessment)
        self.assertIn("probability_of_improvement", assessment["paired_bootstrap"])


if __name__ == "__main__":
    unittest.main()
