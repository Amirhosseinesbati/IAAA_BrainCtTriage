from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.oof import ICH_AREA_COLUMNS, assemble_oof_predictions


class TestOOFAssembly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.fold_path = root / "folds.csv"
        self.metadata_path = root / "metadata.csv"
        fold_rows, metadata_rows = [], []
        for index in range(10):
            study = str(1000 + index)
            fold_rows.append({
                "study_id": study, "patient_id": str(2000 + index),
                "triage_class": index % 3, "fold": index % 5,
            })
            row = {
                "dicom_series.id": study,
                "dicom_series.PixelSpacing0": 0.5,
                "dicom_series.PixelSpacing1": 0.5,
                "dicom_series.SliceThickness": 5.0,
                "SkullFracture": index % 2 == 0,
                "MidlineShiftMM": float(index),
            }
            row.update({column: float(index) for column in ICH_AREA_COLUMNS.values()})
            metadata_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(self.fold_path, index=False)
        pd.DataFrame(metadata_rows).to_csv(self.metadata_path, index=False)
        self.ids = [str(1000 + index) for index in range(10)]

    def tearDown(self):
        self.temp.cleanup()

    def frames(self):
        folds = [index % 5 for index in range(10)]
        ich = pd.DataFrame({"study_id": self.ids, "fold": folds})
        for key in ICH_AREA_COLUMNS:
            ich[f"pred_{key}"] = 0.1
        fracture = pd.DataFrame({
            "study_id": self.ids, "fold": folds, "pred_fracture_prob": 0.2,
        })
        mls = pd.DataFrame({"study_id": self.ids, "fold": folds, "pred_MLS_mm": 2.0})
        return ich, fracture, mls

    def test_complete_tasks_assemble_once_per_study(self):
        result = assemble_oof_predictions(
            *self.frames(), fold_manifest_path=self.fold_path,
            metadata_path=self.metadata_path,
        )
        self.assertEqual(len(result), 10)
        self.assertEqual(result["study_id"].nunique(), 10)
        self.assertIn("pred_triage", result)
        self.assertIn("gt_V_IPH", result)

    def test_missing_study_is_rejected(self):
        ich, fracture, mls = self.frames()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            assemble_oof_predictions(
                ich.iloc[:-1], fracture, mls,
                fold_manifest_path=self.fold_path, metadata_path=self.metadata_path,
            )

    def test_wrong_reported_fold_is_rejected(self):
        ich, fracture, mls = self.frames()
        mls.loc[0, "fold"] = 4
        with self.assertRaisesRegex(ValueError, "incorrect folds"):
            assemble_oof_predictions(
                ich, fracture, mls,
                fold_manifest_path=self.fold_path, metadata_path=self.metadata_path,
            )


if __name__ == "__main__":
    unittest.main()
