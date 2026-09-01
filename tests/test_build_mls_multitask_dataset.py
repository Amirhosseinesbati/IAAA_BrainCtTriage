from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_mls_multitask_dataset import _clean_negative_study_ids


class MLSMultitaskDatasetMetadataTests(unittest.TestCase):
    def test_clean_negative_ids_use_study_maximum(self) -> None:
        metadata = pd.DataFrame({
            "dicom_series.id": ["a", "a", "b", "b", "c"],
            "MidlineShiftMM": [0.0, 0.1, 0.0, 1.5, 0.05],
        })
        self.assertEqual(_clean_negative_study_ids(metadata, set()), {"a", "c"})

    def test_annotated_target_studies_are_excluded(self) -> None:
        metadata = pd.DataFrame({
            "dicom_series.id": [101, 102],
            "MidlineShiftMM": [0.0, 0.0],
        })
        self.assertEqual(_clean_negative_study_ids(metadata, {"101"}), {"102"})

    def test_invalid_truth_is_rejected(self) -> None:
        metadata = pd.DataFrame({
            "dicom_series.id": ["a"],
            "MidlineShiftMM": ["not-a-number"],
        })
        with self.assertRaises(ValueError):
            _clean_negative_study_ids(metadata, set())

    def test_missing_columns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _clean_negative_study_ids(pd.DataFrame({"other": [1]}), set())


if __name__ == "__main__":
    unittest.main()
