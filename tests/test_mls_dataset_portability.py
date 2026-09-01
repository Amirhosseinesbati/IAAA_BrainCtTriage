from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.strategies.mls_heatmap.dataset import (
    MLSHeatmapDataset,
    resolve_mls_image_path,
)


class MLSDatasetPortabilityTests(unittest.TestCase):
    def test_windows_absolute_path_rebases_to_current_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            expected = project / "Data" / "processed" / "mls_dataset" / "images" / "sample.png"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"png")
            resolved = resolve_mls_image_path(
                r"D:\old\repo\Data\processed\mls_dataset\images\sample.png",
                "sample.png",
                project / "Data" / "processed" / "mls_multitask_v2" / "images",
                project_root=project,
            )
            self.assertEqual(resolved, expected)

    def test_image_dir_fallback_handles_relative_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_dir = Path(temporary) / "images"
            image_dir.mkdir()
            expected = image_dir / "negative.png"
            expected.write_bytes(b"png")
            self.assertEqual(
                resolve_mls_image_path("", "negative.png", image_dir),
                expected,
            )

    def test_raw_pickle_supplies_spacing_and_study_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "missing.csv"
            pickle_path = root / "training_df.pkl"
            pd.DataFrame({
                "dicom_series.id": [101, 101, 102],
                "dicom_series.PixelSpacing1": [0.45, 0.47, 0.50],
                "MidlineShiftMM": [2.0, 3.0, 0.0],
            }).to_pickle(pickle_path)
            labels = pd.DataFrame({
                "patient_id": ["101", "102"],
                "spacing_x": [float("nan"), float("nan")],
                "study_mls_mm": [float("nan"), float("nan")],
            })
            with (
                patch("src.strategies.mls_heatmap.dataset.TRAINING_CSV_PATH", csv_path),
                patch("src.strategies.mls_heatmap.dataset.TRAINING_PKL_PATH", pickle_path),
            ):
                result = MLSHeatmapDataset._attach_spacing(labels)
            self.assertAlmostEqual(result.loc[0, "spacing_x"], 0.46)
            self.assertAlmostEqual(result.loc[0, "study_mls_mm"], 3.0)
            self.assertAlmostEqual(result.loc[1, "spacing_x"], 0.50)
            self.assertAlmostEqual(result.loc[1, "study_mls_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
