"""Contract tests for the G1 float32 MLS 2.5D training path.

These are intentionally small, synthetic tests.  They prove the image/cache
contract without loading competition DICOMs or executing a model forward.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError

from scripts.build_mls_2p5d_cache import _full_slice_target_orders, _load_compatible_existing_manifest
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file
from src.strategies.mls_heatmap.dataset import MLSHeatmapDataset
from src.strategies.mls_heatmap.input_contract import (
    create_study_windowed_input,
    create_windowed_input,
)
from src.strategies.mls_heatmap.model import HRNetHeatmapModel


_CACHE_HASH = "a" * 64


class MLS25DInputContractTests(unittest.TestCase):
    def test_adjacent_slice_order_and_edges_are_exact(self) -> None:
        volume = np.stack([
            np.full((2, 2), -1000.0, dtype=np.float32),
            np.full((2, 2), 0.0, dtype=np.float32),
            np.full((2, 2), 1000.0, dtype=np.float32),
        ], axis=-1)
        middle = create_study_windowed_input(volume, 1, 9)
        np.testing.assert_array_equal(middle[:3], create_windowed_input(volume[:, :, 0], 3))
        np.testing.assert_array_equal(middle[3:6], create_windowed_input(volume[:, :, 1], 3))
        np.testing.assert_array_equal(middle[6:9], create_windowed_input(volume[:, :, 2], 3))

        first = create_study_windowed_input(volume, 0, 9)
        last = create_study_windowed_input(volume, 2, 9)
        np.testing.assert_array_equal(first[:3], first[3:6])
        np.testing.assert_array_equal(last[3:6], last[6:9])

    def test_full_slice_target_order_requires_contiguous_z(self) -> None:
        frame = pd.DataFrame({
            "study_id": ["1011", "1011", "1011"],
            "sop_instance_uid": ["a", "b", "c"],
            "slice_index": [0, 1, 2],
        })
        self.assertEqual(_full_slice_target_orders(frame), {"1011": ["a", "b", "c"]})
        frame.loc[2, "slice_index"] = 3
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            _full_slice_target_orders(frame)

    def test_config_locks_2p5d_geometry_and_selector(self) -> None:
        config = MLSHeatmapConfig(
            dataset_variant="multitask_2p5d_v1",
            input_channels=9,
            context_cache_manifest_sha256=_CACHE_HASH,
            context_cache_validation_receipt_sha256=_CACHE_HASH,
            use_selector=True,
            image_size=512,
        )
        self.assertEqual(config.input_channels, 9)
        for override in ({"image_size": 256}, {"use_selector": False}):
            values = config.model_dump()
            values.update(override)
            with self.subTest(override=override), self.assertRaises(ValidationError):
                MLSHeatmapConfig.model_validate(values)

    def test_9ch_adaptation_does_not_advance_head_rng(self) -> None:
        torch.manual_seed(20260905)
        model = HRNetHeatmapModel.__new__(HRNetHeatmapModel)
        torch.nn.Module.__init__(model)
        model.in_channels = 9
        model.backbone = torch.nn.Module()
        model.backbone.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        old_weight = model.backbone.conv1.weight.detach().clone()
        expected = torch.get_rng_state().clone()
        model._adapt_input_channels()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        self.assertTrue(torch.equal(model.backbone.conv1.weight[:, 3:6], old_weight))
        self.assertTrue(torch.count_nonzero(model.backbone.conv1.weight[:, :3]).item() == 0)
        self.assertTrue(torch.count_nonzero(model.backbone.conv1.weight[:, 6:]).item() == 0)


class MLS25DCacheTests(unittest.TestCase):
    def test_memmap_cache_yields_writable_contiguous_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            studies = root / "studies"
            studies.mkdir()
            volume = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
            np.save(studies / "1011.npy", volume, allow_pickle=False)
            labels = pd.DataFrame([{
                "patient_id": "1011", "sop_instance_uid": "uid",
                "image_name": "1011_uid.png", "is_target": 0,
                "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0,
                "slice_index": 0, "slice_target_index": 0, "fold": 3,
                "raw_dicom_count": 2, "spacing_x": 0.5, "spacing_y": 0.5,
                "study_mls_mm": 0.0,
            }])
            labels_path = root / "labels_context.csv"
            labels.to_csv(labels_path, index=False)
            manifest = {
                "schema_version": 1,
                "cache_contract": "mls_2p5d_float32_v1",
                "image_size": 4,
                "base_input_channels": 3,
                "context_input_channels": 9,
                "cache_dtype": "float32",
                "edge_policy": "replicate",
                "labels_csv": labels_path.name,
                "labels_sha256": sha256_file(labels_path),
                "study_cache_dir": studies.name,
                "studies": 1,
                "study_files": {"1011": {
                    "file": "1011.npy",
                    "bytes": (studies / "1011.npy").stat().st_size,
                    "shape": [2, 3, 4, 4],
                }},
                "rows": 1,
                "window_order": ["brain", "subdural", "bone"],
            }
            (root / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            dataset = MLSHeatmapDataset(
                csv_path=str(labels_path), img_dir=str(root), img_size=4, heatmap_size=1,
                include_negatives=True, return_selector=True, input_channels=9,
                context_cache_root=root,
            )
            image = dataset[0][0]
            self.assertTrue(image.is_contiguous())
            self.assertEqual(tuple(image.shape), (9, 4, 4))

    def test_cache_refuses_unpinned_labels_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            studies = root / "studies"
            studies.mkdir()
            volume = np.zeros((1, 3, 4, 4), dtype=np.float32)
            np.save(studies / "1011.npy", volume, allow_pickle=False)
            labels = pd.DataFrame([{
                "patient_id": "1011", "sop_instance_uid": "u", "image_name": "1011_u.png",
                "is_target": 0, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0,
                "slice_index": 0, "slice_target_index": 0, "fold": 3,
                "raw_dicom_count": 1, "spacing_x": 0.5, "spacing_y": 0.5, "study_mls_mm": 0.0,
            }])
            labels_path = root / "labels_context.csv"
            labels.to_csv(labels_path, index=False)
            manifest = {
                "schema_version": 1, "cache_contract": "mls_2p5d_float32_v1", "image_size": 4,
                "base_input_channels": 3, "context_input_channels": 9, "cache_dtype": "float32",
                "edge_policy": "replicate", "labels_csv": labels_path.name,
                "labels_sha256": sha256_file(labels_path), "study_cache_dir": studies.name,
                "studies": 1, "study_files": {"1011": {
                    "file": "1011.npy", "bytes": (studies / "1011.npy").stat().st_size,
                    "shape": [1, 3, 4, 4],
                }}, "rows": 1, "window_order": ["brain", "subdural", "bone"],
            }
            (root / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest-pinned labels"):
                MLSHeatmapDataset(
                    csv_path=str(root / "other.csv"), img_dir=str(root), img_size=4, heatmap_size=1,
                    include_negatives=True, return_selector=True, input_channels=9,
                    context_cache_root=root,
                )

    def test_interrupted_cache_without_manifest_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "studies").mkdir()
            (root / "studies" / "1011.npy").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "no finalized manifest"):
                _load_compatible_existing_manifest(
                    root,
                    labels_sha256="a", slice_targets_sha256="b", raw_metadata_sha256="c",
                    fold_manifest_sha256="d", input_contract_sha256="e", builder_sha256="f",
                    image_size=512,
                )


if __name__ == "__main__":
    unittest.main()
