from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, resize_label_slice
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    split_segmentation_slices,
)
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_loss import ICH25DSegmentationLoss


class ICH25DSegmentationTests(unittest.TestCase):
    def test_label_resize_preserves_categorical_values(self):
        label = np.asarray([[0, 3], [5, 1]], dtype=np.uint8)
        resized = resize_label_slice(label, 8)
        self.assertEqual(resized.shape, (8, 8))
        self.assertEqual(set(np.unique(resized)), {0, 1, 3, 5})

    def test_split_trains_on_known_but_evaluates_all_slices(self):
        rows = []
        for fold in range(5):
            for known in (0, 1):
                rows.append({
                    "study_id": f"{fold}-{known}",
                    "patient_id": f"p{fold}",
                    "fold": fold,
                    "slice_index": known,
                    "known": known,
                })
        train, calibration, outer = split_segmentation_slices(
            pd.DataFrame(rows), outer_fold=0, calibration_fold=1
        )
        self.assertTrue((train["known"] == 1).all())
        self.assertEqual(set(calibration["known"]), {0, 1})
        self.assertEqual(set(outer["known"]), {0, 1})
        self.assertEqual(set(train["fold"]), {2, 3, 4})

    def test_dataset_returns_registered_center_mask_and_physical_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            label_path = root / "label.npy"
            image = np.zeros((3, 3, 8, 8), dtype=np.uint8)
            image[1] = 128
            label = np.zeros((3, 8, 8), dtype=np.uint8)
            label[1, 2:4, 3:6] = 3
            np.save(image_path, image)
            np.save(label_path, label)
            row = {
                "study_id": "a",
                "patient_id": "p",
                "slice_index": 1,
                "known": 1,
                "cache_path": str(image_path),
                "label_cache_path": str(label_path),
                "resized_voxel_volume_ml": 0.002,
                **{name: int(name in {"any_ich", "SDH"}) for name in OUTPUT_LABELS},
            }
            item = ICHAdjacentSegmentationDataset(pd.DataFrame([row]))[0]
            self.assertEqual(tuple(item["image"].shape), (9, 8, 8))
            self.assertEqual(int((item["mask"] == 3).sum()), 6)
            self.assertAlmostEqual(float(item["voxel_volume_ml"]), 0.002)

    def test_multitask_loss_backpropagates(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS))
        )
        mask_logits = torch.zeros((2, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((2, 8, 8), dtype=torch.long)
        masks[0, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[0, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            torch.ones(2),
        )
        components["loss"].backward()
        self.assertGreater(float(mask_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(class_logits.grad.abs().sum()), 0.0)

    def test_evaluation_uses_only_ich_truth_and_physical_pixels(self):
        rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            row = {
                "study_id": study_id,
                "known": 1,
                "voxel_volume_ml": 0.01,
                "prob_any_ich": 0.9 if has_ich else 0.1,
            }
            for label in OUTPUT_LABELS[1:]:
                is_iph = has_ich and label == "IPH"
                row[f"prob_{label}"] = 0.9 if is_iph else 0.1
                row[f"pred_pixels_{label}"] = 100 if is_iph else 0
                row[f"intersection_{label}"] = 100 if is_iph else 0
                row[f"predicted_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"observed_known_pixels_{label}"] = 100 if is_iph else 0
            rows.append(row)
        truth_rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            truth_row = {"study_id": study_id}
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                truth_row[f"gt_{key}"] = 1.0 if has_ich and key == "V_IPH" else 0.0
            truth_rows.append(truth_row)
        studies, summary = summarize_segmentation_predictions(
            pd.DataFrame(rows), pd.DataFrame(truth_rows)
        )
        self.assertEqual(summary["evaluation_scope"], "ich_only_no_mls_no_fracture_no_triage")
        self.assertEqual(summary["presence_f1_at_0_1ml"], 1.0)
        self.assertAlmostEqual(
            float(studies.loc[studies["study_id"] == "positive", "pred_V_IPH"].iloc[0]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
