from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.strategies.ich_v2.geometry import (
    dicom_affine_ras,
    remove_small_components,
    volumes_from_labelmap,
    voxel_volume_ml,
)
from src.strategies.ich_v2.evaluation import summarize_ich_predictions
from src.strategies.ich_v2.supervision import (
    ICH_AREA_COLUMNS,
    clean_negative_study_ids,
    stack_partial_targets,
)
from src.strategies.ich_v2.losses import MaskedDiceFocalLoss, masked_teacher_kl


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

    def test_small_components_use_physical_volume(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[0, 0, 0] = 1
        labels[2:4, 2:4, 2:4] = 1
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        cleaned = remove_small_components(labels, affine, minimum_ml=0.005)
        self.assertEqual(int(np.count_nonzero(cleaned == 1)), 8)


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


class TestICHV2Evaluation(unittest.TestCase):
    def test_oracle_context_metric_detects_ich_presence(self):
        rows = []
        for index, (triage, gt_iph, pred_iph) in enumerate([
            (0, 0.0, 0.0),
            (1, 1.0, 1.0),
            (2, 75.0, 75.0),
        ]):
            row = {
                "study_id": str(index),
                "patient_id": str(index),
                "gt_triage_class": triage,
                "gt_fracture_prob": 0.0,
                "gt_MLS_mm": 0.0,
            }
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                row[f"gt_{key}"] = gt_iph if key == "V_IPH" else 0.0
                row[f"pred_{key}"] = pred_iph if key == "V_IPH" else 0.0
            rows.append(row)
        summary = summarize_ich_predictions(pd.DataFrame(rows))
        self.assertEqual(summary["oracle_context_macro_f1"], 1.0)
        self.assertEqual(summary["total"]["presence_f1_at_0_1ml"], 1.0)


class TestICHV2Loss(unittest.TestCase):
    def test_teacher_distillation_ignores_unknown_voxels(self):
        import torch

        student = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
        teacher_a = torch.zeros_like(student)
        teacher_b = teacher_a.clone()
        teacher_b[:, 1, 1, 1, 1] = 100.0
        supervision = torch.ones((1, 1, 2, 2, 2))
        supervision[..., 1, 1, 1] = 0
        loss_a = masked_teacher_kl(student, teacher_a, supervision)
        loss_b = masked_teacher_kl(student, teacher_b, supervision)
        self.assertAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), places=6)

    def test_unknown_voxels_contribute_no_gradient(self):
        import torch

        loss_fn = MaskedDiceFocalLoss(num_classes=2)
        target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
        supervision = torch.ones_like(target, dtype=torch.float32)
        supervision[..., 1, 1, 1] = 0
        first = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
        second = first.detach().clone()
        second[:, 1, 1, 1, 1] = 100.0
        second.requires_grad_(True)
        loss_a = loss_fn(first, target, supervision)
        loss_b = loss_fn(second, target, supervision)
        self.assertAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), places=6)

    def test_clean_negative_patch_has_focal_signal(self):
        import torch

        loss_fn = MaskedDiceFocalLoss(num_classes=2)
        logits = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
        target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
        supervision = torch.ones_like(target, dtype=torch.float32)
        loss = loss_fn(logits, target, supervision)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
