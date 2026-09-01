from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.compare_ich_2p5d_segmentation_oof import _metric_vector
from scripts.evaluate_ich_2p5d_segmentation_checkpoint import checkpoint_config
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, resize_label_slice
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    segmentation_foreground_weights,
    split_segmentation_slices,
    subtype_aware_sampling_weights,
)
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_loss import ICH25DSegmentationLoss
from src.strategies.ich_2p5d.segmentation_train import (
    _flatten_summary_metrics,
    checkpoint_selection_score,
)


class ICH25DSegmentationTests(unittest.TestCase):
    def test_fpr_penalized_checkpoint_score_uses_preregistered_tradeoff(self):
        summary = {
            "selection_score": 0.61,
            "normal_false_positive_rate_at_0_1ml": 0.36,
        }
        self.assertAlmostEqual(
            checkpoint_selection_score(summary, "fpr_penalized"), 0.574
        )
        self.assertEqual(checkpoint_selection_score(summary, "legacy"), 0.61)
        with self.assertRaisesRegex(ValueError, "checkpoint_selection_strategy"):
            checkpoint_selection_score(summary, "unknown")

    def test_foreground_weights_emphasize_rare_slice_labels(self):
        frame = pd.DataFrame({
            "IVH": [1] * 2 + [0] * 8,
            "IPH": [1] * 10,
            "SDH": [1] * 5 + [0] * 5,
            "EDH": [1] + [0] * 9,
            "SAH": [1] * 4 + [0] * 6,
        })
        weights = segmentation_foreground_weights(frame, power=1.0, maximum=8.0)
        np.testing.assert_allclose(weights.numpy(), [5.0, 1.0, 2.0, 8.0, 2.5])

    def test_foreground_weights_can_use_only_supervised_mask_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "labels.npy"
            labels = np.zeros((3, 4, 4), dtype=np.uint8)
            labels[0].reshape(-1)[:8] = 1
            labels[0].reshape(-1)[8:12] = 2
            labels[0].reshape(-1)[12:14] = 3
            labels[0].reshape(-1)[14] = 4
            labels[0].reshape(-1)[15] = 5
            labels[1] = 5
            labels[2] = 2
            np.save(label_path, labels)
            frame = pd.DataFrame({
                "label_cache_path": [str(label_path)] * 3,
                "slice_index": [0, 1, 2],
                "segmentation_known": [1, 1, 0],
            })
            weights = segmentation_foreground_weights(
                frame, power=1.0, maximum=8.0, basis="pixel"
            )
            np.testing.assert_allclose(
                weights.numpy(), [2.125, 4.25, 8.0, 8.0, 1.0]
            )

    def test_foreground_weight_basis_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "basis"):
            segmentation_foreground_weights(
                pd.DataFrame(), power=1.0, maximum=8.0, basis="voxel"
            )

    def test_study_balanced_sampler_preserves_positive_mass_and_equalizes_studies(self):
        rows = []
        for study_id, subtype, slices in (
            ("ivh-small", "IVH", 1),
            ("ivh-large", "IVH", 4),
            ("iph", "IPH", 1),
            ("sdh", "SDH", 1),
            ("edh", "EDH", 1),
            ("sah", "SAH", 1),
        ):
            for _ in range(slices):
                rows.append({
                    "study_id": study_id,
                    **{label: int(label == subtype) for label in OUTPUT_LABELS[1:]},
                })
        rows.append({
            "study_id": "normal",
            **{label: 0 for label in OUTPUT_LABELS[1:]},
        })
        frame = pd.DataFrame(rows)
        original = subtype_aware_sampling_weights(frame, study_balance_power=0.0)
        balanced = subtype_aware_sampling_weights(frame, study_balance_power=1.0)
        positive = frame[list(OUTPUT_LABELS[1:])].any(axis=1).to_numpy()

        self.assertAlmostEqual(
            float(original[positive].sum()), float(balanced[positive].sum())
        )
        small_mass = float(balanced[frame["study_id"] == "ivh-small"].sum())
        large_mass = float(balanced[frame["study_id"] == "ivh-large"].sum())
        self.assertAlmostEqual(small_mass, large_mass)
        self.assertLess(
            float(original[frame["study_id"] == "ivh-small"].sum()),
            float(original[frame["study_id"] == "ivh-large"].sum()),
        )
        self.assertEqual(float(balanced[~positive].item()), 1.0)

    def test_study_balanced_sampler_power_is_validated(self):
        frame = pd.DataFrame({
            "study_id": ["a"],
            **{label: [1] for label in OUTPUT_LABELS[1:]},
        })
        with self.assertRaisesRegex(ValueError, "study-balance power"):
            subtype_aware_sampling_weights(frame, study_balance_power=1.1)

    def test_label_resize_preserves_categorical_values(self):
        label = np.asarray([[0, 3], [5, 1]], dtype=np.uint8)
        resized = resize_label_slice(label, 8)
        self.assertEqual(resized.shape, (8, 8))
        self.assertEqual(set(np.unique(resized)), {0, 1, 3, 5})

    def test_split_keeps_classification_only_rows_but_evaluates_all_slices(self):
        rows = []
        for fold in range(5):
            for known in (0, 1):
                rows.append({
                    "study_id": f"{fold}-{known}",
                    "patient_id": f"p{fold}",
                    "fold": fold,
                    "slice_index": known,
                    "known": known,
                    "classification_known": 1,
                    "segmentation_known": known,
                    "metadata_missing": 0,
                })
        train, calibration, outer = split_segmentation_slices(
            pd.DataFrame(rows), outer_fold=0, calibration_fold=1
        )
        self.assertEqual(set(train["segmentation_known"]), {0, 1})
        self.assertTrue((train["classification_known"] == 1).all())
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
                "classification_known": 1,
                "segmentation_known": 1,
                "metadata_missing": 0,
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

    def test_known_empty_mask_gets_non_focal_foreground_penalty(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((2, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((2, 8, 8), dtype=torch.long)
        masks[1, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[1, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.ones(2),
        )
        self.assertAlmostEqual(
            float(components["empty_foreground"].detach()),
            float(np.log(6.0)),
            places=5,
        )
        components["loss"].backward()
        self.assertLess(float(mask_logits.grad[0, 0].mean()), 0.0)
        self.assertGreater(float(mask_logits.grad[0, 1:].mean()), 0.0)

    def test_positive_only_batch_has_no_empty_foreground_penalty(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((1, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        masks[0, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[0, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.ones(1),
        )
        self.assertEqual(float(components["empty_foreground"].detach()), 0.0)

    def test_empty_foreground_top_fraction_focuses_sparse_hard_pixels(self):
        common = {
            "classification_pos_weight": torch.ones(len(OUTPUT_LABELS)),
            "empty_foreground_weight": 0.05,
        }
        average_loss = ICH25DSegmentationLoss(
            **common, empty_foreground_top_fraction=1.0
        )
        hard_loss = ICH25DSegmentationLoss(
            **common, empty_foreground_top_fraction=1.0 / 64.0
        )
        mask_logits = torch.full((1, 6, 8, 8), -4.0)
        mask_logits[:, 0] = 4.0
        mask_logits[:, 1, 0, 0] = 8.0
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)))
        targets = torch.zeros_like(class_logits)
        arguments = (
            mask_logits,
            class_logits,
            masks,
            targets,
            torch.ones(1),
        )
        average = average_loss.components(*arguments)["empty_foreground"]
        hard = hard_loss.components(*arguments)["empty_foreground"]
        self.assertGreater(float(hard), float(average) * 20.0)

    def test_empty_foreground_top_fraction_is_validated(self):
        with self.assertRaisesRegex(ValueError, "top_fraction"):
            ICH25DSegmentationLoss(
                classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
                empty_foreground_top_fraction=0.0,
            )

    def test_classification_only_row_has_no_segmentation_gradient(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((1, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        targets = torch.zeros_like(class_logits)
        targets[0, :2] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.zeros(1),
            classification_known=torch.ones(1),
        )
        components["loss"].backward()
        self.assertEqual(float(components["segmentation"].detach()), 0.0)
        self.assertTrue(mask_logits.grad is None or not mask_logits.grad.any())
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

    def test_evaluation_reports_small_ivh_quality_without_changing_selection(self):
        specifications = (
            ("small-hit", 1.0, 1.0, 10, 10),
            ("small-miss", 2.0, 0.0, 10, 0),
            ("medium", 5.0, 4.0, 10, 8),
            ("large", 12.0, 9.0, 10, 9),
            ("normal", 0.0, 0.0, 0, 0),
        )
        prediction_rows = []
        truth_rows = []
        for (
            study_id,
            true_ivh,
            predicted_ivh,
            observed_pixels,
            intersection,
        ) in specifications:
            row = {
                "study_id": study_id,
                "known": 1,
                "voxel_volume_ml": 1.0,
                "prob_any_ich": 0.9 if predicted_ivh > 0 else 0.1,
            }
            for label in OUTPUT_LABELS[1:]:
                is_ivh = label == "IVH"
                row[f"prob_{label}"] = 0.9 if is_ivh and predicted_ivh > 0 else 0.1
                row[f"pred_pixels_{label}"] = int(predicted_ivh) if is_ivh else 0
                row[f"intersection_{label}"] = intersection if is_ivh else 0
                row[f"predicted_known_pixels_{label}"] = (
                    int(predicted_ivh) if is_ivh else 0
                )
                row[f"observed_known_pixels_{label}"] = (
                    observed_pixels if is_ivh else 0
                )
            prediction_rows.append(row)
            truth_row = {"study_id": study_id}
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                truth_row[f"gt_{key}"] = true_ivh if key == "V_IVH" else 0.0
            truth_rows.append(truth_row)

        _, summary = summarize_segmentation_predictions(
            pd.DataFrame(prediction_rows), pd.DataFrame(truth_rows)
        )
        small = summary["subtypes"]["IVH"]["volume_strata"]["small_le_2ml"]
        medium = summary["subtypes"]["IVH"]["volume_strata"][
            "medium_gt_2_le_10ml"
        ]
        large = summary["subtypes"]["IVH"]["volume_strata"]["large_gt_10ml"]
        self.assertEqual(small["positive_studies"], 2)
        self.assertAlmostEqual(small["presence_sensitivity_at_0_1ml"], 0.5)
        self.assertAlmostEqual(small["mae_ml"], 1.0)
        self.assertAlmostEqual(small["dice_known_pixels"], 20.0 / 21.0)
        self.assertEqual(medium["positive_studies"], 1)
        self.assertEqual(large["positive_studies"], 1)
        self.assertAlmostEqual(
            summary["selection_score"],
            0.55 * summary["mean_foreground_dice"]
            + 0.30 * float(summary["any_ich_study_auc"] or 0.0)
            + 0.15 * summary["macro_subtype_study_auc"],
        )
        flattened = _flatten_summary_metrics("calibration", summary)
        self.assertEqual(
            flattened["calibration_ivh_small_le_2ml_positive_studies"], 2.0
        )
        self.assertAlmostEqual(
            flattened[
                "calibration_ivh_small_le_2ml_presence_sensitivity_at_0_1ml"
            ],
            0.5,
        )
        self.assertNotIn("volume_strata", summary)

    def test_oof_metric_vector_reconstructs_perfect_ich_metrics(self):
        rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            row = {
                "study_id": study_id,
                "score_any_ich": 0.9 if has_ich else 0.1,
            }
            for volume_key, label in {
                "V_IVH": "IVH",
                "V_IPH": "IPH",
                "V_SDH": "SDH",
                "V_EDH": "EDH",
                "V_SAH": "SAH",
            }.items():
                is_iph = has_ich and label == "IPH"
                row[f"intersection_{label}"] = 100 if is_iph else 0
                row[f"predicted_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"observed_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"score_{label}"] = 0.9 if is_iph else 0.1
                row[f"gt_{volume_key}"] = 1.0 if is_iph else 0.0
                row[f"pred_{volume_key}"] = 1.0 if is_iph else 0.0
            rows.append(row)
        metrics = _metric_vector(pd.DataFrame(rows), np.ones(2))
        self.assertEqual(metrics["selection_score"], 1.0)
        self.assertEqual(metrics["mean_foreground_dice"], 1.0)
        self.assertEqual(metrics["presence_f1_at_0_1ml"], 1.0)
        self.assertEqual(metrics["normal_false_positive_rate_at_0_1ml"], 0.0)
        self.assertEqual(metrics["total_volume_mae_ml"], 0.0)

    def test_recovery_evaluator_requires_patient_safe_checkpoint_config(self):
        config = {
            "architecture": "unetplusplus",
            "encoder_name": "efficientnet-b2",
            "outer_fold": 0,
            "calibration_fold": 1,
            "batch_size": 16,
            "workers": 4,
            "dropout": 0.2,
            "seed": 42,
        }
        self.assertEqual(checkpoint_config({"config": config}), config)
        with self.assertRaisesRegex(ValueError, "must differ"):
            checkpoint_config({"config": {**config, "calibration_fold": 0}})


if __name__ == "__main__":
    unittest.main()
