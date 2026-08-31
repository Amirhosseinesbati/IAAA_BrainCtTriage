from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, multi_window_slice, window_hu
from src.strategies.ich_2p5d.data import ICHAdjacentSliceDataset, split_known_slices
from src.strategies.ich_2p5d.evaluation import (
    aggregate_studies,
    evaluate_presence_rule,
    pool_scores,
    select_presence_rule,
)
from src.strategies.ich_2p5d.train import _multilabel_focal_bce
from src.strategies.ich_2p5d.evaluation import PresenceRule
from src.strategies.ich_2p5d.gating import gate_volume_predictions
from scripts.analyze_ich_2p5d_oof import _binary_metrics, _empirical_cdf


class ICH25DTests(unittest.TestCase):
    def test_window_maps_center_and_bounds(self):
        values = np.asarray([-100.0, 40.0, 100.0], dtype=np.float32)
        mapped = window_hu(values, center=40.0, width=80.0)
        np.testing.assert_array_equal(mapped, [0, 128, 255])

    def test_multi_window_shape(self):
        image = np.zeros((8, 8), dtype=np.float32)
        self.assertEqual(multi_window_slice(image, 4).shape, (3, 4, 4))

    def test_adjacent_dataset_clamps_edge_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.npy"
            array = np.zeros((2, 3, 4, 4), dtype=np.uint8)
            array[0] = 10
            array[1] = 20
            np.save(path, array)
            row = {
                "study_id": "1", "slice_index": 0, "cache_path": str(path),
                **{name: 0 for name in OUTPUT_LABELS},
            }
            item = ICHAdjacentSliceDataset(pd.DataFrame([row]))[0]
            self.assertEqual(tuple(item["image"].shape), (9, 4, 4))
            # After (x - 0.5) / 0.25, value 10/255 is repeated for prev/current.
            self.assertTrue(np.allclose(item["image"][:6].numpy(), (10 / 255 - 0.5) / 0.25))

    def test_three_way_split_is_patient_disjoint(self):
        rows = [
            {
                "study_id": str(fold), "slice_index": 0,
                "patient_id": f"p{fold}", "fold": fold, "known": 1,
            }
            for fold in range(5)
        ]
        train, calibration, outer = split_known_slices(
            pd.DataFrame(rows), outer_fold=0, calibration_fold=1
        )
        self.assertEqual(set(train["fold"]), {2, 3, 4})
        self.assertEqual(set(calibration["fold"]), {1})
        self.assertEqual(set(outer["fold"]), {0})

    def test_calibration_rule_is_applied_unchanged_to_outer(self):
        slices = pd.DataFrame([
            {"study_id": "a", "slice_index": 0, "truth_any_ich": 0, "prob_any_ich": 0.1},
            {"study_id": "b", "slice_index": 0, "truth_any_ich": 1, "prob_any_ich": 0.9},
            {"study_id": "b", "slice_index": 1, "truth_any_ich": 1, "prob_any_ich": 0.8},
        ])
        for label in OUTPUT_LABELS[1:]:
            slices[f"truth_{label}"] = 0
            slices[f"prob_{label}"] = 0.0
        studies = aggregate_studies(slices)
        rule = select_presence_rule(studies, minimum_sensitivity=1.0)
        metrics = evaluate_presence_rule(studies, rule)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)

    def test_adjacent_pooling_preserves_order(self):
        self.assertAlmostEqual(pool_scores(np.asarray([0.9, 0.1, 0.8]), "adjacent_pair"), 0.3)

    def test_multilabel_loss_backpropagates(self):
        import torch

        logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, :2] = 1
        loss = _multilabel_focal_bce(
            logits,
            targets,
            pos_weight=torch.ones(len(OUTPUT_LABELS)),
            focal_gamma=1.0,
            any_loss_weight=2.0,
        )
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_negative_gate_zeros_all_volume_channels(self):
        rows = []
        for study_id, triage, truth, score in [("a", 0, 0, 0.1), ("b", 1, 1, 0.9)]:
            row = {
                "study_id": study_id,
                "patient_id": study_id,
                "gt_triage_class": triage,
                "gt_fracture_prob": 0.0,
                "gt_MLS_mm": 0.0,
            }
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                row[f"gt_{key}"] = 1.0 if truth and key == "V_IPH" else 0.0
                row[f"pred_{key}"] = 1.0
            rows.append(row)
        volume = pd.DataFrame(rows)
        presence = pd.DataFrame({
            "study_id": ["a", "b"], "truth_any_ich": [0, 1], "score_max": [0.1, 0.9]
        })
        rule = PresenceRule("max", 0.5, 1.0, 1.0, 1.0)
        gated, summary = gate_volume_predictions(volume, presence, rule)
        self.assertEqual(float(gated.loc[gated["study_id"] == "a", "pred_V_IPH"].iloc[0]), 0.0)
        self.assertEqual(summary["presence_gate"]["suppressed_studies"], 1)

    def test_empirical_cdf_normalizes_fold_scores(self):
        values = _empirical_cdf(np.asarray([0.1, 0.2, 0.9]), np.asarray([0.05, 0.2, 1.0]))
        np.testing.assert_allclose(values, [0.0, 2 / 3, 1.0])

    def test_binary_metrics_reports_specificity(self):
        metrics = _binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]))
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 0.5)


if __name__ == "__main__":
    unittest.main()
