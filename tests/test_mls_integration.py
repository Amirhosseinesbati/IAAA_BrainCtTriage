"""
test_mls_integration.py — Unit tests for the heatmap MLS pipeline integration.

Covers the pieces that were added/completed in the strategy integration work:

1. Single heatmap checkpoint path resolution
2. Windowed input channel handling (_create_windowed_input)
3. Masked MSE loss — zero gradient for missing keypoints
4. Validation metrics computed against TRUE keypoints
5. End-to-end _run_pipeline numeric chain (window → heatmap → DARK → MLS)
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from src.strategies.mls_heatmap.predict import (
    _resolve_checkpoint_path,
    _create_windowed_input,
    _create_3channel_window,
    _run_pipeline,
)
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap
from src.strategies.mls_heatmap.predict_multitask import (
    SliceMLSPrediction,
    aggregate_study_mls,
)
from src.strategies.mls_heatmap.train import (
    _compute_validation_metrics,
    competition_aware_heatmap_loss,
    differentiable_mls_mm,
)
from src.strategies.mls_heatmap.train_multitask import multitask_loss
from src.strategies.config_models import MLSHeatmapConfig


class TestStudyPooling(unittest.TestCase):
    """Study pooling must preserve slice adjacency around the selector peak."""

    def setUp(self):
        probabilities = [0.05, 0.30, 0.90, 0.80, 0.20, 0.70]
        values = [0.5, 1.0, 3.0, 5.0, 7.0, 50.0]
        self.predictions = [
            SliceMLSPrediction(index=i, selector_probability=p, mls_mm=value, heatmap_peak=1.0)
            for i, (p, value) in enumerate(zip(probabilities, values))
        ]

    def test_anchor_window_excludes_distant_high_selector_slice(self):
        result = aggregate_study_mls(
            self.predictions,
            selector_threshold=0.5,
            aggregation="anchor_window",
            anchor_window_radius=1,
            aggregation_quantile=0.5,
        )
        self.assertEqual(result, 3.0)

    def test_topk_quantile_uses_configured_quantile(self):
        result = aggregate_study_mls(
            self.predictions,
            selector_threshold=0.5,
            top_k=3,
            aggregation="quantile",
            aggregation_quantile=0.75,
        )
        self.assertEqual(result, 27.5)

    def test_new_config_values_validate(self):
        config = MLSHeatmapConfig(
            aggregation="joint_component",
            anchor_window_radius=2,
            aggregation_quantile=0.65,
            min_active_slices=3,
            heatmap_guard_ratio=0.5,
        )
        self.assertEqual(config.anchor_window_radius, 2)
        self.assertEqual(config.min_active_slices, 3)

    def test_severity_window_uses_plausible_high_mls_neighbourhood(self):
        result = aggregate_study_mls(
            self.predictions,
            selector_threshold=0.5,
            min_active_slices=2,
            aggregation="severity_window",
            anchor_window_radius=1,
            aggregation_quantile=0.75,
        )
        self.assertGreater(result, 3.0)

    def test_min_active_slice_gate_returns_negative_value(self):
        result = aggregate_study_mls(
            self.predictions,
            selector_threshold=0.85,
            min_active_slices=2,
            aggregation="joint_component",
            negative_value=0.1,
        )
        self.assertEqual(result, 0.1)

    def test_joint_component_uses_heatmap_guard(self):
        predictions = [
            SliceMLSPrediction(0, 0.91, 2.0, 1.0),
            SliceMLSPrediction(1, 0.95, 20.0, 0.3),
            SliceMLSPrediction(2, 0.92, 4.0, 1.0),
        ]
        result = aggregate_study_mls(
            predictions,
            selector_threshold=0.9,
            min_active_slices=3,
            aggregation="joint_component",
            relative_ratio=0.5,
            aggregation_quantile=0.9,
            heatmap_guard_ratio=0.5,
        )
        self.assertAlmostEqual(result, 3.8)


class TestCheckpointResolution(unittest.TestCase):
    """Tests for the single heatmap checkpoint resolver."""

    def test_explicit_paths(self):
        """Explicitly provided paths are returned unchanged."""
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "hm.pth")
        open(path, "w").close()
        self.assertEqual(_resolve_checkpoint_path(path), path)

    def test_env_vars(self):
        """Paths are resolved from MLS_*_PATH environment variables."""
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "hm.pth")
        open(path, "w").close()
        os.environ["MLS_HEATMAP_MODEL_PATH"] = path
        try:
            self.assertEqual(_resolve_checkpoint_path(), path)
        finally:
            del os.environ["MLS_HEATMAP_MODEL_PATH"]

    def test_missing_paths_raise(self):
        """Non-existent resolved paths raise a clear FileNotFoundError."""
        os.environ["MLS_HEATMAP_MODEL_PATH"] = "/nonexistent/hm.pth"
        try:
            with self.assertRaises(FileNotFoundError):
                _resolve_checkpoint_path()
        finally:
            del os.environ["MLS_HEATMAP_MODEL_PATH"]


class TestWindowedInput(unittest.TestCase):
    """Tests for _create_windowed_input channel handling."""

    def setUp(self):
        self.hu = np.random.uniform(-100, 100, (512, 512)).astype(np.float32)

    def test_three_channels(self):
        out = _create_windowed_input(self.hu, 3)
        self.assertEqual(out.shape, (3, 512, 512))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_single_channel(self):
        out = _create_windowed_input(self.hu, 1)
        self.assertEqual(out.shape, (1, 512, 512))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_legacy_alias(self):
        """Deprecated 3-channel alias still returns 3 channels."""
        out = _create_3channel_window(self.hu)
        self.assertEqual(out.shape, (3, 512, 512))


class TestMaskedLoss(unittest.TestCase):
    """Masked MSE loss must ignore missing keypoints (zero gradient)."""

    def test_gradient_zero_for_missing_keypoint(self):
        img_size, hm_size = 512, 128
        # Keypoint 1 present, keypoint 2 missing (None), keypoint 3 present.
        keypoints = [(100.0, 200.0), None, (300.0, 300.0)]
        target, mask = generate_gaussian_heatmap(keypoints, img_size, hm_size, sigma=2.0)

        # Prediction is wrong everywhere (all ones).
        pred = torch.ones_like(target, requires_grad=True)
        criterion = nn.MSELoss(reduction="sum")

        loss = 0.0
        for k in range(len(mask)):
            loss = loss + criterion(
                pred[k : k + 1] * mask[k],
                target[k : k + 1] * mask[k],
            )
        loss.backward()

        # Gradient w.r.t. the missing channel must be exactly zero.
        self.assertAlmostEqual(float(pred.grad[1].abs().sum()), 0.0)
        # Present channels must receive non-zero gradient.
        self.assertGreater(float(pred.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(pred.grad[2].abs().sum()), 0.0)

    def test_differentiable_mls_uses_per_sample_spacing(self):
        keypoints = torch.tensor([
            [[0.0, 0.0], [10.0, 0.0], [5.0, 4.0]],
            [[0.0, 0.0], [10.0, 0.0], [5.0, 4.0]],
        ])
        result = differentiable_mls_mm(keypoints, torch.tensor([0.5, 0.8]))
        torch.testing.assert_close(result, torch.tensor([2.0, 3.2]))

    def test_competition_loss_backpropagates_threshold_signal(self):
        config = MLSHeatmapConfig(
            backbone="hrnet_w18", image_size=256, mls_loss_weight=0.25,
            threshold_loss_weight=0.1,
        )
        prediction = torch.randn(2, 3, 64, 64, requires_grad=True)
        target = torch.zeros_like(prediction)
        masks = torch.ones(2, 3)
        keypoints = torch.tensor([
            [[20.0, 20.0], [220.0, 20.0], [120.0, 26.0]],
            [[20.0, 20.0], [220.0, 20.0], [120.0, 32.0]],
        ])
        total, parts = competition_aware_heatmap_loss(
            prediction, target, masks, keypoints, torch.tensor([0.5, 0.5]),
            config, nn.MSELoss(),
        )
        total.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertGreater(float(parts["threshold"]), 0.0)

    def test_peak_aware_selector_loss_prefers_study_max_slice(self):
        config = MLSHeatmapConfig(
            image_size=256,
            use_selector=True,
            dataset_variant="multitask_v2",
            selector_target_mode="peak_aware_soft",
            selector_peak_base=0.5,
            spatial_loss_weight=0.0,
            coordinate_loss_weight=0.0,
            mls_loss_weight=0.0,
            threshold_loss_weight=0.0,
        )
        heatmaps = torch.zeros(3, 3, 64, 64, requires_grad=True)
        targets = torch.zeros_like(heatmaps)
        masks = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        # Horizontal falx; outer-point offsets yield 2 mm and 4 mm at 0.5 mm/px.
        keypoints = torch.tensor([
            [[20.0, 20.0], [220.0, 20.0], [120.0, 24.0]],
            [[20.0, 20.0], [220.0, 20.0], [120.0, 28.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ])
        selector_logits = torch.zeros(3, requires_grad=True)
        total, _ = multitask_loss(
            heatmaps, selector_logits, targets, masks, keypoints,
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([1.0, 1.0, 0.0]),
            torch.tensor([4.0, 4.0, 0.1]),
            config,
        )
        total.backward()
        # At zero logits BCE gradient is sigmoid(logit)-target. The peak slice
        # target is 1.0 and therefore receives a more negative gradient.
        self.assertLess(float(selector_logits.grad[1]), float(selector_logits.grad[0]))
        self.assertGreater(float(selector_logits.grad[2]), 0.0)


class TestValidationMetrics(unittest.TestCase):
    """Validation metrics must be computed against TRUE keypoints."""

    def _make_loader(self):
        img_size, hm_size = 512, 128
        keypoints_list = [
            [(100.0, 200.0), (400.0, 300.0), (266.0, 250.0)],
            [(256.0, 100.0), (256.0, 400.0), (266.0, 250.0)],  # vertical falx, 10px offset
        ]
        images = torch.randn(2, 3, img_size, img_size)
        targets, masks, kps_true = [], [], []
        for kps in keypoints_list:
            hm, m = generate_gaussian_heatmap(kps, img_size, hm_size, sigma=2.0)
            targets.append(hm)
            masks.append(m)
            kps_true.append(torch.tensor(kps, dtype=torch.float32))
        batch = (
            images,
            torch.stack(targets),
            torch.stack(masks),
            torch.stack(kps_true),
        )

        class _FakeLoader:
            def __iter__(self):
                yield batch

        return _FakeLoader(), torch.stack(targets)

    def test_perfect_model_near_zero_error(self):
        """A model reproducing the GT heatmaps should score ~0 error."""
        loader, targets = self._make_loader()

        class _IdentityModel(nn.Module):
            def forward(self, x):  # noqa: D102
                return targets

        metrics = _compute_validation_metrics(
            _IdentityModel(), loader, heatmap_size=128, img_size=512,
            spacing_x=0.5, device=torch.device("cpu"),
            criterion=nn.MSELoss(), epoch=1,
        )

        # DARK decode of exact Gaussians is sub-pixel accurate. Keypoints here
        # are integer-aligned so the error is exactly 0.
        self.assertLess(metrics["kp_mae_px"], 0.5)
        self.assertLess(metrics["mls_mae_mm"], 0.5)
        self.assertEqual(metrics["mls_bin_acc"], 1.0)
        self.assertEqual(metrics["n_samples"], 2)
        self.assertIn("kp_mae_px", metrics)

    def test_shifted_model_reports_positive_error(self):
        """A model with offset heatmaps must report positive keypoint error."""
        loader, targets = self._make_loader()

        # Shift ONLY the OutermostPointOfTheFalx heatmap (channel 2) by
        # (0, 4) heatmap px = (0, 16) image px in x. A uniform translation of
        # all three keypoints would leave MLS unchanged (rigid translation is
        # MLS-invariant), so we shift a single keypoint to verify that the MLS
        # metric captures real error too.
        shifted = targets.clone()
        shifted[:, 2] = torch.roll(targets[:, 2], shifts=(0, 4), dims=(1, 2))

        class _ShiftedModel(nn.Module):
            def forward(self, x):  # noqa: D102
                return shifted

        metrics = _compute_validation_metrics(
            _ShiftedModel(), loader, heatmap_size=128, img_size=512,
            spacing_x=0.5, device=torch.device("cpu"),
            criterion=nn.MSELoss(), epoch=1,
        )

        # The metrics must capture the real error (previously hard-coded 0.0).
        self.assertGreater(metrics["kp_mae_px"], 1.0)
        self.assertGreater(metrics["mls_mae_mm"], 0.0)
        self.assertLess(metrics["mls_bin_acc"], 1.0)


class TestRunPipeline(unittest.TestCase):
    """End-to-end numeric chain through _run_pipeline with stub models."""

    def test_fixed_keypoints_produce_expected_mls(self):
        img_size = 512
        heatmap_size = 128

        # Keypoints in 512-px space (real values drawn from the dataset).
        kps = np.array([[281.0, 141.0], [242.0, 428.0], [273.0, 249.0]])
        from src.strategies.mls_heatmap.utils import compute_mls_from_keypoints
        expected = compute_mls_from_keypoints(kps, 0.5)

        class _FixedHeatmapModel(nn.Module):
            def forward(self, x):  # noqa: D102
                B = x.shape[0]
                out = torch.zeros(B, 3, heatmap_size, heatmap_size)
                yy, xx = torch.meshgrid(
                    torch.arange(heatmap_size, dtype=torch.float32),
                    torch.arange(heatmap_size, dtype=torch.float32),
                    indexing="ij",
                )
                for i, (px, py) in enumerate(kps):
                    cx, cy = px / 4.0, py / 4.0  # image px -> heatmap px
                    out[:, i] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
                return out

        cfg = MLSHeatmapConfig(backbone="hrnet_w18", top_k_slices=3, aggregation="max")
        image_hu = np.random.uniform(-100, 100, (img_size, img_size, 5)).astype(np.float32)

        mls = _run_pipeline(
            _FixedHeatmapModel(), image_hu, 0.5,
            cfg, torch.device("cpu"),
        )

        # Sub-millimeter accuracy through the whole chain.
        self.assertLess(abs(mls - expected), 0.3)


if __name__ == "__main__":
    unittest.main()
