"""A6 synthetic tests: execute only on the target CUDA server."""
from pathlib import Path
import unittest

import torch
import yaml

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.geometry_decoding import decode_training_keypoints, local_softargmax_keypoints
from src.strategies.mls_heatmap.train import differentiable_keypoints_from_heatmaps
from src.strategies.mls_heatmap.train_multitask import multitask_loss


class LocalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("These tests require the target CUDA server")

    def test_legacy_exact(self):
        config = MLSHeatmapConfig()
        self.assertEqual(config.training_geometry_decoder, "global_softargmax")
        logits = torch.randn(2, 3, 32, 32, device="cuda")
        self.assertTrue(torch.equal(decode_training_keypoints(logits, config), differentiable_keypoints_from_heatmaps(logits, config.image_size, config.softargmax_temperature)))

    def test_gaussian_scale_and_gradient(self):
        y, x = torch.meshgrid(torch.arange(128, device="cuda"), torch.arange(128, device="cuda"), indexing="ij")
        logits = (-((x - 40.25)**2 + (y - 62.25)**2) / 18).reshape(1, 1, 128, 128).requires_grad_()
        coords = local_softargmax_keypoints(logits, 512, 1, 6)
        torch.testing.assert_close(coords, coords.new_tensor([[[161., 249.]]]), atol=.75, rtol=0)
        coords.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, :, :20].abs().sum()), 0)

    def test_secondary_peak(self):
        logits = torch.full((1, 1, 64, 64), -100., device="cuda")
        logits[0, 0, 10, 10] = 1
        logits[0, 0, 50, 50] = .9
        local = local_softargmax_keypoints(logits, 256, 1, 6)
        glob = differentiable_keypoints_from_heatmaps(logits, 256, 1)
        torch.testing.assert_close(local, local.new_full((1, 1, 2), 40), atol=.001, rtol=0)
        self.assertGreater(float((glob - local).abs().mean()), 50)

    def test_flat_and_border(self):
        for peak in (False, True):
            logits = torch.zeros(1, 1, 16, 16, device="cuda")
            if peak:
                logits[0, 0, 0, 0] = 10
            logits.requires_grad_()
            coords = local_softargmax_keypoints(logits, 64, 1, 6)
            coords.sum().backward()
            self.assertTrue(torch.isfinite(coords).all() and torch.isfinite(logits.grad).all())
            self.assertTrue((coords >= 0).all() and (coords < 64).all())

    def test_manifest_single_factor_and_radius_validation(self):
        root = Path(__file__).resolve().parents[1] / "config/experiments"
        def config(name):
            return MLSHeatmapConfig.model_validate(yaml.safe_load((root / name).read_text())["training_config"]).model_dump()
        baseline = config("mls-vast-deploy-aligned-baseline-template.yaml")
        a6 = config("mls-vast-deploy-aligned-a6-local-geometry-template.yaml")
        self.assertEqual([k for k in baseline if baseline[k] != a6[k]], ["training_geometry_decoder"])
        with self.assertRaises(ValueError):
            MLSHeatmapConfig(local_softargmax_radius=0)

    def test_negative_only_loss(self):
        config = MLSHeatmapConfig(training_geometry_decoder="local_softargmax")
        h = torch.randn(2, 3, 16, 16, device="cuda", requires_grad=True)
        selector = torch.zeros(2, device="cuda", requires_grad=True)
        loss, parts = multitask_loss(h, selector, torch.zeros_like(h), torch.zeros(2, 3, device="cuda"), torch.zeros(2, 3, 2, device="cuda"), torch.ones(2, device="cuda"), torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(selector.grad).all())
        self.assertEqual(float(parts["coordinate"]), 0)


if __name__ == "__main__":
    unittest.main()
