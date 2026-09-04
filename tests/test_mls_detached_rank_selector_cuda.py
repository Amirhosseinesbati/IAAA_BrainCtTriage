from __future__ import annotations

import unittest

import torch

from src.strategies.mls_heatmap.model import HRNetHeatmapModel


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA; never falls back to CPU")
class DetachedRankSelectorCudaTests(unittest.TestCase):
    def test_auxiliary_selector_path_leaves_geometry_and_backbone_unmodified(self) -> None:
        """The rank-only path must backpropagate only through the selector head."""
        device = torch.device("cuda:0")
        model = HRNetHeatmapModel(
            backbone_name="hrnet_w18",
            in_channels=3,
            num_keypoints=3,
            pretrained=False,
            head_dropout=0.0,
            use_selector=True,
        ).to(device).train()
        first_batch_norm = next(
            module for module in model.backbone.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        )
        mean_before = first_batch_norm.running_mean.detach().clone()
        var_before = first_batch_norm.running_var.detach().clone()
        model.zero_grad(set_to_none=True)

        logits = model.forward_selector_only_detached_backbone(
            torch.randn(2, 3, 128, 128, device=device),
        )
        self.assertEqual(tuple(logits.shape), (2,))
        logits.sum().backward()

        self.assertTrue(any(
            parameter.grad is not None for parameter in model.selector_head.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in model.backbone.parameters()
        ))
        self.assertTrue(all(parameter.grad is None for parameter in model.head.parameters()))
        self.assertTrue(torch.equal(first_batch_norm.running_mean, mean_before))
        self.assertTrue(torch.equal(first_batch_norm.running_var, var_before))
        self.assertTrue(model.backbone.training)


if __name__ == "__main__":
    unittest.main()
