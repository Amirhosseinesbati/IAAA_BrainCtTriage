from __future__ import annotations

import unittest

import torch

from src.strategies.loss_config import LossConfig
from src.strategies.losses import FocalLoss, TverskyLoss, build_composite_loss


class TestVolumetricSegmentationLosses(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.randn(2, 6, 4, 5, 6, requires_grad=True)
        # Mirror MONAI's channel-first floating label tensor.
        self.labels = torch.randint(0, 6, (2, 1, 4, 5, 6)).float()

    def test_default_composite_accepts_monai_3d_labels(self) -> None:
        loss = build_composite_loss(LossConfig(), num_classes=6)(self.logits, self.labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.logits.grad)

    def test_custom_multiclass_losses_support_3d(self) -> None:
        target = self.labels.squeeze(1).long()
        for loss_fn in (FocalLoss(), TverskyLoss()):
            with self.subTest(loss=type(loss_fn).__name__):
                loss = loss_fn(self.logits, target)
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(torch.isfinite(loss))
