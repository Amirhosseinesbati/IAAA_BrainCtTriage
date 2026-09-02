from __future__ import annotations

import copy

import torch

from scripts.diagnose_ich_conditional_subtype_residual import (
    conditional_subtype_residual_probe_decision,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_population_loss_components,
)
from src.strategies.ich_2p5d.segmentation_model import (
    ConditionalSubtypeResidualAdapter,
    conditional_subtype_residual_trainable_parameters,
    set_conditional_subtype_residual_training_mode,
)


class TinySmp(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return [images, images[:, :1]]

    class Decoder(torch.nn.Module):
        def forward(self, features):
            return features[-1]

    def __init__(self) -> None:
        super().__init__()
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.segmentation_head = torch.nn.Conv2d(1, 6, kernel_size=1)
        self.classification_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(1, 6),
        )


def test_residual_adapter_is_identity_and_preserves_support() -> None:
    base = TinySmp().eval()
    adapter = ConditionalSubtypeResidualAdapter(base).eval()
    images = torch.randn((2, 9, 5, 7))
    features = base.encoder(images)
    incumbent_masks = base.segmentation_head(base.decoder(features))
    incumbent_classes = base.classification_head(features[-1])

    outputs = adapter.forward_components(images)

    torch.testing.assert_close(outputs["subtype_logits"], incumbent_masks)
    torch.testing.assert_close(outputs["class_logits"], incumbent_classes)
    assert torch.equal(
        outputs["mask_logits"].argmax(dim=1), incumbent_masks.argmax(dim=1)
    )
    assert sum(
        parameter.numel()
        for parameter in conditional_subtype_residual_trainable_parameters(adapter)
    ) <= 5_000
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_residual_adapter_can_relabel_subtype_but_not_foreground() -> None:
    base = TinySmp().eval()
    with torch.no_grad():
        base.segmentation_head.weight.zero_()
        base.segmentation_head.bias.fill_(-2.0)
        base.segmentation_head.bias[0] = 1.0
        base.segmentation_head.weight[2, 0, 0, 0] = 4.0
    adapter = ConditionalSubtypeResidualAdapter(base, maximum_logit_residual=8.0)
    with torch.no_grad():
        adapter.subtype_residual_head[-1].bias[3] = 2.0
    images = torch.zeros((1, 9, 1, 2))
    images[:, 0, 0] = torch.tensor([0.0, 1.0])
    features = base.encoder(images)
    incumbent = base.segmentation_head(base.decoder(features)).argmax(dim=1)
    candidate = adapter(images)[0].argmax(dim=1)

    assert incumbent.tolist() == [[[0, 2]]]
    assert candidate.tolist() == [[[0, 4]]]
    assert torch.equal(incumbent > 0, candidate > 0)


def test_residual_training_mode_keeps_incumbent_eval() -> None:
    adapter = ConditionalSubtypeResidualAdapter(TinySmp())
    set_conditional_subtype_residual_training_mode(adapter)
    assert not adapter.incumbent_model.training
    assert adapter.subtype_residual_head.training


def test_population_loss_uses_shared_foreground_denominator() -> None:
    incumbent = torch.full((1, 6, 1, 4), -3.0)
    incumbent[:, 2] = 3.0
    candidate = incumbent.clone()
    candidate[:, 5, 0, 0] = 0.0
    candidate.requires_grad_()
    masks = torch.tensor([[[5, 2, 2, 2]]])

    components = conditional_subtype_population_loss_components(
        candidate,
        incumbent,
        masks,
        torch.ones(1),
        correction_class_weights=torch.ones(5),
        correction_weight=1.0,
        stability_weight=1.0,
    )
    components["loss"].backward()

    assert int(components["correction_pixel_count"]) == 1
    assert int(components["stability_pixel_count"]) == 3
    assert int(components["population_pixel_count"]) == 4
    torch.testing.assert_close(
        components["correction_population"], components["correction"] / 4
    )
    assert candidate.grad is not None
    assert candidate.grad[:, :, 0, 0].abs().sum() > 0


def test_population_loss_has_zero_identity_drift_without_errors() -> None:
    incumbent = torch.randn((1, 6, 2, 2))
    incumbent[:, 0] = -10.0
    target = incumbent.argmax(dim=1)
    candidate = incumbent.clone().requires_grad_()
    components = conditional_subtype_population_loss_components(
        candidate,
        incumbent,
        target,
        torch.ones(1),
    )
    components["loss"].backward()
    assert int(components["correction_pixel_count"]) == 0
    assert abs(components["loss"].item()) < 1e-6
    assert candidate.grad is not None
    assert candidate.grad.abs().max().item() < 1e-6


def test_exp70_decision_requires_every_selectivity_gate() -> None:
    metrics = {
        "initial": {"changed_hard_mask_pixels": 0},
        "final": {
            "foreground_support_mismatch_pixels": 0,
            "true_sah_predicted_iph_pixels": 100,
            "sah_from_iph_recovery_fraction": 0.10,
            "correct_iph_harm_fraction": 0.005,
            "correct_other_harm_fraction": 0.005,
            "true_background_subtype_change_fraction": 0.005,
            "conditional_accuracy_delta": 0.0,
            "conditional_macro_recall_delta": 0.0,
        },
    }
    passing = conditional_subtype_residual_probe_decision(
        metrics, trainable_parameter_count=5_000
    )
    assert passing["gates"]["all_passed"]
    failing = copy.deepcopy(metrics)
    failing["final"]["correct_other_harm_fraction"] += 1e-6
    rejected = conditional_subtype_residual_probe_decision(
        failing, trainable_parameter_count=5_000
    )
    assert not rejected["gates"]["all_passed"]
