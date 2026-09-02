from __future__ import annotations

import copy

import pytest
import torch

from scripts.diagnose_ich_conditional_subtype_refiner import (
    conditional_subtype_probe_decision,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_loss_components,
)
from src.strategies.ich_2p5d.segmentation_model import (
    ConditionalSubtypeRefinementModel,
    conditional_subtype_trainable_parameters,
    set_conditional_subtype_training_mode,
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

    def forward(self, images):
        features = self.encoder(images)
        decoded = self.decoder(features)
        return (
            self.segmentation_head(decoded),
            self.classification_head(features[-1]),
        )


def test_conditional_refiner_preserves_initial_hard_mask_and_classification() -> None:
    base = TinySmp().eval()
    model = ConditionalSubtypeRefinementModel(base).eval()
    images = torch.randn((2, 9, 5, 7))

    incumbent_masks, incumbent_classes = base(images)
    outputs = model.forward_components(images)

    assert torch.equal(
        outputs["mask_logits"].argmax(dim=1), incumbent_masks.argmax(dim=1)
    )
    torch.testing.assert_close(
        outputs["class_logits"], incumbent_classes, rtol=0.0, atol=0.0
    )
    incumbent_background = incumbent_masks.argmax(dim=1) == 0
    torch.testing.assert_close(
        outputs["mask_logits"].permute(0, 2, 3, 1)[incumbent_background],
        incumbent_masks.permute(0, 2, 3, 1)[incumbent_background],
        rtol=0.0,
        atol=0.0,
    )


def test_conditional_refiner_can_relabel_subtype_but_not_foreground_support() -> None:
    base = TinySmp().eval()
    with torch.no_grad():
        base.segmentation_head.weight.zero_()
        base.segmentation_head.bias.fill_(-2.0)
        base.segmentation_head.bias[0] = 1.0
        base.segmentation_head.weight[2, 0, 0, 0] = 4.0
    model = ConditionalSubtypeRefinementModel(base).eval()
    with torch.no_grad():
        model.subtype_segmentation_head.weight.zero_()
        model.subtype_segmentation_head.bias.fill_(-2.0)
        model.subtype_segmentation_head.bias[5] = 4.0
    images = torch.zeros((1, 9, 1, 2))
    images[:, 0, 0] = torch.tensor([0.0, 1.0])

    incumbent = base(images)[0].argmax(dim=1)
    candidate = model(images)[0].argmax(dim=1)

    assert incumbent.tolist() == [[[0, 2]]]
    assert candidate.tolist() == [[[0, 5]]]
    assert torch.equal(incumbent > 0, candidate > 0)


def test_conditional_refiner_exposes_only_decoder_and_segmentation_head() -> None:
    base = TinySmp()
    model = ConditionalSubtypeRefinementModel(base)
    parameters = conditional_subtype_trainable_parameters(model)
    expected = sum(
        parameter.numel()
        for parameter in model.subtype_segmentation_head.parameters()
    )
    assert sum(parameter.numel() for parameter in parameters) == expected
    assert all(
        not parameter.requires_grad
        for parameter in model.incumbent_model.parameters()
    )

    set_conditional_subtype_training_mode(model)
    assert not model.incumbent_model.training
    assert model.subtype_decoder.training
    assert model.subtype_segmentation_head.training


def test_conditional_subtype_loss_has_no_gradient_outside_incumbent_support() -> None:
    subtype_logits = torch.zeros((1, 6, 1, 3), requires_grad=True)
    incumbent_logits = torch.full_like(subtype_logits, -2.0)
    incumbent_logits[:, 0] = 2.0
    incumbent_logits[:, 2, 0, 1] = 4.0
    masks = torch.tensor([[[5, 5, 5]]])

    components = conditional_subtype_loss_components(
        subtype_logits,
        incumbent_logits,
        masks,
        torch.ones(1),
        foreground_class_weights=torch.ones(5),
        stability_weight=0.0,
    )
    components["loss"].backward()

    assert components["supervised"].item() > 0
    assert subtype_logits.grad is not None
    assert subtype_logits.grad[:, :, 0, 0].abs().sum().item() == 0
    assert subtype_logits.grad[:, :, 0, 1].abs().sum().item() > 0
    assert subtype_logits.grad[:, :, 0, 2].abs().sum().item() == 0


def test_conditional_subtype_stability_anchors_unknown_foreground() -> None:
    subtype_logits = torch.zeros((1, 6, 1, 1), requires_grad=True)
    incumbent_logits = torch.full_like(subtype_logits, -2.0)
    incumbent_logits[:, 3] = 3.0
    components = conditional_subtype_loss_components(
        subtype_logits,
        incumbent_logits,
        torch.zeros((1, 1, 1), dtype=torch.long),
        torch.zeros(1),
        stability_weight=0.25,
    )
    components["loss"].backward()

    assert components["supervised"].item() == 0
    assert components["stability"].item() > 0
    assert subtype_logits.grad is not None
    assert subtype_logits.grad[0, 3, 0, 0].item() < 0


def test_conditional_subtype_loss_rejects_invalid_class_weights() -> None:
    with pytest.raises(ValueError, match="five"):
        conditional_subtype_loss_components(
            torch.zeros((1, 6, 1, 1)),
            torch.zeros((1, 6, 1, 1)),
            torch.zeros((1, 1, 1), dtype=torch.long),
            torch.ones(1),
            foreground_class_weights=torch.ones(4),
        )


def test_conditional_subtype_probe_decision_requires_every_gate() -> None:
    metrics = {
        "initial": {"changed_hard_mask_pixels": 0},
        "final": {
            "foreground_support_mismatch_pixels": 0,
            "sah_from_iph_recovery_fraction": 0.20,
            "correct_iph_harm_fraction": 0.01,
            "correct_other_harm_fraction": 0.01,
            "true_background_subtype_change_fraction": 0.02,
            "conditional_accuracy_delta": 0.005,
            "conditional_macro_recall_delta": 0.01,
        },
    }
    passing = conditional_subtype_probe_decision(metrics)
    assert passing["gates"]["all_passed"]

    for key in (
        "sah_from_iph_recovery_fraction",
        "conditional_accuracy_delta",
        "conditional_macro_recall_delta",
    ):
        failing = copy.deepcopy(metrics)
        failing["final"][key] -= 1e-6
        assert not conditional_subtype_probe_decision(failing)["gates"][
            "all_passed"
        ]
