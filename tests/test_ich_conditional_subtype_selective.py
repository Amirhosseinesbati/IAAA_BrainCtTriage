from __future__ import annotations

import copy

import torch

from scripts.diagnose_ich_conditional_subtype_selective import (
    conditional_subtype_selective_probe_decision,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_selective_loss_components,
)
from src.strategies.ich_2p5d.segmentation_model import (
    ConditionalSubtypeSelectiveResidualAdapter,
    conditional_subtype_selective_trainable_parameters,
    set_conditional_subtype_selective_training_mode,
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


def test_selective_adapter_is_identity_with_inactive_initial_gate() -> None:
    base = TinySmp().eval()
    model = ConditionalSubtypeSelectiveResidualAdapter(base).eval()
    images = torch.randn((2, 9, 5, 7))
    features = base.encoder(images)
    incumbent_masks = base.segmentation_head(base.decoder(features))
    incumbent_classes = base.classification_head(features[-1])
    outputs = model.forward_components(images)

    assert not outputs["selection_gate_active"].any()
    torch.testing.assert_close(outputs["subtype_logits"], incumbent_masks)
    torch.testing.assert_close(outputs["class_logits"], incumbent_classes)
    assert torch.equal(
        outputs["mask_logits"].argmax(dim=1), incumbent_masks.argmax(dim=1)
    )
    assert sum(
        parameter.numel()
        for parameter in conditional_subtype_selective_trainable_parameters(model)
    ) <= 5_000


def test_selective_adapter_requires_gate_before_relabel() -> None:
    base = TinySmp().eval()
    with torch.no_grad():
        base.segmentation_head.weight.zero_()
        base.segmentation_head.bias.fill_(-2.0)
        base.segmentation_head.bias[0] = 1.0
        base.segmentation_head.weight[2, 0, 0, 0] = 4.0
    model = ConditionalSubtypeSelectiveResidualAdapter(
        base, maximum_logit_residual=8.0
    ).eval()
    with torch.no_grad():
        model.subtype_residual_output.bias[3] = 2.0
    images = torch.zeros((1, 9, 1, 2))
    images[:, 0, 0] = torch.tensor([0.0, 1.0])

    inactive = model(images)[0].argmax(dim=1)
    with torch.no_grad():
        model.selection_gate_output.bias.fill_(2.0)
    active = model(images)[0].argmax(dim=1)

    assert inactive.tolist() == [[[0, 2]]]
    assert active.tolist() == [[[0, 4]]]
    assert torch.equal(inactive > 0, active > 0)


def test_selective_training_mode_uses_soft_gate_and_frozen_incumbent() -> None:
    model = ConditionalSubtypeSelectiveResidualAdapter(TinySmp())
    set_conditional_subtype_selective_training_mode(model)
    assert model.training
    assert not model.incumbent_model.training
    assert model.selective_stem.training


def test_selective_loss_trains_gate_on_incumbent_errors() -> None:
    incumbent = torch.full((1, 6, 1, 3), -3.0)
    incumbent[:, 2] = 3.0
    candidate = incumbent.clone().requires_grad_()
    gate_logits = torch.zeros((1, 1, 1, 3), requires_grad=True)
    masks = torch.tensor([[[5, 2, 2]]])

    components = conditional_subtype_selective_loss_components(
        candidate,
        gate_logits,
        incumbent,
        masks,
        torch.ones(1),
        correction_class_weights=torch.ones(5),
        gate_positive_weight=2.0,
    )
    components["loss"].backward()

    assert int(components["gate_positive_pixel_count"]) == 1
    assert int(components["gate_negative_pixel_count"]) == 2
    assert gate_logits.grad is not None
    assert gate_logits.grad[0, 0, 0, 0] < 0
    assert gate_logits.grad[0, 0, 0, 1] > 0


def test_exp71_decision_requires_gate_and_quality_gates() -> None:
    metrics = {
        "initial": {"changed_hard_mask_pixels": 0},
        "final": {
            "foreground_support_mismatch_pixels": 0,
            "true_sah_predicted_iph_pixels": 100,
            "sah_from_iph_recovery_fraction": 0.10,
            "correct_iph_harm_fraction": 0.003,
            "correct_other_harm_fraction": 0.003,
            "true_background_subtype_change_fraction": 0.003,
            "conditional_accuracy_delta": 0.0,
            "conditional_macro_recall_delta": 0.0,
        },
    }
    gate = {
        "gate_error_precision": 0.10,
        "gate_error_recall": 0.10,
        "gate_coverage": 0.05,
    }
    passing = conditional_subtype_selective_probe_decision(
        metrics, gate, trainable_parameter_count=5_000
    )
    assert passing["gates"]["all_passed"]
    failing = copy.deepcopy(gate)
    failing["gate_error_precision"] -= 1e-6
    rejected = conditional_subtype_selective_probe_decision(
        metrics, failing, trainable_parameter_count=5_000
    )
    assert not rejected["gates"]["all_passed"]
