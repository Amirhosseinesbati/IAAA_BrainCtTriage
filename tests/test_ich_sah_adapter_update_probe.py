from __future__ import annotations

import pytest
import torch

from scripts.diagnose_ich_sah_adapter_updates import (
    _parameter_delta,
    update_probe_interpretation,
)


@pytest.mark.parametrize(
    ("sah", "background", "saturation", "q99", "expected"),
    [
        (0.20, 0.0002, 0.00, 6.0, "train_fit_has_unsafe_background_pressure"),
        (
            0.10,
            0.0001,
            0.00,
            6.0,
            "adapter_fits_train_selectively_calibration_failure_is_generalization_or_support",
        ),
        (
            0.001,
            0.0001,
            0.20,
            7.9,
            "adapter_saturates_but_cannot_fit_train_cap_or_support_limited",
        ),
        (0.001, 0.0001, 0.00, 3.0, "adapter_under_updates_on_train"),
        (
            0.001,
            0.0001,
            0.00,
            6.0,
            "adapter_moves_but_train_margin_or_support_remains_limiting",
        ),
    ],
)
def test_update_probe_interpretation(
    sah: float,
    background: float,
    saturation: float,
    q99: float,
    expected: str,
) -> None:
    assert (
        update_probe_interpretation(
            sah_conversion_fraction=sah,
            background_conversion_fraction=background,
            sah_saturation_fraction=saturation,
            sah_residual_q99=q99,
            maximum_logit_residual=8.0,
        )
        == expected
    )


def test_parameter_delta_reports_relative_change() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    initial = (parameter.detach().clone(),)
    with torch.no_grad():
        parameter.add_(torch.tensor([0.0, 5.0]))

    summary = _parameter_delta((parameter,), initial)

    assert summary["initial_l2"] == pytest.approx(5.0)
    assert summary["delta_l2"] == pytest.approx(5.0)
    assert summary["relative_delta"] == pytest.approx(1.0)
