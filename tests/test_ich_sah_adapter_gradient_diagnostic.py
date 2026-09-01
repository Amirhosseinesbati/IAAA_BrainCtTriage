from __future__ import annotations

import pytest
import torch

from scripts.diagnose_ich_sah_adapter_gradients import (
    _combine_gradients,
    _finite_summary_for_rows,
    gradient_diagnostic_interpretation,
)


@pytest.mark.parametrize(
    ("weighted_ratio", "cosine", "expected"),
    [
        (0.01, -0.20, "underweighted_and_conflicting_sah_signal"),
        (0.01, 0.20, "underweighted_sah_signal"),
        (0.20, -0.30, "meaningful_sah_gradient_but_conflicting_objectives"),
        (
            0.20,
            0.20,
            "sah_gradient_is_material_failure_likely_cap_or_representation",
        ),
    ],
)
def test_gradient_diagnostic_interpretation(
    weighted_ratio: float, cosine: float, expected: str
) -> None:
    assert (
        gradient_diagnostic_interpretation(
            weighted_ratio_median=weighted_ratio,
            cosine_median=cosine,
        )
        == expected
    )


def test_combine_gradients_preserves_unused_parameters() -> None:
    base = (torch.tensor([1.0, 2.0]), None, torch.tensor([3.0]))
    auxiliary = (torch.tensor([2.0, -2.0]), torch.tensor([4.0]), None)

    combined = _combine_gradients(base, auxiliary, weight=0.25)

    torch.testing.assert_close(combined[0], torch.tensor([1.5, 1.5]))
    torch.testing.assert_close(combined[1], torch.tensor([1.0]))
    torch.testing.assert_close(combined[2], torch.tensor([3.0]))


def test_finite_summary_ignores_none_and_nonfinite_values() -> None:
    rows = [
        {"value": None},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 1.0},
        {"value": 3.0},
    ]

    summary = _finite_summary_for_rows(rows, "value")

    assert summary["count"] == 2
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
