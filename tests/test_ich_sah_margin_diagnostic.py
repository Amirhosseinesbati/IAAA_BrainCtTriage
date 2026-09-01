from __future__ import annotations

import numpy as np

from scripts.diagnose_ich_sah_background_margin import (
    diagnostic_interpretation,
    finite_quantiles,
    reachability_summary,
)


def test_margin_reachability_uses_strict_bounded_cap() -> None:
    result = reachability_summary(
        np.asarray([1.0, 4.0, 7.9, 8.0, 12.0]),
        caps=(4.0, 8.0, 12.0),
    )
    assert result["4.0"]["pixels"] == 1
    assert result["8.0"]["pixels"] == 3
    assert result["12.0"]["pixels"] == 4


def test_margin_quantiles_ignore_non_finite_values() -> None:
    result = finite_quantiles(np.asarray([1.0, np.nan, 3.0, np.inf]))
    assert result["q00"] == 1.0
    assert result["q50"] == 2.0
    assert result["q100"] == 3.0


def test_margin_interpretation_prioritizes_support_then_cap() -> None:
    assert diagnostic_interpretation(
        eligible_fraction=0.10,
        reachable_fraction_at_8=0.90,
    ).startswith("support_limited")
    assert diagnostic_interpretation(
        eligible_fraction=0.70,
        reachable_fraction_at_8=0.10,
    ).startswith("cap_limited")
    assert diagnostic_interpretation(
        eligible_fraction=0.70,
        reachable_fraction_at_8=0.80,
    ).startswith("optimization_or_representation_limited")
