from __future__ import annotations

import numpy as np

from scripts.diagnose_ich_sah_background_margin import (
    diagnostic_interpretation,
    finite_quantiles,
    iph_relabel_interpretation,
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


def test_iph_relabel_interpretation_requires_material_safe_support() -> None:
    assert iph_relabel_interpretation(
        sah_predicted_iph_fraction=0.10,
        missed_sah_reachable_at_12=0.90,
        correct_iph_vulnerable_at_12=0.10,
    ).startswith("iph_support_adds_too_few")
    assert iph_relabel_interpretation(
        sah_predicted_iph_fraction=0.40,
        missed_sah_reachable_at_12=0.10,
        correct_iph_vulnerable_at_12=0.10,
    ).startswith("iph_to_sah_margins")
    assert iph_relabel_interpretation(
        sah_predicted_iph_fraction=0.40,
        missed_sah_reachable_at_12=0.80,
        correct_iph_vulnerable_at_12=0.80,
    ).startswith("iph_support_has_high")
    assert iph_relabel_interpretation(
        sah_predicted_iph_fraction=0.40,
        missed_sah_reachable_at_12=0.80,
        correct_iph_vulnerable_at_12=0.20,
    ).startswith("iph_support_is_material")
