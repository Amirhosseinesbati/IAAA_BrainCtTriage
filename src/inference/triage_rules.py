"""UI compatibility wrapper around the canonical official triage rule."""

from __future__ import annotations

from typing import Mapping

from src.evaluation.triage import triage_from_intermediates

DISPLAY_LABELS = {0: "Normal", 1: "Level 2", 2: "Level 1"}


def apply_triage_rules(
    ich_volumes: Mapping[str, float],
    has_fracture: bool | float,
    mls_mm: float,
) -> str:
    """Preserve the legacy display API while applying the exact official rule."""
    values = {
        f"V_{name}": float(ich_volumes.get(f"V_{name}", ich_volumes.get(name, 0.0)))
        for name in ("EDH", "SDH", "IPH", "SAH", "IVH")
    }
    values["fracture_prob"] = float(has_fracture)
    values["MLS_mm"] = float(mls_mm)
    return DISPLAY_LABELS[triage_from_intermediates(values)]
