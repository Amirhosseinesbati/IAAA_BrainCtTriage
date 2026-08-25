"""Canonical, configuration-backed implementation of the official rule."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.config import TRIAGE_REQUIRED_KEYS, TRIAGE_THRESHOLDS

VOLUME_KEYS = ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH")


def validate_intermediates(values: Mapping[str, Any]) -> dict[str, float]:
    missing = TRIAGE_REQUIRED_KEYS - values.keys()
    extra = values.keys() - TRIAGE_REQUIRED_KEYS
    if missing:
        raise ValueError(f"Missing intermediate keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected intermediate keys: {sorted(extra)}")
    cleaned: dict[str, float] = {}
    for key in TRIAGE_REQUIRED_KEYS:
        value = float(values[key])
        if not math.isfinite(value):
            raise ValueError(f"Intermediate {key!r} must be finite, got {value}")
        cleaned[key] = value
    return cleaned


def triage_rule_trace(values: Mapping[str, Any]) -> tuple[int, str]:
    v = validate_intermediates(values)
    t = TRIAGE_THRESHOLDS
    volumes = {key: max(0.0, v[key]) for key in VOLUME_KEYS}
    total = sum(volumes.values())
    mls = max(0.0, v["MLS_mm"])
    fracture = v["fracture_prob"] >= t["FRACTURE_PRESENCE_THRESHOLD"]
    has_ich = total >= t["EPS_VOLUME"]

    if mls >= t["MLS_CRITICAL"] and (has_ich or fracture):
        return 2, "critical_mls_with_pathology"
    if volumes["V_EDH"] >= t["EDH_CRIT"]:
        return 2, "critical_edh_volume"
    if volumes["V_SDH"] >= t["SDH_CRIT"]:
        return 2, "critical_sdh_volume"
    if volumes["V_IPH"] >= t["IPH_CRIT"]:
        return 2, "critical_iph_volume"
    if total >= t["TOTAL_VOL_CRIT"]:
        return 2, "critical_total_volume"
    if has_ich and mls >= t["COMBO_MLS"] and total >= t["COMBO_VOL"]:
        return 2, "critical_mls_volume_combo"
    if fracture and total >= t["FRAC_VOL_CRIT"]:
        return 2, "critical_fracture_volume_combo"
    if mls >= t["MLS_CRITICAL"] and not (has_ich or fracture):
        return 1, "urgent_isolated_high_mls"
    if has_ich:
        return 1, "urgent_any_ich"
    if t["MLS_URGENT_LOW"] <= mls < t["MLS_CRITICAL"]:
        return 1, "urgent_moderate_mls"
    if fracture and total < t["FRAC_VOL_CRIT"]:
        return 1, "urgent_fracture"
    if total >= t["EPS_VOLUME"] and mls >= t["EPS_MLS"]:
        return 1, "urgent_volume_mls"
    return 0, "non_urgent"


def triage_from_intermediates(values: Mapping[str, Any]) -> int:
    return triage_rule_trace(values)[0]


def decision_margins(values: Mapping[str, Any]) -> dict[str, float]:
    """Signed distances to every clinically relevant decision boundary."""
    v = validate_intermediates(values)
    t = TRIAGE_THRESHOLDS
    total = sum(max(0.0, v[key]) for key in VOLUME_KEYS)
    return {
        "total_to_any_ich": total - t["EPS_VOLUME"],
        "mls_to_present": max(0.0, v["MLS_mm"]) - t["EPS_MLS"],
        "mls_to_urgent": max(0.0, v["MLS_mm"]) - t["MLS_URGENT_LOW"],
        "mls_to_critical": max(0.0, v["MLS_mm"]) - t["MLS_CRITICAL"],
        "edh_to_critical": max(0.0, v["V_EDH"]) - t["EDH_CRIT"],
        "sdh_to_critical": max(0.0, v["V_SDH"]) - t["SDH_CRIT"],
        "iph_to_critical": max(0.0, v["V_IPH"]) - t["IPH_CRIT"],
        "total_to_combo": total - t["COMBO_VOL"],
        "total_to_critical": total - t["TOTAL_VOL_CRIT"],
        "fracture_to_present": v["fracture_prob"] - t["FRACTURE_PRESENCE_THRESHOLD"],
        "total_to_fracture_combo": total - t["FRAC_VOL_CRIT"],
    }
