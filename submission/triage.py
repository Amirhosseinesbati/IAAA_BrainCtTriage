"""
triage.py — Official competition triage function.

Implements the exact triage_from_intermediates() function from the
IAAA 2026 Brain CT Triage Challenge specification.

This module has NO external dependencies beyond the Python standard library.
It is intentionally kept lightweight so it can be safely included inside
the submission zip without pulling in heavy ML packages.
"""

from typing import Any, Dict, Mapping

# ---------------------------------------------------------------------------
# Required intermediate keys
# ---------------------------------------------------------------------------
TRIAGE_REQUIRED_KEYS = {
    "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
    "fracture_prob", "MLS_mm",
}


def validate_intermediates(intermediates: Mapping[str, Any]) -> Dict[str, float]:
    """Validate and normalize intermediate values for a single series.

    Args:
        intermediates: A mapping containing exactly the 7 required keys.

    Returns:
        A dict with the same keys and float values.

    Raises:
        ValueError: If required keys are missing or extra keys are present.
        TypeError: If any value is not convertible to float.
    """
    missing = TRIAGE_REQUIRED_KEYS - intermediates.keys()
    extra = intermediates.keys() - TRIAGE_REQUIRED_KEYS
    if missing:
        raise ValueError(f"Missing keys in intermediates: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected keys in intermediates: {sorted(extra)}.")

    cleaned: Dict[str, float] = {}
    for key in TRIAGE_REQUIRED_KEYS:
        try:
            cleaned[key] = float(intermediates[key])
        except Exception as exc:
            raise TypeError(
                f"Value for key {key!r} must be convertible to float, "
                f"got type {type(intermediates[key]).__name__}."
            ) from exc
    return cleaned


def triage_from_intermediates(intermediates: Mapping[str, Any]) -> int:
    """Compute triage class from intermediate imaging primitives.

    Args:
        intermediates: Mapping with the 7 required keys
                       (V_EDH, V_SDH, V_IPH, V_SAH, V_IVH,
                        fracture_prob, MLS_mm).

    Returns:
        0 = non-urgent
        1 = urgent
        2 = critical
    """
    vals = validate_intermediates(intermediates)

    # -- Clamp primitives to non-negative -----------------------------------
    V_EDH = max(0.0, vals["V_EDH"])
    V_SDH = max(0.0, vals["V_SDH"])
    V_IPH = max(0.0, vals["V_IPH"])
    V_SAH = max(0.0, vals["V_SAH"])
    V_IVH = max(0.0, vals["V_IVH"])
    MLS_mm = max(0.0, vals["MLS_mm"])
    fracture_prob = float(vals["fracture_prob"])

    total_vol = V_EDH + V_SDH + V_IPH + V_SAH + V_IVH

    # -- Hard-coded thresholds (from competition PDF) -----------------------
    EPS_VOLUME = 0.1
    EPS_MLS = 1.0
    MLS_CRITICAL = 5.0
    MLS_URGENT_LOW = 3.0
    EDH_CRIT = 30.0
    SDH_CRIT = 70.0
    IPH_CRIT = 70.0
    TOTAL_VOL_CRIT = 60.0
    COMBO_MLS = 3.0
    COMBO_VOL = 40.0
    FRAC_VOL_CRIT = 15.0
    FRACTURE_PRESENCE_THRESHOLD = 0.5

    # -- Derived flags ------------------------------------------------------
    has_ich = total_vol >= EPS_VOLUME
    mls_present = MLS_mm >= EPS_MLS
    fracture_present = fracture_prob >= FRACTURE_PRESENCE_THRESHOLD

    # -- Critical triage (2) ------------------------------------------------
    if MLS_mm >= MLS_CRITICAL and (has_ich or fracture_present):
        return 2
    if V_EDH >= EDH_CRIT:
        return 2
    if V_SDH >= SDH_CRIT:
        return 2
    if V_IPH >= IPH_CRIT:
        return 2
    if total_vol >= TOTAL_VOL_CRIT:
        return 2
    if has_ich and MLS_mm >= COMBO_MLS and total_vol >= COMBO_VOL:
        return 2
    if fracture_present and total_vol >= FRAC_VOL_CRIT:
        return 2

    # -- Urgent triage (1) --------------------------------------------------
    if MLS_mm >= MLS_CRITICAL and not (has_ich or fracture_present):
        return 1
    if has_ich:
        return 1
    if MLS_URGENT_LOW <= MLS_mm < MLS_CRITICAL:
        return 1
    if fracture_present and total_vol < FRAC_VOL_CRIT:
        return 1
    if total_vol >= EPS_VOLUME and mls_present:
        return 1

    # -- Non-urgent (0) -----------------------------------------------------
    return 0
