"""
src.strategies — Strategy Pattern for ICH (Intracranial Hemorrhage) Segmentation
and MLS (Midline Shift) Estimation.

This package provides a pluggable architecture for multiple ML tasks.
Each strategy is a self-contained module that implements either the
ICHStrategy or MLSStrategy interface and auto-registers with the
corresponding central registry.

Public API
----------
ICH strategies:
- ``get_strategy(name)`` → ICHStrategy instance
- ``list_strategies()`` → list of ICH strategy metadata dicts (for UI)
- ``STRATEGY_NAMES`` → sorted list of registered ICH strategy names

MLS strategies:
- ``get_mls_strategy(name)`` → MLSStrategy instance
- ``list_mls_strategies()`` → list of MLS strategy metadata dicts (for UI)
- ``MLS_STRATEGY_NAMES`` → sorted list of registered MLS strategy names
"""

from src.strategies.registry import StrategyRegistry as _Registry
from src.strategies.mls_registry import MLSStrategyRegistry as _MLSRegistry

# Trigger auto-registration for all built-in ICH strategies.
# Each sub-package's __init__.py calls _Registry.register() at import time.
# We import the *module* (not a name from it) to trigger that side-effect.
from src.strategies import monai as _monai        # noqa: F401

# Trigger auto-registration for MLS strategies.
from src.strategies import mls_heatmap as _mls_heatmap  # noqa: F401


# ═════════════════════════════════════════════════════════════════════════
# ICH Strategy API
# ═════════════════════════════════════════════════════════════════════════

def get_strategy(name: str):
    """Retrieve the active ICH strategy (``monai`` / 3D SegResNet)."""
    return _Registry.get(name)


def list_strategies() -> list[dict]:
    """
    Return metadata for every registered ICH strategy.

    Each dict: ``name``, ``display_name``, ``description``,
    ``config_schema``, ``default_config``.
    """
    return _Registry.list_all()


STRATEGY_NAMES = sorted(
    s["name"] for s in _Registry.list_all()
)


# ═════════════════════════════════════════════════════════════════════════
# MLS Strategy API
# ═════════════════════════════════════════════════════════════════════════

def get_mls_strategy(name: str):
    """Retrieve a registered MLS strategy by name ('mls_heatmap', ...)."""
    return _MLSRegistry.get(name)


def list_mls_strategies() -> list[dict]:
    """
    Return metadata for every registered MLS strategy.

    Each dict: ``name``, ``display_name``, ``description``,
    ``config_schema``, ``default_config``.
    """
    return _MLSRegistry.list_all()


MLS_STRATEGY_NAMES = sorted(
    s["name"] for s in _MLSRegistry.list_all()
)
