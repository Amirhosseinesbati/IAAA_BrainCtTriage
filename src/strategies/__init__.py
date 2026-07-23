"""
src.strategies — Strategy Pattern for ICH (Intracranial Hemorrhage) Segmentation.

This package provides a pluggable architecture for selecting between
different ML approaches for the hemorrhage segmentation task. Each
strategy is a self-contained module that implements the ICHStrategy
interface and auto-registers with the central registry.

Public API
----------
- ``get_strategy(name)`` → ICHStrategy instance
- ``list_strategies()`` → list of strategy metadata dicts (for UI)
- ``STRATEGY_NAMES`` → sorted list of registered strategy names
"""

from src.strategies.registry import StrategyRegistry as _Registry

# Trigger auto-registration for all built-in strategies.
# Each sub-package calls _Registry.register() at import time.
from src.strategies.nnunet import strategy as _nnunet  # noqa: F401
from src.strategies.smp import strategy as _smp  # noqa: F401
from src.strategies.monai import strategy as _monai  # noqa: F401
from src.strategies.yolo_seg import strategy as _yolo_seg  # noqa: F401


# ── Re-exported convenience functions ────────────────────────────

def get_strategy(name: str):
    """Retrieve a registered strategy by name ('nnunet', 'smp', ...)."""
    return _Registry.get(name)


def list_strategies() -> list[dict]:
    """
    Return metadata for every registered strategy.

    Each dict: ``name``, ``display_name``, ``description``,
    ``config_schema``, ``default_config``.
    """
    return _Registry.list_all()


STRATEGY_NAMES = sorted(
    s["name"] for s in _Registry.list_all()
)
