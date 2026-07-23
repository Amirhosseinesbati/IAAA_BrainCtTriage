"""
registry.py — Lightweight strategy registry with auto-discovery.

Strategies register themselves when their module is imported.
The `src/strategies/__init__.py` triggers imports of all known
strategy packages so they self-register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.base import ICHStrategy


class StrategyRegistry:
    """
    Central registry that maps strategy `name` → `ICHStrategy` instance.

    Usage::

        # Registration (called during import):
        StrategyRegistry.register(MyStrategy())

        # Lookup:
        strategy = StrategyRegistry.get("my_strategy")
        strategy.prepare_data()
        strategy.train(config)

        # Enumeration (for UI):
        for info in StrategyRegistry.list_all():
            print(info["name"], info["display_name"])
    """

    _strategies: dict[str, "ICHStrategy"] = {}

    @classmethod
    def register(cls, strategy: "ICHStrategy") -> None:
        """
        Register a strategy instance.

        If a strategy with the same name already exists it is
        overwritten (last import wins), and a warning is printed.
        """
        if strategy.name in cls._strategies:
            existing = cls._strategies[strategy.name]
            print(f"⚠️  StrategyRegistry: '{strategy.name}' overwriting "
                  f"{existing.__class__.__name__} → {strategy.__class__.__name__}")
        cls._strategies[strategy.name] = strategy

    @classmethod
    def get(cls, name: str) -> "ICHStrategy":
        """
        Retrieve a registered strategy by name.

        Raises:
            KeyError: if the strategy name is not registered.
        """
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise KeyError(
                f"Strategy '{name}' not found in registry. "
                f"Available: [{available}]"
            )
        return cls._strategies[name]

    @classmethod
    def list_all(cls) -> list[dict]:
        """
        Return metadata for every registered strategy.

        Each dict contains keys suitable for building a Streamlit
        selector: ``name``, ``display_name``, ``description``,
        ``config_schema``, ``default_config``.
        """
        result: list[dict] = []
        for name, strategy in cls._strategies.items():
            result.append({
                "name": strategy.name,
                "display_name": strategy.display_name,
                "description": strategy.description,
                "config_schema": strategy.get_config_schema(),
                "default_config": strategy.get_default_config(),
            })
        # Sort for deterministic UI ordering
        result.sort(key=lambda s: s["name"])
        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether a strategy name is registered."""
        return name in cls._strategies

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies (mainly for testing)."""
        cls._strategies.clear()

    @classmethod
    def count(cls) -> int:
        """Return the number of registered strategies."""
        return len(cls._strategies)


# ── Convenience module-level aliases ──────────────────────────────
register = StrategyRegistry.register
get = StrategyRegistry.get
list_all = StrategyRegistry.list_all
