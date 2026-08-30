from __future__ import annotations

from pathlib import Path

from scripts.screen_fracture_checkpoints import discover_checkpoints


def test_discover_checkpoints_orders_epochs_then_named_weights(tmp_path: Path) -> None:
    for name in ("last.pt", "epoch15.pt", "best.pt", "epoch0.pt", "notes.pt"):
        (tmp_path / name).touch()

    assert [path.name for path in discover_checkpoints(tmp_path)] == [
        "epoch0.pt",
        "epoch15.pt",
        "best.pt",
        "last.pt",
        "notes.pt",
    ]
