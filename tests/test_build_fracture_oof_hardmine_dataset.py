from __future__ import annotations

import numpy as np

from scripts.build_fracture_oof_hardmine_dataset import (
    _symlink_directory,
    _top_context_indices,
)


def test_top_context_indices_are_ordered_unique_and_clamped() -> None:
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    assert _top_context_indices(scores, top_k=2, radius=1) == [0, 1, 2, 3]


def test_top_context_indices_handle_single_slice() -> None:
    assert _top_context_indices(np.asarray([0.5]), top_k=3, radius=2) == [0]


def test_symlink_directory_uses_absolute_source_and_verifies_link(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "base" / "images"
    source.mkdir(parents=True)
    destination = tmp_path / "derived" / "images"
    destination.parent.mkdir()
    calls = []

    def fake_symlink(src, dst, *, target_is_directory):
        calls.append((src, dst, target_is_directory))
        dst.mkdir()

    monkeypatch.setattr(
        "scripts.build_fracture_oof_hardmine_dataset.os.symlink", fake_symlink
    )

    _symlink_directory(source, destination)

    assert calls == [(source.resolve(), destination, True)]
