from __future__ import annotations

import numpy as np

from scripts.build_fracture_oof_hardmine_dataset import _top_context_indices


def test_top_context_indices_are_ordered_unique_and_clamped() -> None:
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    assert _top_context_indices(scores, top_k=2, radius=1) == [0, 1, 2, 3]


def test_top_context_indices_handle_single_slice() -> None:
    assert _top_context_indices(np.asarray([0.5]), top_k=3, radius=2) == [0]
