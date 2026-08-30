from __future__ import annotations

import numpy as np

from src.fracture.dataset import _neighbor_channel_bgr


def test_neighbor_channel_storage_yields_expected_model_rgb_order() -> None:
    previous = np.full((2, 3), 11, dtype=np.uint8)
    current = np.full((2, 3), 22, dtype=np.uint8)
    following = np.full((2, 3), 33, dtype=np.uint8)

    stored_bgr = _neighbor_channel_bgr(previous, current, following)
    model_rgb = stored_bgr[..., ::-1]

    np.testing.assert_array_equal(model_rgb[..., 0], previous)
    np.testing.assert_array_equal(model_rgb[..., 1], current)
    np.testing.assert_array_equal(model_rgb[..., 2], following)


def test_neighbor_channel_storage_rejects_shape_mismatch() -> None:
    try:
        _neighbor_channel_bgr(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((3, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        )
    except ValueError as exc:
        assert "identical shapes" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise ValueError")
