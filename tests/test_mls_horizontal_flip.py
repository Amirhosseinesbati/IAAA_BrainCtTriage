"""Unit contract for opt-in MLS left-right reflection."""

import numpy as np

from src.strategies.mls_heatmap.dataset import horizontal_flip_image_and_keypoints
from src.strategies.mls_heatmap.utils import compute_mls_from_keypoints


def test_horizontal_flip_reflects_all_channels_and_preserves_absolute_mls() -> None:
    image = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    keypoints = np.asarray([[0.0, 0.0], [4.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    flipped_image, flipped_keypoints = horizontal_flip_image_and_keypoints(image, keypoints)

    np.testing.assert_array_equal(flipped_image, image[:, ::-1, :])
    np.testing.assert_allclose(flipped_keypoints, [[4.0, 0.0], [0.0, 0.0], [3.0, 2.0]])
    assert compute_mls_from_keypoints(keypoints, spacing_x=0.5) == compute_mls_from_keypoints(
        flipped_keypoints, spacing_x=0.5,
    )
