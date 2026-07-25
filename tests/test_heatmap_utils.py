"""
test_heatmap_utils.py — Unit tests for heatmap generation, DARK decoding, and MLS computation.

Critical test: Round-trip test (keypoint → heatmap → DARK decode → verify error < 0.5px).
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.strategies.mls_heatmap.utils import (
    generate_gaussian_heatmap,
    decode_heatmap_dark,
    decode_soft_argmax,
    compute_mls_from_keypoints,
    compute_mls_batch,
    compute_mls_binning_accuracy,
    assign_mls_bin,
)


class TestGaussianHeatmap(unittest.TestCase):
    """Tests for Gaussian heatmap generation."""

    def setUp(self):
        self.img_size = 512
        self.heatmap_size = 128
        self.sigma = 2.0

    def test_all_keypoints_present(self):
        """All 3 keypoints present → 3 heatmap channels + mask all 1."""
        keypoints = [(100.0, 200.0), (400.0, 300.0), (256.0, 256.0)]
        heatmaps, mask = generate_gaussian_heatmap(
            keypoints, self.img_size, self.heatmap_size, self.sigma
        )
        self.assertEqual(heatmaps.shape, (3, 128, 128))
        self.assertEqual(mask.shape, (3,))
        self.assertTrue(torch.all(mask == 1.0))
        # Each heatmap should have a peak close to 1.0
        for i in range(3):
            self.assertAlmostEqual(heatmaps[i].max().item(), 1.0, places=4)

    def test_missing_keypoint(self):
        """One keypoint is None → heatmap all zeros + mask 0 for that channel."""
        keypoints = [(100.0, 200.0), None, (256.0, 256.0)]
        heatmaps, mask = generate_gaussian_heatmap(
            keypoints, self.img_size, self.heatmap_size, self.sigma
        )
        self.assertEqual(mask[0].item(), 1.0)
        self.assertEqual(mask[1].item(), 0.0)
        self.assertEqual(mask[2].item(), 1.0)
        # Channel 1 should be all zeros
        self.assertAlmostEqual(heatmaps[1].sum().item(), 0.0)
        # Channels 0 and 2 should have content
        self.assertGreater(heatmaps[0].sum().item(), 0.0)
        self.assertGreater(heatmaps[2].sum().item(), 0.0)

    def test_all_missing(self):
        """All keypoints None → all heatmaps zero, all masks zero."""
        keypoints = [None, None, None]
        heatmaps, mask = generate_gaussian_heatmap(
            keypoints, self.img_size, self.heatmap_size, self.sigma
        )
        self.assertTrue(torch.all(mask == 0.0))
        self.assertAlmostEqual(heatmaps.sum().item(), 0.0)


class TestDARKDecoding(unittest.TestCase):
    """Tests for DARK sub-pixel decoding — the most critical component."""

    def setUp(self):
        self.img_size = 512
        self.heatmap_size = 128
        self.sigma = 2.0

    def _place_and_decode(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Place a keypoint, generate heatmap, decode with DARK, return error."""
        keypoints = [(x_px, y_px), (x_px + 100, y_px + 50), (x_px - 50, y_px - 30)]
        # Only decode the first keypoint for this test
        kp = keypoints[0]
        heatmaps, _ = generate_gaussian_heatmap(
            [kp], self.img_size, self.heatmap_size, self.sigma
        )
        x_dec, y_dec = decode_heatmap_dark(heatmaps[0], self.heatmap_size, self.img_size)
        return x_dec, y_dec

    def test_integer_coordinates(self):
        """Keypoint at exact integer pixel should decode with < 0.1px error."""
        x_dec, y_dec = self._place_and_decode(100.0, 200.0)
        error = np.sqrt((x_dec - 100.0) ** 2 + (y_dec - 200.0) ** 2)
        self.assertLess(error, 0.1, f"DARK error too high: {error:.4f}px")

    def test_subpixel_coordinates(self):
        """Keypoint at sub-pixel position (100.3, 200.7) should decode with < 0.5px error."""
        x_dec, y_dec = self._place_and_decode(100.3, 200.7)
        error = np.sqrt((x_dec - 100.3) ** 2 + (y_dec - 200.7) ** 2)
        self.assertLess(error, 0.5, f"DARK sub-pixel error too high: {error:.4f}px")

    def test_edge_of_image(self):
        """Keypoint near image edge should still decode without crash."""
        x_dec, y_dec = self._place_and_decode(5.0, 500.0)
        # Should return some reasonable value, not crash
        self.assertGreaterEqual(x_dec, 0)
        self.assertGreaterEqual(y_dec, 0)

    def test_empty_heatmap(self):
        """All-zero heatmap → DARK returns (-1, -1)."""
        heatmap = torch.zeros((128, 128))
        x, y = decode_heatmap_dark(heatmap, 128, 512)
        self.assertEqual(x, -1.0)
        self.assertEqual(y, -1.0)

    def test_round_trip_accuracy(self):
        """
        Round-trip test: place known keypoint → generate heatmap →
        DARK decode → verify error < 0.5px for all 3 keypoints.
        """
        # Test multiple positions across the image
        test_positions = [
            (100.0, 200.0),
            (400.0, 300.0),
            (256.0, 256.0),
            (50.5, 50.5),
            (400.8, 100.3),
            (200.2, 400.7),
            (300.0, 150.0),
            (150.0, 350.0),
        ]

        max_error = 0.0
        for x_true, y_true in test_positions:
            x_dec, y_dec = self._place_and_decode(x_true, y_true)
            error = np.sqrt((x_dec - x_true) ** 2 + (y_dec - y_true) ** 2)
            max_error = max(max_error, error)
            self.assertLess(
                error, 0.5,
                f"DARK error {error:.4f}px at ({x_true}, {y_true}) "
                f"→ decoded ({x_dec:.2f}, {y_dec:.2f})"
            )

        print(f"DARK round-trip: max error = {max_error:.4f}px (threshold: 0.5px)")


class TestSoftArgmax(unittest.TestCase):
    """Tests for soft-argmax decoding (fallback method)."""

    def setUp(self):
        self.img_size = 512
        self.heatmap_size = 128
        self.sigma = 2.0

    def test_soft_argmax_center(self):
        """Soft-argmax at center should be reasonably close."""
        x_true, y_true = 256.0, 256.0
        heatmaps, _ = generate_gaussian_heatmap(
            [(x_true, y_true)], self.img_size, self.heatmap_size, self.sigma
        )
        x_dec, y_dec = decode_soft_argmax(heatmaps[0], self.heatmap_size, self.img_size)
        error = np.sqrt((x_dec - x_true) ** 2 + (y_dec - y_true) ** 2)
        # Soft-argmax is less accurate than DARK, so use larger threshold
        self.assertLess(error, 3.0, f"Soft-argmax error too high: {error:.4f}px")


class TestMLSComputation(unittest.TestCase):
    """Tests for MLS computation from keypoint coordinates."""

    def test_known_mls_value(self):
        """
        Test with a known geometry:
        Points 1 (100, 200) and 2 (400, 300) form a falx line.
        Point 3 (250, 250) lies on the line → MLS should be 0.
        """
        keypoints = np.array([
            [100.0, 200.0],  # Anterior
            [400.0, 300.0],  # Posterior
            [250.0, 250.0],  # Outermost — on the line
        ], dtype=np.float32)
        spacing_x = 0.5  # mm/px
        mls = compute_mls_from_keypoints(keypoints, spacing_x)
        self.assertAlmostEqual(mls, 0.0, places=4)

    def test_known_offset(self):
        """
        Vertical line at x=256, point 3 offset by 10px to x=266.
        MLS should be 10px * spacing_x mm.
        """
        keypoints = np.array([
            [256.0, 100.0],  # Anterior
            [256.0, 400.0],  # Posterior
            [266.0, 250.0],  # Outermost — 10px to the right
        ], dtype=np.float32)
        spacing_x = 0.5
        mls = compute_mls_from_keypoints(keypoints, spacing_x)
        self.assertAlmostEqual(mls, 5.0, places=4)  # 10px * 0.5 = 5mm

    def test_degenerate_falx_line(self):
        """Both attachment points coincident → MLS should be 0."""
        keypoints = np.array([
            [100.0, 100.0],
            [100.0, 100.0],
            [200.0, 200.0],
        ], dtype=np.float32)
        mls = compute_mls_from_keypoints(keypoints, 0.5)
        self.assertEqual(mls, 0.0)

    def test_batch_mls(self):
        """Batch MLS computation should produce same results as individual."""
        keypoints_batch = np.array([
            [[256.0, 100.0], [256.0, 400.0], [266.0, 250.0]],
            [[100.0, 200.0], [400.0, 300.0], [250.0, 250.0]],
        ], dtype=np.float32)
        mls_batch = compute_mls_batch(keypoints_batch, 0.5)
        self.assertEqual(len(mls_batch), 2)
        self.assertAlmostEqual(mls_batch[0], 5.0, places=4)
        self.assertAlmostEqual(mls_batch[1], 0.0, places=4)


class TestMLSBinning(unittest.TestCase):
    """Tests for triage-relevant MLS binning."""

    def test_assign_bins(self):
        self.assertEqual(assign_mls_bin(0.5), 0)   # < 1mm
        self.assertEqual(assign_mls_bin(1.0), 1)   # 1-3mm
        self.assertEqual(assign_mls_bin(2.5), 1)   # 1-3mm
        self.assertEqual(assign_mls_bin(3.0), 2)   # 3-5mm
        self.assertEqual(assign_mls_bin(4.9), 2)   # 3-5mm
        self.assertEqual(assign_mls_bin(5.0), 3)   # >= 5mm
        self.assertEqual(assign_mls_bin(10.0), 3)  # >= 5mm

    def test_binning_accuracy_perfect(self):
        true = np.array([0.5, 2.0, 4.0, 6.0])
        pred = np.array([0.5, 2.0, 4.0, 6.0])
        acc = compute_mls_binning_accuracy(true, pred)
        self.assertEqual(acc, 1.0)

    def test_binning_accuracy_partial(self):
        true = np.array([0.5, 2.0, 4.0, 6.0])
        pred = np.array([0.5, 2.0, 6.0, 4.0])  # last two swapped bins
        acc = compute_mls_binning_accuracy(true, pred)
        self.assertEqual(acc, 0.5)  # 2/4 correct


class TestDARKvsSoftArgmax(unittest.TestCase):
    """
    DARK should be significantly more accurate than soft-argmax
    for sub-pixel positions.
    """

    def setUp(self):
        self.img_size = 512
        self.heatmap_size = 128
        self.sigma = 2.0

    def test_dark_more_accurate(self):
        """DARK should have lower error than soft-argmax on sub-pixel positions."""
        positions = [(100.3, 200.7), (400.8, 100.2), (50.5, 50.5)]

        dark_errors = []
        soft_errors = []

        for x_true, y_true in positions:
            heatmaps, _ = generate_gaussian_heatmap(
                [(x_true, y_true)], self.img_size, self.heatmap_size, self.sigma
            )
            hm = heatmaps[0]

            x_dark, y_dark = decode_heatmap_dark(hm, self.heatmap_size, self.img_size)
            x_soft, y_soft = decode_soft_argmax(hm, self.heatmap_size, self.img_size)

            dark_err = np.sqrt((x_dark - x_true) ** 2 + (y_dark - y_true) ** 2)
            soft_err = np.sqrt((x_soft - x_true) ** 2 + (y_soft - y_true) ** 2)

            dark_errors.append(dark_err)
            soft_errors.append(soft_err)

        avg_dark = float(np.mean(dark_errors))
        avg_soft = float(np.mean(soft_errors))

        print(f"DARK avg error: {avg_dark:.4f}px | Soft-argmax avg error: {avg_soft:.4f}px")
        self.assertLess(avg_dark, avg_soft,
                        "DARK should be more accurate than soft-argmax")


if __name__ == "__main__":
    unittest.main()
