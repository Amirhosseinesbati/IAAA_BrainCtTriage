import unittest
import numpy as np
import torch
from scripts.probe_mls_translation_cuda import translate, aligned_js, mls
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap


class TranslationProbeTest(unittest.TestCase):
    def test_no_wrap_and_identity(self):
        x = torch.arange(36).reshape(1, 1, 6, 6)
        self.assertTrue(torch.equal(translate(x, 0, 0), x))
        moved = translate(x, 2, 1)
        self.assertTrue(torch.equal(moved[..., 1:, 2:], x[..., :5, :4]))
        self.assertEqual(int(moved[..., :1, :].sum()), 0)
        self.assertEqual(int(moved[..., :, :2].sum()), 0)

    def test_coordinate_convention(self):
        points = [(100., 120.), (120., 400.), (140., 260.)]
        a, _ = generate_gaussian_heatmap(points, 512, 128)
        for dx, dy in ((8, 0), (0, 8)):
            b, _ = generate_gaussian_heatmap([(x+dx, y+dy) for x, y in points], 512, 128)
            h, w = a.shape[-2:]
            self.assertTrue(torch.equal(a[..., :h-dy//4, :w-dx//4], b[..., dy//4:, dx//4:]))
            self.assertLess(float(aligned_js(a[None], b[None], dx//4, dy//4).abs().max()), 1e-7)

    def test_mls_translation_invariant(self):
        coords = np.array([[[100., 100.], [120., 400.], [160., 250.]]])
        np.testing.assert_allclose(mls(coords, np.array([.5])), mls(coords+8, np.array([.5])), atol=1e-12)

    def test_bad_shift(self):
        for shift in ((-1, 0), (6, 0), (0, 6), (.5, 0)):
            with self.assertRaises(ValueError):
                translate(torch.zeros(1, 1, 6, 6), *shift)


if __name__ == '__main__':
    unittest.main()
