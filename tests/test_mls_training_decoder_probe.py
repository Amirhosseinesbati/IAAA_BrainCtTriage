from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.probe_mls_training_decoders_cuda import select_rows, summarize


class TrainingDecoderProbeTests(unittest.TestCase):
    def test_selection_is_order_invariant_and_refuses_heldout_overlap(self):
        frame = pd.DataFrame([
            dict(patient_id=f"train{i // 2}", image_name=f"slice{i}.png",
                 is_target=1, x1=20, y1=20, x2=20, y2=80, x3=25, y3=50)
            for i in range(160)
        ])
        indices, digest = select_rows(frame, {"heldout"}, 512)
        shuffled = frame.sample(frac=1, random_state=123).reset_index(drop=True)
        other_indices, other_digest = select_rows(shuffled, {"heldout"}, 512)
        self.assertEqual(len(indices), 128)
        self.assertEqual(digest, other_digest)
        self.assertEqual(frame.iloc[indices].image_name.tolist(), shuffled.iloc[other_indices].image_name.tolist())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            select_rows(frame, {"train0"}, 512)

    def test_known_physical_displacement_is_aggregated_correctly(self):
        truth = np.array([[[0., 0.], [0., 10.], [2., 5.]]])
        dark = truth.copy()
        dark[0, 2, 0] = 4.
        result = summarize(truth, dark, truth, np.array([0.5]))
        self.assertEqual(result["decoders"]["softargmax"]["slice_mls_mae_mm"], 0.)
        self.assertEqual(result["decoders"]["dark"]["slice_mls_mae_mm"], 1.)
        self.assertEqual(result["decoders"]["dark"]["landmark_mean_error_mm"], [0., 0., 1.])
        self.assertAlmostEqual(result["inter_decoder_coordinate_mean_mm"], 1. / 3.)
        self.assertEqual(result["inter_decoder_mls_mean_absolute_difference_mm"], 1.)
        self.assertNotIn("predictions", result)

    def test_invalid_decoded_coordinates_do_not_produce_metrics(self):
        coords = np.zeros((2, 3, 2))
        dark = coords.copy()
        dark[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Invalid decoded"):
            summarize(coords, dark, coords, np.ones(2))


if __name__ == "__main__":
    unittest.main()
