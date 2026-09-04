import unittest
import pandas as pd
import numpy as np
from scripts.audit_mls_training_boundary_exposure import geometry_mm, summarize_training


def rows():
    return pd.DataFrame([
        dict(patient_id=study, image_name=str(i), is_target=target,
             x1=10., y1=0., x2=10., y2=20., x3=10.+local, y3=10.,
             spacing_x=1., study_mls_mm=official)
        for i, (study, target, local, official) in enumerate([
            ('a', 1, 2., 4.), ('a', 1, 4., 4.), ('b', 1, 5., 5.),
            ('a', 0, 0., 4.), ('b', 0, 0., 5.), ('c', 0, 0., 0.)])])


class ExposureTests(unittest.TestCase):
    def test_geometry_units_and_invariance(self):
        frame = rows().iloc[:3].copy()
        np.testing.assert_allclose(geometry_mm(frame), [2., 4., 5.])
        frame['spacing_x'] = .5
        np.testing.assert_allclose(geometry_mm(frame), [1., 2., 2.5])

    def test_sampler_expectation_and_cross_level_disagreement(self):
        result = summarize_training(rows(), batch_size=5)
        self.assertEqual(result['consumed_draws_per_epoch'], 5)
        self.assertAlmostEqual(result['expected_positive_draws_per_epoch'], 2.5)
        self.assertEqual(result['boundaries']['3.0']['study_positive_but_local_negative_slices'], 1)
        self.assertEqual(result['boundaries']['5.0']['positive_studies_above_threshold_without_annotated_slice_above'], 0)
        self.assertAlmostEqual(sum(x['expected_positive_draws_per_epoch_by_local_mls'] for x in result['strata']), 2.5)

    def test_rejects_inconsistent_truth_and_duplicate_rows(self):
        frame = rows()
        frame.loc[1, 'study_mls_mm'] = 8.
        with self.assertRaises(ValueError): summarize_training(frame)
        with self.assertRaises(ValueError): summarize_training(pd.concat([rows(), rows().iloc[:1]]))

    def test_rejects_degenerate_line(self):
        frame = rows().iloc[:1].copy()
        frame['y2'] = 0.
        with self.assertRaises(ValueError): geometry_mm(frame)


if __name__ == '__main__': unittest.main()
