from __future__ import annotations

import numpy as np

from scripts.evaluate_fracture_mil_deployable_blend import _empirical_cdf


def test_empirical_cdf_uses_training_distribution_and_mid_ranks() -> None:
    training = np.asarray([1.0, 2.0, 2.0, 4.0])
    actual = _empirical_cdf(training, np.asarray([0.0, 2.0, 3.0, 5.0]))
    np.testing.assert_allclose(actual, [0.1, 0.5, 0.7, 0.9])
