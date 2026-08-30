from __future__ import annotations

from scripts.analyze_fracture_mil_errors import _driver, _outcome


def test_outcome_labels_binary_confusion_cells() -> None:
    assert _outcome(0, 1) == "FP"
    assert _outcome(1, 0) == "FN"
    assert _outcome(1, 1) == "TP"


def test_driver_uses_cdf_disagreement() -> None:
    assert _driver(0.9, 0.2) == "detector_dominant"
    assert _driver(0.2, 0.9) == "mil_dominant"
    assert _driver(0.5, 0.55) == "joint"
