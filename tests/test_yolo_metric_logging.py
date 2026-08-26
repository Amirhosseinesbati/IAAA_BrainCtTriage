from __future__ import annotations

import unittest

import numpy as np

from src.training.train_yolo import _mlflow_safe_yolo_metrics


class TestYOLOMetricLogging(unittest.TestCase):
    def test_ultralytics_metric_names_are_stable_and_mlflow_safe(self) -> None:
        cleaned = _mlflow_safe_yolo_metrics({
            "metrics/precision(B)": 0.589,
            "metrics/recall(B)": np.float64(0.532),
            "metrics/mAP50(B)": 0.541,
            "metrics/mAP50-95(B)": 0.222,
            "fitness": 0.3,
        })
        self.assertEqual(set(cleaned), {
            "box_precision", "box_recall", "box_map50", "box_map50_95", "fitness",
        })

    def test_non_finite_and_non_numeric_values_are_not_logged(self) -> None:
        cleaned = _mlflow_safe_yolo_metrics({
            "valid(metric)": 1.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "text": "not-a-number",
        })
        self.assertEqual(cleaned, {"valid_metric": 1.0})
