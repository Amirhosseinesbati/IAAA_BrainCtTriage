from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.mlops import tracking


class MLflowResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        tracking._MLFLOW_CIRCUIT_OPEN_UNTIL = 0.0

    def test_failed_call_is_queued_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = Path(temp_dir) / "pending.jsonl"
            with patch.dict(os.environ, {
                "IAAA_MLFLOW_PENDING_QUEUE": str(queue),
                "IAAA_MLFLOW_RETRY_ATTEMPTS": "1",
                "IAAA_MLFLOW_CIRCUIT_SECONDS": "0",
            }, clear=False):
                result = tracking.resilient_mlflow_call(
                    "log_metrics",
                    lambda: (_ for _ in ()).throw(ConnectionError("synthetic outage")),
                    payload={"metrics": {"mae": 1.23}, "step": 7},
                    base_delay_seconds=0,
                )

            self.assertIsNone(result)
            event = json.loads(queue.read_text(encoding="utf-8"))
            self.assertEqual(event["operation"], "log_metrics")
            self.assertEqual(event["payload"]["step"], 7)
            self.assertIn("synthetic outage", event["error"])

    def test_successful_call_does_not_create_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = Path(temp_dir) / "pending.jsonl"
            with patch.dict(os.environ, {
                "IAAA_MLFLOW_PENDING_QUEUE": str(queue),
            }, clear=False):
                result = tracking.resilient_mlflow_call(
                    "noop", lambda: "ok", payload={}, attempts=1,
                )
            self.assertEqual(result, "ok")
            self.assertFalse(queue.exists())


if __name__ == "__main__":
    unittest.main()
