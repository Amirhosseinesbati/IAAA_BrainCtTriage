from __future__ import annotations

import json
from pathlib import Path

from scripts.train_fracture_detector_v2 import _write_run_identity


def test_write_run_identity_is_recovery_ready(tmp_path: Path) -> None:
    marker = tmp_path / "mlflow_run.json"

    _write_run_identity(
        marker,
        run_id="run-123",
        experiment_id="experiment-17",
        run_name="fracture-fold-2",
        stage="study_evaluation_complete",
        metrics_only_tracking=True,
        defer_model_artifacts=False,
    )

    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "run_id": "run-123",
        "experiment_id": "experiment-17",
        "run_name": "fracture-fold-2",
        "stage": "study_evaluation_complete",
        "metrics_only_tracking": True,
        "defer_model_artifacts": False,
    }
    assert not (tmp_path / ".mlflow_run.json.tmp").exists()
