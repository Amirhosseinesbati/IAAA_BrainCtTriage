from __future__ import annotations

from pathlib import Path

import pytest

from scripts.wait_for_artifact_then_run import wait_for_artifact


def test_wait_for_artifact_returns_for_completed_file(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}\n", encoding="utf-8")

    wait_for_artifact(
        artifact,
        min_bytes=2,
        poll_seconds=0.001,
        timeout_seconds=0.1,
    )


def test_wait_for_artifact_times_out(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="Timed out waiting for artifact"):
        wait_for_artifact(
            tmp_path / "missing.json",
            min_bytes=1,
            poll_seconds=0.001,
            timeout_seconds=0.002,
        )
