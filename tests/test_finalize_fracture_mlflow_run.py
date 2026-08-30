from __future__ import annotations

import pytest

from scripts.finalize_fracture_mlflow_run import _strip_tracking_port


def test_strip_tracking_port_updates_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "https://example.invalid:8443/owner/repository.mlflow",
    )

    _strip_tracking_port(8443)

    assert (
        __import__("os").environ["MLFLOW_TRACKING_URI"]
        == "https://example.invalid/owner/repository.mlflow"
    )


def test_strip_tracking_port_rejects_mismatched_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid/mlflow")

    with pytest.raises(RuntimeError, match="does not use port 8443"):
        _strip_tracking_port(8443)
