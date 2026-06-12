"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_tracking(tmp_path, monkeypatch):
    """Redirect MLflow tracking to a tmp dir during tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
