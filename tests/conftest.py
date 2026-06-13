"""Shared pytest fixtures."""

from __future__ import annotations

import os

# LightGBM (libomp) and torch each ship an OpenMP runtime; loading both in one
# process aborts on macOS ("OMP Error #15"), which surfaces as a segfault when
# the LightGBM and neural tests run in the same session. Tolerate the duplicate.
# Must be set before either library is imported, i.e. before test collection.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_tracking(tmp_path, monkeypatch):
    """Redirect MLflow tracking to a tmp dir during tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
