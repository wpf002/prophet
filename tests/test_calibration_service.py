"""Calibration service + API endpoint tests (Steps 3 & 7).

Log calibration to a temp MLflow store, then read it back through the service
and the FastAPI endpoints.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from statistics import NormalDist

import numpy as np
from fastapi.testclient import TestClient

from prophet.api.main import app
from prophet.calibration.service import (
    calibrate_and_log,
    drifted_calibrations,
    latest_calibration,
    records_from_resolved,
)
from prophet.config import settings

_Z95 = NormalDist().inv_cdf((1 + 95 / 100) / 2)
_BASE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _rows(n, *, true_sigma, stated_sigma, start_i=0, series="UNRATE", seed=1):
    """Resolved-forecast rows (store shape) with a known predictive spread."""
    rng = np.random.default_rng(seed)
    half = _Z95 * stated_sigma
    return [
        {
            "model": "macro",
            "series_id": series,
            "ds": _BASE + dt.timedelta(days=start_i + k),
            "horizon": 6,
            "y_hat": 100.0,
            "y_lo_95": 100.0 - half,
            "y_hi_95": 100.0 + half,
            "actual": 100.0 + rng.normal(0, true_sigma),
        }
        for k in range(n)
    ]


def test_records_from_resolved_maps_rows() -> None:
    recs = records_from_resolved(_rows(3, true_sigma=1.0, stated_sigma=1.0))
    assert recs[0].level == 95
    assert recs[0].series_id == "UNRATE"
    assert recs[0].horizon == 6


def test_calibrate_and_log_then_read_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    recs = records_from_resolved(_rows(400, true_sigma=1.0, stated_sigma=1.0))

    reports = calibrate_and_log("macro", recs)
    assert len(reports) == 1
    rep = reports[0]
    assert rep["series_id"] == "UNRATE" and rep["horizon"] == 6
    assert rep["ece"] < 0.05  # well calibrated
    assert rep["basis"] == "empirical"  # plenty of effective evidence

    # Read back through MLflow.
    latest = latest_calibration("macro")
    assert len(latest) == 1
    assert latest[0]["series_id"] == "UNRATE"
    assert latest[0]["ece"] < 0.05


def test_drift_recorded_and_queryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    # Older half calibrated, newer half overconfident → drift.
    rows = _rows(300, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1) + _rows(
        300, true_sigma=1.0, stated_sigma=1 / 3, start_i=300, seed=2
    )
    calibrate_and_log("macro", records_from_resolved(rows))
    drifted = drifted_calibrations("macro")
    assert len(drifted) == 1
    assert drifted[0]["regressed_from_good"] is True


def test_calibration_endpoint_returns_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    calibrate_and_log("macro", records_from_resolved(_rows(400, true_sigma=1.0, stated_sigma=1.0)))

    client = TestClient(app)
    resp = client.get("/calibration/macro")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "macro"
    assert body["reports"][0]["series_id"] == "UNRATE"
    assert body["reports"][0]["ece"] < 0.05


def test_calibration_endpoint_404_when_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    client = TestClient(app)
    assert client.get("/calibration/__none__").status_code == 404


def test_drift_endpoint_lists_drifted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    rows = _rows(300, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1) + _rows(
        300, true_sigma=1.0, stated_sigma=1 / 3, start_i=300, seed=2
    )
    calibrate_and_log("macro", records_from_resolved(rows))

    client = TestClient(app)
    resp = client.get("/calibration/drift")
    assert resp.status_code == 200
    drifted = resp.json()["drifted"]
    assert len(drifted) == 1
    assert drifted[0]["regressed_from_good"] is True
