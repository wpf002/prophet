"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from prophet.api.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


def test_forecast_returns_501_until_phase_6() -> None:
    client = TestClient(app)
    response = client.post(
        "/forecast",
        json={"series_id": "test", "horizon": 24},
    )
    assert response.status_code == 501


def test_forecast_validates_horizon_bounds() -> None:
    client = TestClient(app)
    response = client.post(
        "/forecast",
        json={"series_id": "test", "horizon": 0},
    )
    assert response.status_code == 422
