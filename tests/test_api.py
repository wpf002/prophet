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


def test_forecast_503_when_model_absent(monkeypatch) -> None:
    """With no production model configured, /forecast reports unavailable."""
    from prophet.config import settings
    from prophet.serving import registry

    monkeypatch.setattr(settings, "production_model", "__no_such_model__")
    registry.get_production_model.cache_clear()

    client = TestClient(app)
    response = client.post("/forecast", json={"series_id": "test", "horizon": 5})
    assert response.status_code == 503


def test_forecast_validates_horizon_bounds() -> None:
    client = TestClient(app)
    response = client.post(
        "/forecast",
        json={"series_id": "test", "horizon": 0},
    )
    assert response.status_code == 422
