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


def test_models_endpoint_lists_default_and_models() -> None:
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert "default" in body
    assert isinstance(body["models"], list)


def test_forecast_404_for_unknown_named_model() -> None:
    """An explicitly named, missing model is a 404 (not 503)."""
    client = TestClient(app)
    response = client.post(
        "/forecast",
        json={"series_id": "test", "horizon": 5, "model": "__no_such_model__"},
    )
    assert response.status_code == 404


def _monthly_series(n: int, fn) -> list[dict]:
    import datetime as dt

    base = dt.datetime(2019, 1, 1, tzinfo=dt.UTC)
    return [{"ds": (base + dt.timedelta(days=30 * i)).isoformat(), "y": fn(i)} for i in range(n)]


def test_adhoc_forecast_seasonal_beats_naive() -> None:
    """A clean seasonal+trend series is forecastable and returns intervals."""
    import math

    client = TestClient(app)
    series = _monthly_series(72, lambda i: 100 + 0.5 * i + 10 * math.sin(2 * math.pi * i / 12))
    response = client.post(
        "/forecast/adhoc",
        json={"series": series, "horizon": 6, "freq": "MS", "level": [80]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecasts"]) == 6
    assert body["beats_naive"] is True
    assert body["forecasts"][0]["lo"]["80"] <= body["forecasts"][0]["y_hat"]
    assert body["n_obs"] == 72


def test_adhoc_forecast_short_series_verdict_unknown() -> None:
    """Too short to cross-validate → still forecasts, verdict is null."""
    client = TestClient(app)
    series = _monthly_series(10, lambda i: 100.0 + i)
    response = client.post(
        "/forecast/adhoc",
        json={"series": series, "horizon": 3, "freq": "MS"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["beats_naive"] is None
    assert len(body["forecasts"]) == 3


def test_adhoc_forecast_rejects_too_short() -> None:
    """Fewer points than horizon+2 is a 400."""
    client = TestClient(app)
    series = _monthly_series(4, lambda i: 100.0 + i)
    response = client.post(
        "/forecast/adhoc",
        json={"series": series, "horizon": 3, "freq": "MS"},
    )
    assert response.status_code == 400


def test_adhoc_forecast_requires_two_points() -> None:
    """Schema enforces at least two observations."""
    client = TestClient(app)
    response = client.post(
        "/forecast/adhoc",
        json={"series": [{"ds": "2020-01-01T00:00:00Z", "y": 1.0}], "horizon": 2, "freq": "MS"},
    )
    assert response.status_code == 422
