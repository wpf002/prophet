"""Unit tests for the Prophet Python client (mocked HTTP — no network)."""

from __future__ import annotations

import json

import httpx
import pytest
from prophet_client import AdhocForecast, Prophet, ProphetError


def _client(handler) -> Prophet:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://test")
    return Prophet("http://test", client=http)


def test_forecast_adhoc_maps_verdict_and_points() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/forecast/adhoc"
        body = json.loads(request.content)
        assert body["freq"] == "MS"
        return httpx.Response(
            200,
            json={
                "model": "AutoETS",
                "seasonality": 12,
                "horizon": 3,
                "n_obs": 72,
                "beats_naive": True,
                "naive_mase": 1.0,
                "model_mase": 0.4,
                "generated_at": "2026-06-15T00:00:00Z",
                "forecasts": [{"ds": "2026-07-01T00:00:00Z", "y_hat": 4.3, "lo": {"80": 4.1}, "hi": {"80": 4.5}}],
            },
        )

    out = _client(handler).forecast_adhoc(
        series=[{"ds": "2020-01-01", "y": 1.0}, {"ds": "2020-02-01", "y": 2.0}],
        horizon=3,
        freq="MS",
        level=[80],
    )
    assert isinstance(out, AdhocForecast)
    assert out.model == "AutoETS"
    assert out.beats_naive is True
    assert out.forecasts[0].lo == {"80": 4.1}


def test_forecast_adhoc_accepts_tuple_series() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["obs"] = json.loads(request.content)["series"]
        return httpx.Response(
            200,
            json={
                "model": "Naive", "seasonality": 1, "horizon": 2, "n_obs": 10,
                "beats_naive": None, "naive_mase": None, "model_mase": None,
                "generated_at": "2026-06-15T00:00:00Z",
                "forecasts": [{"ds": "2026-07-01T00:00:00Z", "y_hat": 5.0}],
            },
        )

    import datetime as dt

    out = _client(handler).forecast_adhoc(
        series=[(dt.datetime(2020, 1, 1, tzinfo=dt.UTC), 1.0), (dt.datetime(2020, 2, 1, tzinfo=dt.UTC), 2.0)],
        horizon=2,
        freq="MS",
    )
    assert out.beats_naive is None
    assert seen["obs"][0]["y"] == 1.0
    assert "2020-01-01" in seen["obs"][0]["ds"]


def test_error_response_raises_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "Series too short."})

    with pytest.raises(ProphetError) as exc:
        _client(handler).forecast_adhoc(series=[{"ds": "2020-01-01", "y": 1.0}], horizon=9, freq="MS")
    assert exc.value.status == 400
    assert "too short" in str(exc.value)


def test_models_maps_summaries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"default": "macro", "models": [{"name": "macro", "model": "AutoETS", "n_series": 5}]},
        )

    models = _client(handler).models()
    assert models[0].name == "macro"
    assert models[0].n_series == 5
