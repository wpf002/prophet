"""Prophet client — the Python seam every app forecasts against.

Mirrors ``@prophet/client`` (TypeScript): construct once with the service URL,
then call ``forecast`` / ``forecast_adhoc`` / ``models`` / ``health``. App code
never touches the raw HTTP API.

    from prophet_client import Prophet

    prophet = Prophet("https://prophet-api-production.up.railway.app")

    # Managed model (pre-trained on the server):
    result = prophet.forecast(model="macro", series_id="UNRATE", horizon=6, level=[80])

    # Bring your own series — stateless, works with anything:
    out = prophet.forecast_adhoc(
        series=[(ds, y), ...], horizon=12, freq="MS", level=[80]
    )
    if out.beats_naive:
        print(out.model, "beats naive", out.forecasts)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

__all__ = [
    "AdhocForecast",
    "Forecast",
    "ForecastPoint",
    "ModelSummary",
    "Prophet",
    "ProphetError",
]

DEFAULT_TIMEOUT = 30.0


class ProphetError(RuntimeError):
    """Raised when the Prophet service returns an error or an unexpected shape."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ForecastPoint:
    """One forecast step. ``lo``/``hi`` are keyed by confidence level, e.g. {"80": 4.1}."""

    ds: str
    y_hat: float
    lo: dict[str, float] | None = None
    hi: dict[str, float] | None = None


@dataclass(frozen=True)
class Forecast:
    """Result of a managed-model forecast."""

    series_id: str
    horizon: int
    model: str
    generated_at: str
    forecasts: list[ForecastPoint]


@dataclass(frozen=True)
class AdhocForecast:
    """Result of a bring-your-own-series forecast, with the forecastability verdict."""

    model: str
    seasonality: int
    horizon: int
    n_obs: int
    beats_naive: bool | None
    naive_mase: float | None
    model_mase: float | None
    generated_at: str
    forecasts: list[ForecastPoint]


@dataclass(frozen=True)
class ModelSummary:
    """Lightweight description of one served model."""

    name: str
    model: str | None = None
    freq: str | None = None
    horizon: int | None = None
    seasonality: int | None = None
    n_series: int | None = None
    trained_at: str | None = None


def _points(raw: list[dict[str, Any]]) -> list[ForecastPoint]:
    return [ForecastPoint(ds=p["ds"], y_hat=p["y_hat"], lo=p.get("lo"), hi=p.get("hi")) for p in raw]


def _iso(ds: Any) -> str:
    return ds.isoformat() if isinstance(ds, datetime) else str(ds)


class Prophet:
    """Synchronous Prophet client. Thread-safe for concurrent calls."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ProphetError("Prophet base_url is required.")
        headers = {"accept": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    def forecast(
        self,
        *,
        series_id: str,
        horizon: int,
        model: str | None = None,
        level: list[int] | None = None,
    ) -> Forecast:
        """Forecast a series from a pre-trained served model."""
        body = {"series_id": series_id, "horizon": horizon, "model": model, "level": level}
        data = self._post("/forecast", body)
        return Forecast(
            series_id=data["series_id"],
            horizon=data["horizon"],
            model=data["model"],
            generated_at=data["generated_at"],
            forecasts=_points(data["forecasts"]),
        )

    def forecast_adhoc(
        self,
        *,
        series: list[tuple[Any, float]] | list[dict[str, Any]],
        horizon: int,
        freq: str,
        seasonality: int | None = None,
        level: list[int] | None = None,
    ) -> AdhocForecast:
        """Forecast a caller-supplied series with automatic model selection.

        Stateless and universal — no pre-trained model needed. ``series`` is a
        list of ``(ds, y)`` pairs or ``{"ds": ..., "y": ...}`` dicts. The result's
        ``beats_naive`` is the forecastability verdict on your own data.
        """
        obs = [
            o if isinstance(o, dict) else {"ds": _iso(o[0]), "y": o[1]}
            for o in series
        ]
        body = {
            "series": obs,
            "horizon": horizon,
            "freq": freq,
            "seasonality": seasonality,
            "level": level,
        }
        data = self._post("/forecast/adhoc", body)
        return AdhocForecast(
            model=data["model"],
            seasonality=data["seasonality"],
            horizon=data["horizon"],
            n_obs=data["n_obs"],
            beats_naive=data.get("beats_naive"),
            naive_mase=data.get("naive_mase"),
            model_mase=data.get("model_mase"),
            generated_at=data["generated_at"],
            forecasts=_points(data["forecasts"]),
        )

    def models(self) -> list[ModelSummary]:
        """List the models this service can forecast with."""
        data = self._get("/models")
        return [ModelSummary(**m) for m in data["models"]]

    def health(self) -> dict[str, Any]:
        """Liveness check."""
        return self._get("/health")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Prophet:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str) -> dict[str, Any]:
        return self._send("GET", path, None)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._send("POST", path, body)

    def _send(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        try:
            resp = self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise ProphetError(f"Request to {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            detail = _detail(resp)
            raise ProphetError(detail or f"{path} returned {resp.status_code}.", status=resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProphetError(f"Could not parse JSON from {path}.") from exc


def _detail(resp: httpx.Response) -> str | None:
    try:
        data = resp.json()
    except ValueError:
        return None
    detail = data.get("detail") if isinstance(data, dict) else None
    return detail if isinstance(detail, str) else None
