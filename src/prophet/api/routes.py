"""API route definitions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from prophet import __version__
from prophet.config import settings
from prophet.serving.registry import ForecastPoint as RegistryPoint
from prophet.serving.registry import forecast_series, get_production_model

router = APIRouter()
logger = logging.getLogger("prophet")


def _log_forecast(
    model_name: str, series_id: str, horizon: int, points: list[RegistryPoint]
) -> None:
    """Persist served forecasts for later accuracy scoring. Never raises."""
    if not settings.monitor_dsn:
        return
    try:
        from prophet.monitoring.store import forecast_rows, log_forecasts

        log_forecasts(settings.monitor_dsn, forecast_rows(series_id, model_name, horizon, points))
    except Exception:  # logging must never break a forecast response
        logger.warning("forecast logging failed", exc_info=True)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime


class ForecastRequest(BaseModel):
    """Forecast request body."""

    series_id: str = Field(..., description="Unique identifier for the time series.")
    horizon: int = Field(..., ge=1, le=720, description="Number of steps to forecast.")
    level: list[int] | None = Field(
        default=None,
        description="Prediction interval confidence levels (e.g. [80, 95]).",
    )


class ForecastPoint(BaseModel):
    """Single forecast point."""

    ds: datetime
    y_hat: float
    lo: dict[str, float] | None = None
    hi: dict[str, float] | None = None


class ForecastResponse(BaseModel):
    """Forecast response body."""

    series_id: str
    horizon: int
    model: str
    generated_at: datetime
    forecasts: list[ForecastPoint]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(tz=UTC),
    )


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, background: BackgroundTasks) -> ForecastResponse:
    """Generate a forecast (with optional prediction intervals) for a series."""
    try:
        model = get_production_model(settings.production_model)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        points = forecast_series(model, request.series_id, request.horizon, level=request.level)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown series_id '{request.series_id}'. "
            f"This model serves {model.metadata['n_series']} series.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    model_name = f"{model.name}:{model.model_col}"
    background.add_task(_log_forecast, model_name, request.series_id, request.horizon, points)
    return ForecastResponse(
        series_id=request.series_id,
        horizon=request.horizon,
        model=model_name,
        generated_at=datetime.now(tz=UTC),
        forecasts=[ForecastPoint(ds=p.ds, y_hat=p.y_hat, lo=p.lo, hi=p.hi) for p in points],
    )
