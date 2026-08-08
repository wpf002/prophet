"""API route definitions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from prophet import __version__
from prophet.config import settings
from prophet.serving.registry import ForecastPoint as RegistryPoint
from prophet.serving.registry import (
    forecast_series,
    get_production_model,
    list_production_models,
)

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
    model: str | None = Field(
        default=None,
        description="Which served model to use. Defaults to the configured production model.",
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


class SeriesObservation(BaseModel):
    """One (timestamp, value) point of a caller-supplied series."""

    ds: datetime
    y: float


class AdhocForecastRequest(BaseModel):
    """Bring-your-own-series forecast request (no pre-trained model needed)."""

    series: list[SeriesObservation] = Field(
        ..., min_length=2, description="Regularly-spaced (ds, y) observations."
    )
    horizon: int = Field(..., ge=1, le=720, description="Number of steps to forecast.")
    freq: str = Field(..., description="Pandas offset alias, e.g. 'MS', 'D', 'h'.")
    seasonality: int | None = Field(
        default=None, description="Seasonal period. Inferred from freq when omitted."
    )
    level: list[int] | None = Field(
        default=None, description="Prediction-interval confidence levels (e.g. [80])."
    )


class AdhocForecastResponse(BaseModel):
    """Ad-hoc forecast plus the built-in forecastability verdict."""

    model: str
    seasonality: int
    horizon: int
    n_obs: int
    beats_naive: bool | None = Field(
        default=None, description="Did the chosen model beat naive on the caller's data? "
        "None when the series was too short to validate."
    )
    naive_mase: float | None = None
    model_mase: float | None = None
    generated_at: datetime
    forecasts: list[ForecastPoint]


class ModelSummary(BaseModel):
    """Lightweight description of one served model."""

    name: str
    model: str | None = None
    freq: str | None = None
    horizon: int | None = None
    seasonality: int | None = None
    n_series: int | None = None
    trained_at: str | None = None


class ModelsResponse(BaseModel):
    """List of available served models."""

    default: str
    models: list[ModelSummary]


class CalibrationReport(BaseModel):
    """Calibration of one (model, series, horizon) group against outcomes."""

    model: str | None = None
    series_id: str | None = None
    horizon: int | None = None
    level: int | None = None
    n: int
    coverage: float | None = None  # raw fraction inside the stated interval
    ece: float | None = None  # expected calibration error (0 = perfect)
    brier: float | None = None
    log_score: float | None = None
    basis: str | None = None  # empirical | model_derived | assumed
    evidence_count: int | None = None
    independence_score: float | None = None
    effective_evidence: float | None = None
    reliability: list[tuple[float, float]] = Field(default_factory=list)
    drifted: bool | None = None
    regressed_from_good: bool | None = None
    last_calibrated: str | None = None


class CalibrationResponse(BaseModel):
    """Latest calibration per series/horizon for a model."""

    model: str
    reports: list[CalibrationReport]


class CalibrationDriftResponse(BaseModel):
    """Calibration groups currently flagged as drifted (was calibrated, no longer)."""

    drifted: list[CalibrationReport]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(tz=UTC),
    )


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """List the models this service can forecast with."""
    return ModelsResponse(
        default=settings.production_model,
        models=[ModelSummary(**m) for m in list_production_models()],
    )


@router.get("/calibration/drift", response_model=CalibrationDriftResponse)
async def calibration_drift_route(model: str | None = None) -> CalibrationDriftResponse:
    """Calibration groups currently drifted — a model that was calibrated and
    stopped being so. Optionally filter to one ``model``."""
    from prophet.calibration.service import drifted_calibrations

    return CalibrationDriftResponse(
        drifted=[CalibrationReport(**r) for r in drifted_calibrations(model)]
    )


@router.get("/calibration/{model}", response_model=CalibrationResponse)
async def calibration_route(model: str) -> CalibrationResponse:
    """Latest calibration per series/horizon for a model (reliability, ECE,
    Brier, log score, coverage, structured confidence)."""
    from prophet.calibration.service import latest_calibration

    reports = latest_calibration(model)
    if not reports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No calibration recorded for model '{model}'.",
        )
    return CalibrationResponse(model=model, reports=[CalibrationReport(**r) for r in reports])


@router.post("/forecast/adhoc", response_model=AdhocForecastResponse)
async def forecast_adhoc_route(request: AdhocForecastRequest) -> AdhocForecastResponse:
    """Forecast a caller-supplied series with automatic model selection.

    Stateless and universal: any app can send a raw time series and get a
    forecast plus a forecastability verdict (does the best model beat naive?),
    with no pre-trained model or server-side registration.
    """
    from prophet.serving.adhoc import forecast_adhoc

    pairs = [(o.ds, o.y) for o in request.series]
    try:
        result = forecast_adhoc(
            pairs,
            horizon=request.horizon,
            freq=request.freq,
            seasonality=request.seasonality,
            level=request.level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return AdhocForecastResponse(
        model=result.model,
        seasonality=result.seasonality,
        horizon=request.horizon,
        n_obs=result.n_obs,
        beats_naive=result.beats_naive,
        naive_mase=result.naive_mase,
        model_mase=result.model_mase,
        generated_at=datetime.now(tz=UTC),
        forecasts=[ForecastPoint(ds=p.ds, y_hat=p.y_hat, lo=p.lo, hi=p.hi) for p in result.points],
    )


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, background: BackgroundTasks) -> ForecastResponse:
    """Generate a forecast (with optional prediction intervals) for a series."""
    requested = request.model
    model_key = requested or settings.production_model
    try:
        model = get_production_model(model_key)
    except FileNotFoundError as exc:
        # An explicitly named-but-missing model is a client error (404);
        # a missing default model means the service isn't ready (503).
        if requested is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model '{requested}'. See GET /models.",
            ) from exc
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
