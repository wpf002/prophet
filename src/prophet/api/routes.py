"""API route definitions."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from prophet import __version__

router = APIRouter()


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
async def forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate forecasts for a registered series.

    Phase 6 — not yet implemented. Returns 501.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Phase 6 not started. Forecast endpoint not yet wired to a model registry.",
    )
