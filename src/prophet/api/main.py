"""Prophet API — FastAPI service exposing forecast endpoints.

Phase 6 deliverable. Currently exposes /health and a stub /forecast endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prophet import __version__
from prophet.api.routes import router
from prophet.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks."""
    # Startup: connect to MLflow, load any production models, etc.
    yield
    # Shutdown: close resources


app = FastAPI(
    title="Prophet API",
    description="Time-series forecasting service.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router)


def main() -> None:
    """Entry point for `uv run prophet-api` (if added as a script)."""
    import uvicorn

    uvicorn.run(
        "prophet.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
