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
    # Warm the production model into memory so the first request is fast. Missing
    # model is non-fatal — /forecast then returns 503 until one is trained.
    import logging

    from prophet.serving.registry import get_production_model

    try:
        model = get_production_model(settings.production_model)
        logging.getLogger("prophet").info(
            "Loaded production model '%s' (%d series).",
            settings.production_model,
            model.metadata["n_series"],
        )
    except FileNotFoundError:
        logging.getLogger("prophet").warning(
            "No production model '%s' on startup; /forecast will return 503.",
            settings.production_model,
        )
    yield
    # Shutdown: nothing to release (model is in-process memory).


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
