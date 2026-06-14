"""Production model registry — load persisted MLForecast models and serve them.

Models are saved by ``scripts/train_production.py`` under
``models/production/<name>/``. ``get_production_model`` loads any of them by name
(cached in memory); ``list_production_models`` advertises what's available;
``forecast_series`` produces point forecasts and conformal prediction intervals
for a single series.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCTION_DIR = Path("models/production")


@dataclass(frozen=True)
class ProductionModel:
    """A loaded production model and its metadata."""

    name: str
    mlf: Any
    metadata: dict[str, Any]

    @property
    def series(self) -> list[str]:
        return list(self.metadata["series"])

    @property
    def horizon(self) -> int:
        return int(self.metadata["horizon"])

    @property
    def model_col(self) -> str:
        return str(self.metadata["model"])


@dataclass(frozen=True)
class ForecastPoint:
    """One forecast step: point estimate plus optional interval bounds by level."""

    ds: datetime
    y_hat: float
    lo: dict[str, float] | None
    hi: dict[str, float] | None


def _load(name: str, base_dir: Path) -> ProductionModel:
    from mlforecast import MLForecast

    path = base_dir / name
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No production model at {path}. Run "
            f"`uv run python scripts/train_production.py --dataset {name}`."
        )
    metadata = json.loads(meta_path.read_text())
    mlf = MLForecast.load(str(path))
    return ProductionModel(name=name, mlf=mlf, metadata=metadata)


@lru_cache(maxsize=8)
def get_production_model(name: str, base_dir: str = str(PRODUCTION_DIR)) -> ProductionModel:
    """Load (and cache in memory) the named production model."""
    return _load(name, Path(base_dir))


def list_production_models(base_dir: Path = PRODUCTION_DIR) -> list[dict[str, Any]]:
    """Summarize every persisted model under ``base_dir`` (no model load).

    Returns one dict per model with the lightweight metadata fields, so the API
    can advertise what it serves without pulling the heavy MLForecast objects.
    """
    if not base_dir.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for meta_path in sorted(base_dir.glob("*/metadata.json")):
        metadata = json.loads(meta_path.read_text())
        summaries.append(
            {
                "name": meta_path.parent.name,
                "model": metadata.get("model"),
                "freq": metadata.get("freq"),
                "horizon": metadata.get("horizon"),
                "seasonality": metadata.get("seasonality"),
                "n_series": metadata.get("n_series"),
                "trained_at": metadata.get("trained_at"),
            }
        )
    return summaries


def forecast_series(
    model: ProductionModel,
    series_id: str,
    horizon: int,
    level: list[int] | None = None,
) -> list[ForecastPoint]:
    """Forecast a single series with optional prediction intervals.

    Raises:
        KeyError: series_id is not one of the model's trained series.
        ValueError: horizon exceeds the model's calibrated horizon.
    """
    if series_id not in set(model.series):
        raise KeyError(series_id)
    if horizon > model.horizon:
        raise ValueError(f"horizon {horizon} exceeds model's calibrated horizon {model.horizon}.")

    levels = level or []
    # MLForecast wants level=None (not []) when no intervals are requested.
    forecast = model.mlf.predict(h=horizon, level=levels or None, ids=[series_id])
    col = model.model_col
    non_negative = bool(model.metadata.get("non_negative", False))

    def _clip(value: float) -> float:
        return max(0.0, value) if non_negative else value

    points: list[ForecastPoint] = []
    for record in forecast.to_dict("records"):
        lo = {str(lv): _clip(float(record[f"{col}-lo-{lv}"])) for lv in levels} or None
        hi = {str(lv): _clip(float(record[f"{col}-hi-{lv}"])) for lv in levels} or None
        points.append(ForecastPoint(ds=record["ds"], y_hat=_clip(float(record[col])), lo=lo, hi=hi))
    return points
