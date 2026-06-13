"""Production model registry — load a persisted MLForecast model and serve it.

The API loads one production model (saved by ``scripts/train_production.py``) on
startup and caches it in memory. ``forecast_series`` produces point forecasts and
conformal prediction intervals for a single series.
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


@lru_cache(maxsize=4)
def get_production_model(name: str, base_dir: str = str(PRODUCTION_DIR)) -> ProductionModel:
    """Load (and cache in memory) the named production model."""
    return _load(name, Path(base_dir))


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
