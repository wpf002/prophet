"""Tests for the production model registry and serving.

Builds a tiny real MLForecast model (synthetic data, conformal intervals) into a
temp dir and exercises load + forecast + error paths — no 28-ticker model needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pytest
from mlforecast.utils import PredictionIntervals

from prophet.data.loaders import load_synthetic
from prophet.models.ml import UNTUNED_COL, _build_mlf
from prophet.serving.registry import forecast_series, get_production_model

HORIZON = 12


def _make_model(base_dir: Path, name: str, *, non_negative: bool = True) -> None:
    panel = load_synthetic(n_series=2, n_obs=300, frequency="1h", seed=42).sort(["unique_id", "ds"])
    mlf = _build_mlf(
        {UNTUNED_COL: lgb.LGBMRegressor(random_state=42, n_jobs=2, verbosity=-1)},
        freq="h",
        lags=[1, 24],
        rolling_windows=(24,),
        seasonal_diff=24,
    )
    mlf.fit(
        panel.to_pandas(),
        prediction_intervals=PredictionIntervals(n_windows=2, h=HORIZON),
    )
    model_dir = base_dir / name
    model_dir.mkdir(parents=True)
    mlf.save(str(model_dir))
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": name,
                "model": UNTUNED_COL,
                "freq": "h",
                "horizon": HORIZON,
                "seasonality": 24,
                "n_series": 2,
                "series": ["series_000", "series_001"],
                "n_obs": panel.height,
                "non_negative": non_negative,
            }
        )
    )


def test_load_and_forecast_with_intervals(tmp_path: Path) -> None:
    _make_model(tmp_path, "demo")
    model = get_production_model("demo", str(tmp_path))
    points = forecast_series(model, "series_000", horizon=6, level=[80, 95])
    assert len(points) == 6
    p = points[0]
    assert p.lo is not None and p.hi is not None
    assert set(p.lo) == {"80", "95"} and set(p.hi) == {"80", "95"}
    # non_negative clip holds, and intervals bracket the point estimate.
    assert p.lo["95"] >= 0.0
    assert p.lo["95"] <= p.y_hat <= p.hi["95"]


def test_no_intervals_when_level_omitted(tmp_path: Path) -> None:
    _make_model(tmp_path, "demo")
    model = get_production_model("demo", str(tmp_path))
    points = forecast_series(model, "series_000", horizon=4)
    assert all(p.lo is None and p.hi is None for p in points)


def test_unknown_series_raises_keyerror(tmp_path: Path) -> None:
    _make_model(tmp_path, "demo")
    model = get_production_model("demo", str(tmp_path))
    with pytest.raises(KeyError):
        forecast_series(model, "nope", horizon=4)


def test_horizon_beyond_calibration_raises(tmp_path: Path) -> None:
    _make_model(tmp_path, "demo")
    model = get_production_model("demo", str(tmp_path))
    with pytest.raises(ValueError, match="exceeds"):
        forecast_series(model, "series_000", horizon=HORIZON + 1)


def test_missing_model_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_production_model("absent", str(tmp_path))
