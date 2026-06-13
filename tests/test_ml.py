"""Smoke tests for the ML model (LightGBM via MLForecast).

Runs on tiny synthetic data with tuning disabled (and a single tuning trial in
the tuned case) so the suite exercises the real MLForecast feature pipeline
without paying a full Optuna search.
"""

from __future__ import annotations

import polars as pl

from prophet.data.loaders import load_synthetic
from prophet.evaluation.cross_validation import split_train_test
from prophet.models.ml import TUNED_COL, UNTUNED_COL, forecast_ml

HORIZON = 24
SEASONALITY = 24


def _train() -> pl.DataFrame:
    df = load_synthetic(n_series=3, n_obs=400, frequency="1h", seed=42)
    train, _ = split_train_test(df, horizon=HORIZON)
    return train


class TestForecastMLUntuned:
    def test_untuned_shape_and_columns(self) -> None:
        forecasts, info = forecast_ml(
            _train(),
            horizon=HORIZON,
            season_length=SEASONALITY,
            freq="h",
            tune=False,
            cv_windows=1,
            n_jobs=2,
        )
        assert UNTUNED_COL in forecasts.columns
        assert TUNED_COL not in forecasts.columns
        assert forecasts.height == 3 * HORIZON
        assert forecasts.schema["ds"] == pl.Datetime("us", "UTC")
        assert info.cv_mase_tuned is None
        assert info.best_params == {}
        # Feature importance covers lag, rolling, and calendar features.
        assert "lag1" in info.feature_importance
        assert any(name.startswith("rolling_") for name in info.feature_importance)
        assert "is_weekend" in info.feature_importance


class TestForecastMLTuned:
    def test_tuned_adds_column_and_params(self) -> None:
        forecasts, info = forecast_ml(
            _train(),
            horizon=HORIZON,
            season_length=SEASONALITY,
            freq="h",
            tune=True,
            n_trials=1,
            cv_windows=1,
            n_jobs=2,
        )
        assert {UNTUNED_COL, TUNED_COL}.issubset(set(forecasts.columns))
        assert info.cv_mase_tuned is not None
        assert info.best_params  # non-empty after tuning
        assert info.n_trials == 1
