"""Smoke tests for baseline forecasting models.

These run StatsForecast on small synthetic data, so they stay fast while
exercising the real model wrappers. The no-leak test guards the regression
fixed in Phase 1, where the M4 loader fed the held-out test period back in as
training data.
"""

from __future__ import annotations

import polars as pl

from prophet.data.loaders import load_synthetic
from prophet.evaluation.cross_validation import split_train_test
from prophet.models.baselines import forecast_baselines

HORIZON = 24
SEASONALITY = 24
MODEL_COLS = {"Naive", "SeasonalNaive", "HistoricAverage", "RWD"}


def _forecast() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = load_synthetic(n_series=3, n_obs=200, frequency="1h", seed=42)
    train, test = split_train_test(df, horizon=HORIZON)
    forecasts = forecast_baselines(train, horizon=HORIZON, seasonality=SEASONALITY, freq="h")
    return train, test, forecasts


class TestForecastBaselines:
    def test_returns_all_models(self) -> None:
        _, _, forecasts = _forecast()
        assert MODEL_COLS.issubset(set(forecasts.columns))

    def test_horizon_rows_per_series(self) -> None:
        _, _, forecasts = _forecast()
        assert forecasts.height == 3 * HORIZON
        assert forecasts["unique_id"].n_unique() == 3

    def test_ds_is_utc_microsecond(self) -> None:
        _, _, forecasts = _forecast()
        assert forecasts.schema["ds"] == pl.Datetime("us", "UTC")

    def test_no_train_test_leak(self) -> None:
        """Every forecast timestamp must fall strictly after its training window."""
        train, _, forecasts = _forecast()
        last_train = train.group_by("unique_id").agg(pl.col("ds").max().alias("train_end"))
        joined = forecasts.join(last_train, on="unique_id", how="left")
        assert joined.filter(pl.col("ds") <= pl.col("train_end")).height == 0

    def test_forecasts_align_with_test_ds(self) -> None:
        """Forecast timestamps join 1:1 with the held-out test timestamps."""
        _, test, forecasts = _forecast()
        merged = test.join(forecasts, on=["unique_id", "ds"], how="inner")
        assert merged.height == test.height
