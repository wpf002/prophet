"""Smoke tests for the statistical model ladder.

Runs the real StatsForecast wrappers on tiny synthetic data with a single CV
window, so the suite exercises the ensemble logic without paying the full
AutoARIMA hourly cost.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prophet.data.loaders import load_synthetic
from prophet.evaluation.cross_validation import split_train_test
from prophet.models.statistical import (
    ENSEMBLE_COL,
    MODEL_NAMES,
    forecast_statistical,
)

HORIZON = 24
SEASONALITY = 24


def _run() -> tuple[pl.DataFrame, object]:
    df = load_synthetic(n_series=2, n_obs=240, frequency="1h", seed=42)
    train, _ = split_train_test(df, horizon=HORIZON)
    return forecast_statistical(
        train,
        horizon=HORIZON,
        season_length=SEASONALITY,
        freq="h",
        cv_windows=1,
    )


class TestForecastStatistical:
    def test_returns_all_models_and_ensemble(self) -> None:
        forecasts, _ = _run()
        expected = set(MODEL_NAMES) | {ENSEMBLE_COL}
        assert expected.issubset(set(forecasts.columns))

    def test_horizon_rows_per_series(self) -> None:
        forecasts, _ = _run()
        assert forecasts.height == 2 * HORIZON

    def test_ds_is_utc_microsecond(self) -> None:
        forecasts, _ = _run()
        assert forecasts.schema["ds"] == pl.Datetime("us", "UTC")

    def test_ensemble_is_weighted_mean_of_members(self) -> None:
        forecasts, info = _run()
        assert len(info.ensemble_members) == 3
        # Weights are normalized inverse-CV-MASE over the members.
        assert set(info.ensemble_weights) == set(info.ensemble_members)
        assert abs(sum(info.ensemble_weights.values()) - 1.0) < 1e-9
        recomputed = forecasts.select(
            pl.sum_horizontal(
                [pl.col(name) * w for name, w in info.ensemble_weights.items()]
            ).alias("expected")
        )["expected"]
        assert np.allclose(forecasts[ENSEMBLE_COL].to_numpy(), recomputed.to_numpy())
