"""Baseline forecasting models.

These are the floor. Every more-sophisticated model must beat SeasonalNaive
consistently before we treat it as useful. If it doesn't, the data prep or
evaluation is broken — fix that first.

Implementations are thin wrappers around statsforecast's models, returning
forecasts as Polars DataFrames in long format.
"""

from __future__ import annotations

import polars as pl


def forecast_baselines(
    train_df: pl.DataFrame,
    *,
    horizon: int,
    seasonality: int,
    freq: str,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Generate forecasts from all baseline models.

    Args:
        train_df: Training data (unique_id, ds, y) in Polars.
        horizon: Number of steps to forecast.
        seasonality: Seasonal period for SeasonalNaive.
        freq: Pandas frequency string (e.g. "H", "D", "M").
        n_jobs: Worker processes for StatsForecast. -1 uses all cores.

    Returns:
        Polars DataFrame with columns: unique_id, ds,
        Naive, SeasonalNaive, HistoricAverage, RandomWalkWithDrift.
    """
    # Lazy import to keep top-of-file imports lightweight for the test suite
    from statsforecast import StatsForecast
    from statsforecast.models import (
        HistoricAverage,
        Naive,
        RandomWalkWithDrift,
        SeasonalNaive,
    )

    models = [
        Naive(),
        SeasonalNaive(season_length=seasonality),
        HistoricAverage(),
        RandomWalkWithDrift(),
    ]

    # Nixtla currently expects pandas at the boundary
    train_pd = train_df.sort(["unique_id", "ds"]).to_pandas()

    sf = StatsForecast(models=models, freq=freq, n_jobs=n_jobs)
    forecasts_pd = sf.forecast(df=train_pd, h=horizon)
    forecasts = pl.from_pandas(
        forecasts_pd.reset_index() if "unique_id" not in forecasts_pd.columns else forecasts_pd
    )
    # Honor the project invariant: all timestamps are Datetime("us", "UTC").
    # StatsForecast hands ds back as nanosecond precision; align it so forecasts
    # join cleanly against the us-precision actuals.
    return forecasts.with_columns(pl.col("ds").cast(pl.Datetime("us", "UTC")))
