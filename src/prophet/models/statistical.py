"""Statistical forecasting models (Phase 2).

Classical methods from StatsForecast — AutoARIMA, AutoETS, AutoTheta,
DynamicOptimizedTheta — plus a simple ensemble: the mean of the top-3 models
ranked by training-set cross-validation MASE.

These must beat the SeasonalNaive floor (see docs/phase-1-results.md). Like the
baselines, every wrapper returns a Polars long-format DataFrame with
``ds`` as ``Datetime("us", "UTC")`` per the project invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Model column names produced by StatsForecast (each model's alias).
MODEL_NAMES = ["AutoARIMA", "AutoETS", "AutoTheta", "DynamicOptimizedTheta"]
ENSEMBLE_COL = "Ensemble"


@dataclass(frozen=True)
class StatisticalInfo:
    """Selection metadata from a statistical forecast run.

    Attributes:
        cv_mase: Mean cross-validation MASE per model (lower is better).
        ensemble_members: The top-3 model names blended into the ensemble.
        ensemble_weights: Normalized inverse-CV-MASE weight per member.
        cv_windows: Number of CV windows used for ranking.
    """

    cv_mase: dict[str, float]
    ensemble_members: list[str]
    ensemble_weights: dict[str, float]
    cv_windows: int


def forecast_statistical(
    train_df: pl.DataFrame,
    *,
    horizon: int,
    season_length: int,
    freq: str,
    cv_windows: int = 2,
    rank_sample_n: int | None = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[pl.DataFrame, StatisticalInfo]:
    """Fit the statistical model ladder and produce forecasts plus an ensemble.

    The ensemble is the inverse-CV-MASE weighted mean of the three models with
    the lowest training-set cross-validation MASE: the better a model's CV MASE,
    the more it counts. This degrades gracefully when one model dominates —
    unlike a flat mean, which a weak member can drag down. Ranking/weighting CV
    uses ``cv_windows`` windows (kept small by default — AutoARIMA on hourly data
    is expensive — since it only shapes the ensemble, not the per-model metrics).

    Ensemble membership is a single global choice, so the ranking CV can run on a
    representative sample of series (``rank_sample_n``) while the final forecast
    still runs over every series. On large hourly sets this avoids paying the
    AutoARIMA CV cost across the whole dataset.

    Args:
        train_df: Training data (unique_id, ds, y) in Polars long format.
        horizon: Number of steps to forecast.
        season_length: Seasonal period (24 for hourly, etc.).
        freq: Pandas frequency string (e.g. "h", "D", "MS").
        cv_windows: CV windows used to rank models for the ensemble.
        rank_sample_n: If set, rank models on this many sampled series rather
            than the full set. The final forecast always covers all series.
        seed: Seed for the ranking-sample draw.
        n_jobs: Worker processes for StatsForecast. -1 uses all cores; cap it
            to leave headroom for interactive use on a shared machine.

    Returns:
        (forecasts, info) where forecasts has columns unique_id, ds, the four
        model columns, and an Ensemble column; info records the CV ranking.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        DynamicOptimizedTheta,
    )
    from utilsforecast.losses import mase as uf_mase

    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        DynamicOptimizedTheta(season_length=season_length),
    ]

    # Nixtla expects pandas at the boundary.
    train_sorted = train_df.sort(["unique_id", "ds"])
    train_pd = train_sorted.to_pandas()
    sf = StatsForecast(models=models, freq=freq, n_jobs=n_jobs)

    # Rank models by training-set CV MASE to choose the top-3 ensemble members,
    # optionally on a sample of series to bound the AutoARIMA CV cost.
    rank_df = train_sorted
    if rank_sample_n is not None:
        ids = train_sorted["unique_id"].unique().sort()
        sampled = ids.sample(n=min(rank_sample_n, len(ids)), seed=seed)
        rank_df = train_sorted.filter(pl.col("unique_id").is_in(sampled))
    rank_pd = rank_df.to_pandas()

    cv = sf.cross_validation(df=rank_pd, h=horizon, n_windows=cv_windows, step_size=horizon)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    per_series = uf_mase(
        cv,
        models=MODEL_NAMES,
        seasonality=season_length,
        train_df=rank_pd,
    )
    cv_mase = {name: float(per_series[name].mean()) for name in MODEL_NAMES}
    ensemble_members = sorted(MODEL_NAMES, key=lambda n: cv_mase[n])[:3]

    # Inverse-CV-MASE weights over the chosen members (normalized to sum to 1).
    inv = {name: 1.0 / cv_mase[name] for name in ensemble_members}
    inv_total = sum(inv.values())
    ensemble_weights = {name: w / inv_total for name, w in inv.items()}

    # Final forecast refits on the full training series.
    forecasts_pd = sf.forecast(df=train_pd, h=horizon)
    forecasts = pl.from_pandas(
        forecasts_pd.reset_index() if "unique_id" not in forecasts_pd.columns else forecasts_pd
    ).with_columns(pl.col("ds").cast(pl.Datetime("us", "UTC")))

    # Ensemble = inverse-CV-MASE weighted mean of the top-3 models.
    ensemble_expr = pl.sum_horizontal(
        [pl.col(name) * weight for name, weight in ensemble_weights.items()]
    )
    forecasts = forecasts.with_columns(ensemble_expr.alias(ENSEMBLE_COL))

    info = StatisticalInfo(
        cv_mase=cv_mase,
        ensemble_members=ensemble_members,
        ensemble_weights=ensemble_weights,
        cv_windows=cv_windows,
    )
    return forecasts, info
