"""Stateless ad-hoc forecasting — bring your own series, get a forecast + verdict.

This is the "works with anything" primitive: a caller sends a raw time series and
Prophet picks a model, forecasts, and reports whether the series is even
forecastable (does the best model beat a naive baseline?). No domain
registration, no pre-training, no server-side code per source.

The model pool is deliberately fast (baselines + AutoETS/Theta, no AutoARIMA) so
the endpoint stays interactive. When the series is long enough, the winner is
chosen by a short expanding-window cross-validation on the caller's own data;
when it's too short to validate, a robust default is used and the verdict is
reported as "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

# Seasonal period per pandas offset alias. Falls back to 1 (non-seasonal).
SEASONALITY_BY_FREQ: dict[str, int] = {
    "h": 24, "H": 24,
    "D": 7, "B": 5,
    "W": 52, "W-SUN": 52, "W-MON": 52,
    "MS": 12, "M": 12, "ME": 12,
    "QS": 4, "Q": 4, "QE": 4,
    "YS": 1, "Y": 1, "A": 1, "YE": 1,
}

# Candidate models kept fast for interactive latency (AutoARIMA excluded).
_STATISTICAL = ("AutoETS", "AutoTheta", "DynamicOptimizedTheta")
_BASELINES = ("Naive", "SeasonalNaive")


@dataclass(frozen=True)
class AdhocPoint:
    """One forecast step."""

    ds: datetime
    y_hat: float
    lo: dict[str, float] | None
    hi: dict[str, float] | None


@dataclass(frozen=True)
class AdhocForecast:
    """Result of an ad-hoc forecast plus the forecastability verdict."""

    model: str
    seasonality: int
    points: list[AdhocPoint]
    beats_naive: bool | None  # None when the series was too short to validate
    naive_mase: float | None
    model_mase: float | None
    n_obs: int


def seasonality_for(freq: str, override: int | None) -> int:
    """Resolve the seasonal period from an explicit override or the frequency."""
    if override is not None:
        return max(1, override)
    return SEASONALITY_BY_FREQ.get(freq, SEASONALITY_BY_FREQ.get(freq.upper(), 1))


def _to_pandas(series: list[tuple[datetime, float]]) -> object:
    """Build the single-series pandas frame StatsForecast expects."""
    df = pl.DataFrame(
        {"unique_id": ["series"] * len(series), "ds": [s[0] for s in series], "y": [s[1] for s in series]}
    ).sort("ds")
    return df.to_pandas()


def forecast_adhoc(
    series: list[tuple[datetime, float]],
    *,
    horizon: int,
    freq: str,
    seasonality: int | None = None,
    level: list[int] | None = None,
) -> AdhocForecast:
    """Select a model on the caller's own series, forecast, and report the verdict.

    Args:
        series: (ds, y) pairs. Must be regularly spaced at ``freq``.
        horizon: steps to forecast.
        freq: pandas offset alias (e.g. "MS", "D", "h").
        seasonality: seasonal period; inferred from ``freq`` when omitted.
        level: prediction-interval confidence levels (e.g. [80]).

    Returns:
        AdhocForecast with the chosen model, forecast points, and — when the
        series was long enough to cross-validate — whether it beats naive.

    Raises:
        ValueError: fewer than two points, or fewer than seasonality + horizon.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoETS,
        AutoTheta,
        DynamicOptimizedTheta,
        Naive,
        SeasonalNaive,
    )
    from statsforecast.utils import ConformalIntervals

    n = len(series)
    s = seasonality_for(freq, seasonality)
    if n < horizon + 2:
        raise ValueError(f"Series too short: {n} points for horizon {horizon} (need >= {horizon + 2}).")
    # Not enough history for the seasonal period → degrade to non-seasonal rather
    # than reject, so short series still get a (non-seasonal) forecast.
    if n < 2 * s:
        s = 1

    df = _to_pandas(series)
    levels = level or []
    models = [
        Naive(),
        SeasonalNaive(season_length=s),
        AutoETS(season_length=s),
        AutoTheta(season_length=s),
        DynamicOptimizedTheta(season_length=s),
    ]

    # Validate on the caller's own data when there's room for expanding-window CV.
    # Statistical models (AutoETS/Theta) also need a minimum amount of history —
    # below that we stay on baselines, which always fit.
    cv_ok = n >= 2 * horizon + s + 2 and n >= 12
    beats_naive: bool | None = None
    naive_mase: float | None = None
    model_mase: float | None = None
    # Robust default when we can't validate: seasonal-naive if we have a season of
    # history, else plain naive. Both always fit, even on tiny series.
    winner = "SeasonalNaive" if s > 1 and n >= 2 * s else "Naive"

    if cv_ok:
        from utilsforecast.losses import mase as uf_mase

        sf = StatsForecast(models=models, freq=freq, n_jobs=1)
        n_windows = 3 if n >= 3 * horizon + s + 2 else 2
        cv = sf.cross_validation(df=df, h=horizon, n_windows=n_windows, step_size=horizon)
        cv = cv.reset_index() if "unique_id" not in cv.columns else cv
        per = uf_mase(cv, models=[*_BASELINES, *_STATISTICAL], seasonality=s, train_df=df)
        scores = {m: float(per[m].iloc[0]) for m in (*_BASELINES, *_STATISTICAL)}
        naive_mase = min(scores["Naive"], scores["SeasonalNaive"])
        best_stat = min(_STATISTICAL, key=lambda m: scores[m])
        model_mase = scores[best_stat]
        beats_naive = model_mase < naive_mase
        # Serve the statistical winner if it beats naive, else the naive baseline.
        winner = best_stat if beats_naive else min(_BASELINES, key=lambda m: scores[m])

    # Refit the winner on the full series and forecast, with conformal intervals
    # when there's enough history to calibrate them. If the chosen model fails to
    # fit (e.g. a degenerate series), fall back to Naive so a forecast still
    # returns — the endpoint must work with anything.
    def _build(name: str) -> object:
        return {
            "Naive": Naive(),
            "SeasonalNaive": SeasonalNaive(season_length=s),
            "AutoETS": AutoETS(season_length=s),
            "AutoTheta": AutoTheta(season_length=s),
            "DynamicOptimizedTheta": DynamicOptimizedTheta(season_length=s),
        }[name]

    use_intervals = bool(levels) and cv_ok
    try:
        final = StatsForecast(models=[_build(winner)], freq=freq, n_jobs=1)
        intervals = ConformalIntervals(n_windows=2, h=horizon) if use_intervals else None
        final.fit(df, prediction_intervals=intervals)
        fc = final.predict(h=horizon, level=levels if use_intervals else None)
    except Exception:
        winner, use_intervals = "Naive", False
        final = StatsForecast(models=[Naive()], freq=freq, n_jobs=1)
        final.fit(df)
        fc = final.predict(h=horizon)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()

    points: list[AdhocPoint] = []
    for row in fc.to_dict("records"):
        lo = {str(lv): float(row[f"{winner}-lo-{lv}"]) for lv in levels} if use_intervals else None
        hi = {str(lv): float(row[f"{winner}-hi-{lv}"]) for lv in levels} if use_intervals else None
        points.append(AdhocPoint(ds=row["ds"], y_hat=float(row[winner]), lo=lo, hi=hi))

    return AdhocForecast(
        model=winner,
        seasonality=s,
        points=points,
        beats_naive=beats_naive,
        naive_mase=naive_mase,
        model_mase=model_mase,
        n_obs=n,
    )
