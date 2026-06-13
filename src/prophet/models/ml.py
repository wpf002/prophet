"""ML forecasting models (Phase 3).

LightGBM via MLForecast — one global gradient-boosted model trained across all
series on engineered features: lags, rolling mean/std, and calendar features.
Ships an untuned model (library defaults) and an Optuna-tuned model.

Target: beat the best Phase 2 statistical model (AutoARIMA, MASE 0.948).

Caveat on calendar features: M4 series are anonymised and carry only an integer
step index, so the synthesised UTC timestamps don't map to real-world clock time.
The calendar features therefore encode each series' position modulo 24h/168h (a
consistent within-series phase), not a real hour-of-day — the lag features do the
real seasonal work. They're included per the roadmap; LightGBM ignores what
doesn't help.

Like the other model modules, forecasts come back as Polars long format with
``ds`` as ``Datetime("us", "UTC")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

DEFAULT_LAGS = [1, 24, 48, 168]
DEFAULT_ROLLING_WINDOWS = (24, 168)
UNTUNED_COL = "LightGBM"
TUNED_COL = "LightGBM_tuned"


def is_weekend(dates: Any) -> npt.NDArray[np.int_]:
    """Calendar feature: 1 on Sat/Sun, else 0. (Name becomes the feature column.)"""
    flags: npt.NDArray[np.int_] = (dates.dayofweek >= 5).astype(int)
    return flags


_DATE_FEATURES: list[Any] = ["hour", "dayofweek", "day", "month", is_weekend]


@dataclass(frozen=True)
class MLInfo:
    """Selection metadata from an ML forecast run.

    Attributes:
        best_params: Optuna-selected LightGBM hyperparameters (empty if untuned).
        feature_importance: Gain-based importance per feature (tuned model if
            tuning ran, else untuned).
        cv_mase_untuned: Cross-validation MASE of the untuned model.
        cv_mase_tuned: Cross-validation MASE of the tuned model (None if untuned).
        n_trials: Number of Optuna trials run.
        feature_names: Ordered feature names used by the model.
    """

    best_params: dict[str, Any]
    feature_importance: dict[str, float]
    cv_mase_untuned: float
    cv_mase_tuned: float | None
    n_trials: int
    feature_names: list[str] = field(default_factory=list)


def _build_mlf(
    models: dict[str, Any],
    *,
    freq: str,
    lags: list[int],
    rolling_windows: tuple[int, ...],
    seasonal_diff: int,
) -> Any:
    """Construct an MLForecast with the project's standard feature recipe.

    Target transforms (fresh per call — they are fitted in place): seasonal
    differencing at ``seasonal_diff`` then per-series standardization. Both are
    essential on M4 hourly — differencing removes the dominant daily cycle (lifts
    untuned MASE from ~2.0 to ~0.97) and the scaler stops large-magnitude series
    from dominating the single global model.
    """
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean, RollingStd
    from mlforecast.target_transforms import Differences, LocalStandardScaler

    # Rolling stats are lagged by 1 so they only ever see data up to t-1.
    transforms: list[Any] = []
    for window in rolling_windows:
        transforms.append(RollingMean(window_size=window))
        transforms.append(RollingStd(window_size=window))

    return MLForecast(
        models=models,
        freq=freq,
        lags=lags,
        lag_transforms={1: transforms},
        date_features=_DATE_FEATURES,
        target_transforms=[Differences([seasonal_diff]), LocalStandardScaler()],
    )


def _cv_mase(
    mlf: Any,
    train_pd: Any,
    *,
    model_name: str,
    horizon: int,
    cv_windows: int,
    season_length: int,
) -> float:
    """Mean cross-validation MASE for a single model column."""
    from utilsforecast.losses import mase as uf_mase

    cv = mlf.cross_validation(df=train_pd, n_windows=cv_windows, h=horizon)
    per_series = uf_mase(cv, models=[model_name], seasonality=season_length, train_df=train_pd)
    return float(per_series[model_name].mean())


def forecast_ml(
    train_df: pl.DataFrame,
    *,
    horizon: int,
    season_length: int,
    freq: str,
    lags: list[int] | None = None,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    tune: bool = True,
    n_trials: int = 50,
    cv_windows: int = 2,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[pl.DataFrame, MLInfo]:
    """Forecast with LightGBM via MLForecast: untuned and (optionally) Optuna-tuned.

    Args:
        train_df: Training data (unique_id, ds, y) in Polars long format.
        horizon: Number of steps to forecast.
        season_length: Seasonal period (24 for hourly) — used for MASE scaling.
        freq: Pandas frequency string (e.g. "h").
        lags: Lag features. Defaults to [1, 24, 48, 168] (hourly).
        rolling_windows: Window sizes for rolling mean/std (lagged by 1).
        tune: If True, run Optuna to tune a second LightGBM.
        n_trials: Optuna trials (only when tune=True).
        cv_windows: CV windows used for tuning and reported CV MASE.
        seed: Random seed for LightGBM and the Optuna sampler.
        n_jobs: LightGBM threads. -1 uses all cores; cap to leave headroom.

    Returns:
        (forecasts, info). forecasts has columns unique_id, ds, "LightGBM", and
        "LightGBM_tuned" (when tuned); info records params, importance, CV MASE.
    """
    import lightgbm as lgb

    if lags is None:
        lags = list(DEFAULT_LAGS)

    train_pd = train_df.sort(["unique_id", "ds"]).to_pandas()

    def _lgbm(**params: Any) -> Any:
        return lgb.LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbosity=-1, **params)

    # Untuned baseline (library defaults).
    mlf_untuned = _build_mlf(
        {UNTUNED_COL: _lgbm()},
        freq=freq,
        lags=lags,
        rolling_windows=rolling_windows,
        seasonal_diff=season_length,
    )
    cv_mase_untuned = _cv_mase(
        mlf_untuned,
        train_pd,
        model_name=UNTUNED_COL,
        horizon=horizon,
        cv_windows=cv_windows,
        season_length=season_length,
    )

    final_models: dict[str, Any] = {UNTUNED_COL: _lgbm()}
    best_params: dict[str, Any] = {}
    cv_mase_tuned: float | None = None

    if tune:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
            mlf_trial = _build_mlf(
                {UNTUNED_COL: _lgbm(**params)},
                freq=freq,
                lags=lags,
                rolling_windows=rolling_windows,
                seasonal_diff=season_length,
            )
            return _cv_mase(
                mlf_trial,
                train_pd,
                model_name=UNTUNED_COL,
                horizon=horizon,
                cv_windows=cv_windows,
                season_length=season_length,
            )

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = dict(study.best_params)
        cv_mase_tuned = float(study.best_value)
        final_models[TUNED_COL] = _lgbm(subsample_freq=1, **best_params)

    # Fit on all training data and forecast the horizon.
    mlf = _build_mlf(
        final_models,
        freq=freq,
        lags=lags,
        rolling_windows=rolling_windows,
        seasonal_diff=season_length,
    )
    mlf.fit(train_pd)
    forecasts_pd = mlf.predict(horizon)
    forecasts = pl.from_pandas(forecasts_pd).with_columns(
        pl.col("ds").cast(pl.Datetime("us", "UTC"))
    )

    # Feature importance from the model we'd actually deploy (tuned if available).
    chosen = TUNED_COL if tune else UNTUNED_COL
    fitted = mlf.models_[chosen]
    names = list(fitted.booster_.feature_name())
    importances = [float(v) for v in fitted.feature_importances_]
    feature_importance = dict(zip(names, importances, strict=True))

    info = MLInfo(
        best_params=best_params,
        feature_importance=feature_importance,
        cv_mase_untuned=cv_mase_untuned,
        cv_mase_tuned=cv_mase_tuned,
        n_trials=n_trials if tune else 0,
        feature_names=names,
    )
    return forecasts, info
