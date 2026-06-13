"""Forecast evaluation metrics.

Primary metric: MASE (Mean Absolute Scaled Error) — scale-free, comparable
across series of different magnitudes.

All functions accept and return Polars DataFrames in long format:
    unique_id (str), ds (datetime), y (float, actual), y_hat (float, forecast)
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

# Metric functions operate on 1-D float arrays (typically `Series.to_numpy()`).
FloatArray = npt.NDArray[np.float64]


def mae(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def smape(y: FloatArray, y_hat: FloatArray) -> float:
    """Symmetric Mean Absolute Percentage Error (M4 official definition).

    Range: [0, 200]. Lower is better.
    """
    denom = (np.abs(y) + np.abs(y_hat)) / 2.0
    # Avoid division by zero: where both are zero, error is zero
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y[mask] - y_hat[mask]) / denom[mask]) * 100)


def mape(y: FloatArray, y_hat: FloatArray) -> float:
    """Mean Absolute Percentage Error. Undefined where y == 0."""
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask])) * 100)


def wape(y: FloatArray, y_hat: FloatArray) -> float:
    """Weighted Absolute Percentage Error (a.k.a. WMAPE).

    Equivalent to sum(|y - y_hat|) / sum(|y|). Robust to zeros in y as long as
    some y values are nonzero.
    """
    denom = float(np.sum(np.abs(y)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - y_hat)) / denom * 100)


def mase(
    y: FloatArray,
    y_hat: FloatArray,
    y_train: FloatArray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    Scaled by the in-sample MAE of a seasonal naive forecast.
    A MASE < 1 means the forecast is better than seasonal naive on training data.

    Args:
        y: Test set actual values.
        y_hat: Test set forecasts.
        y_train: Training set values (used to compute scaling denominator).
        seasonality: Seasonal period (1 for non-seasonal, 24 for hourly, etc.).

    Returns:
        MASE value. Lower is better. NaN if scaling denominator is zero.
    """
    if len(y_train) <= seasonality:
        raise ValueError(
            f"Training series length ({len(y_train)}) must exceed seasonality ({seasonality})."
        )
    scale = float(np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y - y_hat)) / scale)


def pinball_loss(y: FloatArray, q_hat: FloatArray, tau: float) -> float:
    """Pinball loss for quantile forecasts.

    Args:
        y: Actual values.
        q_hat: Quantile forecast at level tau.
        tau: Quantile level in (0, 1).

    Returns:
        Mean pinball loss across observations. Lower is better.
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    diff = y - q_hat
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def evaluate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    forecast_df: pl.DataFrame,
    *,
    seasonality: int = 1,
    model_col: str = "y_hat",
) -> pl.DataFrame:
    """Evaluate forecasts across all series and return per-series metrics.

    Args:
        train_df: Training data (unique_id, ds, y).
        test_df: Test data (unique_id, ds, y).
        forecast_df: Forecasts (unique_id, ds, <model_col>).
        seasonality: Seasonal period for MASE scaling.
        model_col: Column name in forecast_df containing the forecast values.

    Returns:
        DataFrame with one row per unique_id containing MAE, RMSE, sMAPE, MASE, WAPE.
    """
    merged = test_df.join(
        forecast_df.select(["unique_id", "ds", model_col]),
        on=["unique_id", "ds"],
        how="inner",
    )
    if merged.height == 0:
        raise ValueError("No overlap between test_df and forecast_df on (unique_id, ds).")

    rows: list[dict[str, float | str]] = []
    for uid in merged["unique_id"].unique().sort():
        actual = merged.filter(pl.col("unique_id") == uid)["y"].to_numpy()
        pred = merged.filter(pl.col("unique_id") == uid)[model_col].to_numpy()
        train_y = train_df.filter(pl.col("unique_id") == uid)["y"].to_numpy()
        rows.append(
            {
                "unique_id": uid,
                "mae": mae(actual, pred),
                "rmse": rmse(actual, pred),
                "smape": smape(actual, pred),
                "wape": wape(actual, pred),
                "mase": mase(actual, pred, train_y, seasonality=seasonality),
            }
        )
    return pl.DataFrame(rows)


def aggregate_metrics(metrics_df: pl.DataFrame) -> dict[str, float]:
    """Compute mean of each metric across series."""
    return {
        col: float(cast(float, metrics_df[col].mean()))
        for col in ["mae", "rmse", "smape", "wape", "mase"]
        if col in metrics_df.columns
    }
