"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from prophet.evaluation.metrics import (
    aggregate_metrics,
    evaluate,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    smape,
    wape,
)


class TestPointMetrics:
    def test_mae_perfect_forecast_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known_value(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([2.0, 3.0, 4.0])
        assert mae(y, y_hat) == pytest.approx(1.0)

    def test_rmse_known_value(self) -> None:
        y = np.array([0.0, 0.0, 0.0])
        y_hat = np.array([3.0, 4.0, 0.0])
        assert rmse(y, y_hat) == pytest.approx(np.sqrt(25 / 3))

    def test_smape_perfect_forecast_is_zero(self) -> None:
        y = np.array([10.0, 20.0])
        assert smape(y, y) == 0.0

    def test_smape_handles_zero_zero(self) -> None:
        y = np.array([0.0, 0.0])
        y_hat = np.array([0.0, 0.0])
        assert smape(y, y_hat) == 0.0

    def test_mape_returns_nan_on_all_zeros(self) -> None:
        y = np.array([0.0, 0.0])
        y_hat = np.array([1.0, 1.0])
        assert np.isnan(mape(y, y_hat))

    def test_wape_known_value(self) -> None:
        y = np.array([10.0, 20.0])
        y_hat = np.array([11.0, 19.0])
        # sum errors = 2, sum |y| = 30, wape = 2/30 * 100
        assert wape(y, y_hat) == pytest.approx(2 / 30 * 100)


class TestMase:
    def test_mase_requires_train_longer_than_seasonality(self) -> None:
        with pytest.raises(ValueError, match="seasonality"):
            mase(
                y=np.array([1.0]),
                y_hat=np.array([1.0]),
                y_train=np.array([1.0, 2.0]),
                seasonality=5,
            )

    def test_mase_perfect_forecast_is_zero(self) -> None:
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0])
        result = mase(y=y, y_hat=y, y_train=y_train, seasonality=1)
        assert result == 0.0

    def test_mase_equals_one_when_matching_seasonal_naive_scale(self) -> None:
        # Training series with constant unit step
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Test forecast with average error of 1.0 (matches scale of 1.0)
        y = np.array([6.0, 7.0])
        y_hat = np.array([7.0, 8.0])
        assert mase(y, y_hat, y_train, seasonality=1) == pytest.approx(1.0)


class TestPinball:
    def test_pinball_rejects_tau_out_of_range(self) -> None:
        y = np.array([1.0])
        q_hat = np.array([1.0])
        with pytest.raises(ValueError, match="tau"):
            pinball_loss(y, q_hat, tau=0.0)
        with pytest.raises(ValueError, match="tau"):
            pinball_loss(y, q_hat, tau=1.0)

    def test_pinball_perfect_forecast_is_zero(self) -> None:
        y = np.array([1.0, 2.0])
        assert pinball_loss(y, y, tau=0.5) == 0.0


class TestEvaluate:
    def _toy_data(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)
        train = pl.DataFrame(
            {
                "unique_id": ["a"] * 10 + ["b"] * 10,
                "ds": [base + timedelta(hours=i) for i in range(10)] * 2,
                "y": [float(i) for i in range(10)] * 2,
            }
        )
        test = pl.DataFrame(
            {
                "unique_id": ["a"] * 2 + ["b"] * 2,
                "ds": [base + timedelta(hours=i) for i in range(10, 12)] * 2,
                "y": [10.0, 11.0, 10.0, 11.0],
            }
        )
        forecast = pl.DataFrame(
            {
                "unique_id": ["a"] * 2 + ["b"] * 2,
                "ds": [base + timedelta(hours=i) for i in range(10, 12)] * 2,
                "y_hat": [10.0, 11.0, 11.0, 12.0],
            }
        )
        return train, test, forecast

    def test_evaluate_produces_one_row_per_series(self) -> None:
        train, test, forecast = self._toy_data()
        result = evaluate(train, test, forecast, seasonality=1)
        assert result.height == 2
        assert set(result["unique_id"].to_list()) == {"a", "b"}

    def test_evaluate_columns(self) -> None:
        train, test, forecast = self._toy_data()
        result = evaluate(train, test, forecast, seasonality=1)
        assert {"unique_id", "mae", "rmse", "smape", "wape", "mase"}.issubset(result.columns)

    def test_aggregate_metrics_returns_means(self) -> None:
        train, test, forecast = self._toy_data()
        result = evaluate(train, test, forecast, seasonality=1)
        agg = aggregate_metrics(result)
        assert "mase" in agg
        assert "smape" in agg
        assert isinstance(agg["mase"], float)
