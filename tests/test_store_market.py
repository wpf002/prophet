"""Tests for pure monitoring/market helpers (no DB or network)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from prophet.data.market import TARGETS, to_long
from prophet.monitoring.store import forecast_rows
from prophet.serving.registry import ForecastPoint


class TestForecastRows:
    def test_shapes_rows_with_95_bounds(self) -> None:
        pts = [
            ForecastPoint(
                ds=dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
                y_hat=100.0,
                lo={"80": 90.0, "95": 80.0},
                hi={"80": 110.0, "95": 120.0},
            )
        ]
        rows = forecast_rows("AAPL", "market-vol:LightGBM", 5, pts)
        assert rows == [
            (
                "AAPL",
                dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
                5,
                "market-vol:LightGBM",
                100.0,
                80.0,
                120.0,
            )
        ]

    def test_handles_missing_intervals(self) -> None:
        pts = [
            ForecastPoint(ds=dt.datetime(2000, 1, 1, tzinfo=dt.UTC), y_hat=5.0, lo=None, hi=None)
        ]
        rows = forecast_rows("X", "m", 1, pts)
        assert rows[0][5] is None and rows[0][6] is None


class TestMarketReindex:
    def test_to_long_reindexes_trading_days_to_regular_grid(self) -> None:
        # Two tickers with a weekend gap; reindex should make consecutive steps.
        bars = pl.DataFrame(
            {
                "unique_id": ["AAPL", "AAPL", "AAPL", "MSFT"],
                "date": ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-05"],
                "close": [10.0, 11.0, 12.0, 20.0],
                "volume": [100, 200, 300, 50],
            }
        )
        out = to_long(bars, TARGETS["market-vol"])
        assert set(out.columns) == {"unique_id", "ds", "y"}
        aapl = out.filter(pl.col("unique_id") == "AAPL").sort("ds")
        # 3 consecutive daily steps despite the calendar gap, volume preserved.
        gaps = aapl["ds"].diff().drop_nulls().unique().to_list()
        assert gaps == [dt.timedelta(days=1)]
        assert aapl["y"].to_list() == [100, 200, 300]

    def test_close_target_is_cents(self) -> None:
        bars = pl.DataFrame(
            {"unique_id": ["A"], "date": ["2024-01-05"], "close": [1.23], "volume": [10]}
        )
        out = to_long(bars, TARGETS["market-close"])
        assert out["y"].to_list() == [123]  # 1.23 dollars -> 123 cents
