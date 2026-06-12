"""Smoke tests for data loaders."""

from __future__ import annotations

import polars as pl

from prophet.data.loaders import load_synthetic


class TestLoadSynthetic:
    def test_returns_long_format(self) -> None:
        df = load_synthetic(n_series=3, n_obs=100, frequency="1h", seed=42)
        assert set(df.columns) == {"unique_id", "ds", "y"}

    def test_correct_series_count(self) -> None:
        df = load_synthetic(n_series=5, n_obs=50, frequency="1h", seed=42)
        assert df["unique_id"].n_unique() == 5

    def test_correct_obs_per_series(self) -> None:
        df = load_synthetic(n_series=2, n_obs=100, frequency="1h", seed=42)
        for uid in df["unique_id"].unique():
            assert df.filter(pl.col("unique_id") == uid).height == 100

    def test_reproducible_with_seed(self) -> None:
        df1 = load_synthetic(n_series=2, n_obs=50, frequency="1h", seed=42)
        df2 = load_synthetic(n_series=2, n_obs=50, frequency="1h", seed=42)
        assert df1.equals(df2)
