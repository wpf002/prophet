"""Tests for cross-validation utilities."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from prophet.evaluation.cross_validation import CVConfig, split_train_test


class TestCVConfig:
    def test_defaults(self) -> None:
        config = CVConfig()
        assert config.n_windows == 5
        assert config.horizon == 24
        assert config.effective_step_size == 24

    def test_step_size_defaults_to_horizon(self) -> None:
        config = CVConfig(horizon=10)
        assert config.effective_step_size == 10

    def test_step_size_explicit(self) -> None:
        config = CVConfig(horizon=10, step_size=5)
        assert config.effective_step_size == 5

    def test_rejects_zero_windows(self) -> None:
        with pytest.raises(ValueError, match="n_windows"):
            CVConfig(n_windows=0)

    def test_rejects_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            CVConfig(horizon=0)


class TestSplitTrainTest:
    def _build_df(self, n_series: int = 2, n_obs: int = 20) -> pl.DataFrame:
        base = datetime(2024, 1, 1)
        rows: list[dict[str, object]] = []
        for s in range(n_series):
            for i in range(n_obs):
                rows.append(
                    {
                        "unique_id": f"s_{s}",
                        "ds": base + timedelta(hours=i),
                        "y": float(s * 100 + i),
                    }
                )
        return pl.DataFrame(rows)

    def test_train_test_lengths_match_horizon(self) -> None:
        df = self._build_df(n_series=2, n_obs=20)
        train, test = split_train_test(df, horizon=5)
        for uid in ["s_0", "s_1"]:
            assert train.filter(pl.col("unique_id") == uid).height == 15
            assert test.filter(pl.col("unique_id") == uid).height == 5

    def test_test_contains_latest_observations(self) -> None:
        df = self._build_df(n_series=1, n_obs=10)
        train, test = split_train_test(df, horizon=3)
        last_train_ds = train["ds"].max()
        first_test_ds = test["ds"].min()
        assert last_train_ds < first_test_ds

    def test_no_overlap(self) -> None:
        df = self._build_df(n_series=2, n_obs=20)
        train, test = split_train_test(df, horizon=5)
        overlap = train.join(test, on=["unique_id", "ds"], how="inner")
        assert overlap.height == 0
