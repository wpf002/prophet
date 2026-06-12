"""Dataset loaders for M4 and M5 competition data.

These loaders return Polars DataFrames in the Nixtla long format:
    unique_id (str): time series identifier
    ds (datetime): timestamp
    y (float): observed value
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import polars as pl


class M4Frequency(StrEnum):
    """M4 dataset frequencies with associated seasonal periods."""

    HOURLY = "Hourly"
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    YEARLY = "Yearly"

    @property
    def seasonality(self) -> int:
        """Seasonal period for this frequency (used by seasonal naive, sMAPE)."""
        return {
            "Hourly": 24,
            "Daily": 7,
            "Weekly": 1,
            "Monthly": 12,
            "Quarterly": 4,
            "Yearly": 1,
        }[self.value]

    @property
    def horizon(self) -> int:
        """Standard M4 forecast horizon for this frequency."""
        return {
            "Hourly": 48,
            "Daily": 14,
            "Weekly": 13,
            "Monthly": 18,
            "Quarterly": 8,
            "Yearly": 6,
        }[self.value]


def load_m4(
    frequency: M4Frequency,
    data_dir: Path,
    *,
    sample_n: int | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load M4 train and test data for a given frequency.

    Args:
        frequency: M4 frequency to load.
        data_dir: Root directory containing M4 parquet files.
        sample_n: Optional cap on number of unique series (for faster iteration).
        seed: Random seed for sampling.

    Returns:
        (train, test) tuple of Polars DataFrames in long format.

    Raises:
        FileNotFoundError: If processed M4 parquet files are missing.
            Run `make download-m4` to download and preprocess.
    """
    train_path = data_dir / "m4" / f"{frequency.value.lower()}-train.parquet"
    test_path = data_dir / "m4" / f"{frequency.value.lower()}-test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"M4 {frequency.value} parquet files not found at {train_path.parent}. "
            "Run `make download-m4` first."
        )

    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)

    if sample_n is not None:
        unique_ids = train["unique_id"].unique().sort()
        sampled_ids = unique_ids.sample(n=min(sample_n, len(unique_ids)), seed=seed)
        train = train.filter(pl.col("unique_id").is_in(sampled_ids))
        test = test.filter(pl.col("unique_id").is_in(sampled_ids))

    return train, test


def load_synthetic(
    n_series: int = 5,
    n_obs: int = 200,
    frequency: str = "1h",
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic time series for testing.

    Produces n_series series, each with n_obs observations,
    with linear trend + sinusoidal seasonality + gaussian noise.

    Args:
        n_series: Number of unique series to generate.
        n_obs: Number of observations per series.
        frequency: Interval string. Supported: "1h", "1d".
        seed: Random seed.

    Returns:
        Polars DataFrame in long format (unique_id, ds, y).
    """
    import datetime as dt

    import numpy as np

    delta_map = {"1h": dt.timedelta(hours=1), "1d": dt.timedelta(days=1)}
    if frequency not in delta_map:
        raise ValueError(f"Unsupported frequency {frequency!r}. Supported: {list(delta_map)}")
    delta = delta_map[frequency]
    season_period = 24 if frequency == "1h" else 7

    rng = np.random.default_rng(seed)
    start_ts = dt.datetime(2024, 1, 1)
    timestamps = [start_ts + delta * j for j in range(n_obs)]

    dfs: list[pl.DataFrame] = []
    for i in range(n_series):
        trend = np.linspace(0, 10, n_obs)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(n_obs) / season_period)
        noise = rng.normal(0, 1, n_obs)
        level = 50 + i * 20
        y = level + trend + seasonality + noise

        dfs.append(
            pl.DataFrame(
                {
                    "unique_id": [f"series_{i:03d}"] * n_obs,
                    "ds": timestamps,
                    "y": y.astype(float),
                }
            )
        )

    return pl.concat(dfs)
