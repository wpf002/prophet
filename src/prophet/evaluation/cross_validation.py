"""Time-series cross-validation utilities.

Wraps Nixtla's cross_validation to enforce expanding-window CV with
project-standard defaults. No random splits, ever.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation configuration.

    Attributes:
        n_windows: Number of CV windows (folds).
        horizon: Forecast horizon (number of steps to predict each fold).
        step_size: Stride between windows. Defaults to horizon (non-overlapping).
        seasonality: Seasonal period for the data.
    """

    n_windows: int = 5
    horizon: int = 24
    step_size: int | None = None
    seasonality: int = 24

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    @property
    def effective_step_size(self) -> int:
        return self.step_size if self.step_size is not None else self.horizon


def split_train_test(
    df: pl.DataFrame,
    horizon: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a long-format DataFrame into train/test by holding out the last `horizon` per series.

    Args:
        df: Long-format data (unique_id, ds, y).
        horizon: Number of observations per series to hold out for testing.

    Returns:
        (train, test) tuple.
    """
    sorted_df = df.sort(["unique_id", "ds"])
    test = sorted_df.group_by("unique_id", maintain_order=True).tail(horizon)
    train = (
        sorted_df.join(
            test.select(["unique_id", "ds"]).with_columns(pl.lit(True).alias("_is_test")),
            on=["unique_id", "ds"],
            how="left",
        )
        .filter(pl.col("_is_test").is_null())
        .drop("_is_test")
    )
    return train, test
