"""Download and preprocess the M4 competition dataset.

Pulls M4 train + test splits via ``datasetsforecast`` (Nixtla's mirror) and
stores them as Parquet under ``data/raw/m4/`` in Nixtla long format:

    unique_id (str), ds (Datetime[us, UTC]), y (float)

M4 series are anonymised and carry only an integer step index, not real
timestamps. We synthesise monotonic UTC timestamps per series at the
frequency's natural interval so the data satisfies the project's
``Datetime("us", "UTC")`` invariant and round-trips cleanly through
StatsForecast.

Note on the split: ``datasetsforecast.M4.load`` returns each series as the FULL
sequence (train concatenated with the held-out test, ds made continuous), not
the training portion alone. We therefore hold out the last ``horizon`` points
per series as the test set — which, by construction of that loader, is exactly
the official M4 test split. Using the raw ``M4.load`` output as training data
would leak the test period.

Usage:
    uv run python scripts/download_m4.py                 # all frequencies
    uv run python scripts/download_m4.py --frequency Hourly
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from prophet.config import settings
from prophet.data.loaders import M4Frequency
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

# Fixed synthetic epoch. The absolute value is irrelevant to scale-free metrics;
# only per-series ordering and spacing matter.
_EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)

# polars `offset_by` unit per M4 frequency.
_OFFSET_UNIT: dict[str, str] = {
    "Hourly": "h",
    "Daily": "d",
    "Weekly": "w",
    "Monthly": "mo",
    "Quarterly": "q",
    "Yearly": "y",
}


def _synthesise_ds(offset_col: str, unit: str) -> pl.Expr:
    """Build a Datetime(us, UTC) column from an integer step offset."""
    return (
        pl.lit(_EPOCH)
        .dt.offset_by(pl.format("{}" + unit, pl.col(offset_col)))
        .cast(pl.Datetime("us", "UTC"))
        .alias("ds")
    )


def _build_splits(cache_dir: Path, frequency: M4Frequency) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Download one M4 frequency and return (train, test) in long format.

    Args:
        cache_dir: Directory datasetsforecast downloads raw CSVs into.
        frequency: M4 frequency to build.

    Returns:
        (train, test) Polars DataFrames with columns (unique_id, ds, y).
    """
    # Imported lazily so the test suite never pays the import cost.
    from datasetsforecast.m4 import M4

    group = frequency.value
    unit = _OFFSET_UNIT[group]

    # M4.load returns the FULL series per unique_id (train + held-out test, with a
    # continuous integer ds starting at 1). We synthesise timestamps over the
    # whole series, then hold out the last `horizon` points as the test split.
    full_pd, _, _ = M4.load(directory=str(cache_dir), group=group)
    full = (
        pl.from_pandas(full_pd)
        .select(
            pl.col("unique_id").cast(pl.String),
            pl.col("ds").cast(pl.Int64),
            pl.col("y").cast(pl.Float64),
        )
        .sort(["unique_id", "ds"])
        # Integer ds is 1-based, so the step offset from the epoch is ds - 1.
        .with_columns((pl.col("ds") - 1).alias("offset"))
        .with_columns(_synthesise_ds("offset", unit))
        .select("unique_id", "ds", "y")
    )

    # Holding out the last `horizon` per series reproduces the official M4 test
    # split exactly (see module docstring) with no train/test overlap.
    train, test = split_train_test(full, horizon=frequency.horizon)
    return train, test


@app.command()
def main(
    frequency: str | None = typer.Option(
        None,
        help="Single frequency to download (Hourly, Daily, etc.). Default: all.",
    ),
) -> None:
    """Download M4 data and save as Parquet."""
    target_dir = settings.data_raw / "m4"
    cache_dir = settings.data_raw / "_m4cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frequencies = [M4Frequency(frequency)] if frequency else list(M4Frequency)

    console.print(f"[bold]Target directory:[/bold] {target_dir}")
    console.print(f"[bold]Frequencies:[/bold] {[f.value for f in frequencies]}\n")

    summary = Table(title="M4 download summary")
    summary.add_column("Frequency")
    summary.add_column("Series", justify="right")
    summary.add_column("Train rows", justify="right")
    summary.add_column("Test rows", justify="right")
    summary.add_column("Horizon", justify="right")

    for freq in frequencies:
        console.print(f"[cyan]Downloading {freq.value}...[/cyan]")
        train, test = _build_splits(cache_dir, freq)

        train_path = target_dir / f"{freq.value.lower()}-train.parquet"
        test_path = target_dir / f"{freq.value.lower()}-test.parquet"
        train.write_parquet(train_path)
        test.write_parquet(test_path)

        n_series = train["unique_id"].n_unique()
        test_h = test.height // n_series if n_series else 0
        summary.add_row(
            freq.value,
            str(n_series),
            f"{train.height:,}",
            f"{test.height:,}",
            str(test_h),
        )

    console.print()
    console.print(summary)
    console.print(f"\n[green]Done.[/green] Parquet written under {target_dir}")


if __name__ == "__main__":
    app()
