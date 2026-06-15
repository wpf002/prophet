"""Connector — FRED macro series (via Bloomberg) -> domain Parquet.

Pulls monthly macro series from FRED's PUBLIC CSV endpoint (no API key, no auth)
into the Nixtla long format (unique_id = FRED series id, ds = Datetime us UTC,
y = level). These are the series Bloomberg already surfaces
(``backend/data/sources/fred_source.py``); Prophet forecasts them so Bloomberg
can consume the projection via ``@prophet/client``.

Defaults to the trending / mean-reverting series that beat a naive forecast
(CPI, unemployment). The random-walk FRED series (DGS10, VIXCLS, DCOILWTICO, …)
are intentionally excluded — they fail the "beats naive" test, same as raw
stock prices.

Usage:
    uv run python scripts/ingest_macro.py
    uv run python scripts/ingest_macro.py --series CPIAUCSL --series UNRATE
"""

from __future__ import annotations

import io
import urllib.request
from typing import Annotated

import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

NAME = "macro"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# Forecastable defaults — each verified to beat a naive forecast by >5% MASE
# (trend + seasonality + mean-reversion), not random walks. CPI/unemployment are
# the originals; retail sales, PCE price index, and vehicle sales were added
# after the same filter (PAYEMS/INDPRO/HOUST dropped — naive wins or ties).
DEFAULT_SERIES = ["CPIAUCSL", "UNRATE", "RSAFS", "PCEPI", "TOTALSA"]


def _read_fred(series: str, retries: int = 3) -> pl.DataFrame:
    """READ-ONLY pull of one FRED series as long format (unique_id, ds, y).

    Retries on transient network errors — important on container boot, where a
    single slow FRED response must not abort the whole macro build.
    """
    raw = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(FRED_CSV.format(series=series), timeout=30) as resp:
                raw = resp.read().decode()
            break
        except OSError:  # timeouts, connection resets
            if attempt == retries - 1:
                raise
    df = pl.read_csv(io.StringIO(raw))
    df.columns = ["ds", "y"]
    return (
        df.with_columns(
            pl.col("ds")
            .str.to_datetime("%Y-%m-%d")
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us"),
            pl.col("y").cast(pl.Float64, strict=False),
        )
        .drop_nulls()
        .with_columns(pl.lit(series).alias("unique_id"))
        .select("unique_id", "ds", "y")
    )


def build_macro_panel(series: list[str] | None = None) -> pl.DataFrame:
    """Fetch the macro series from FRED and return a clean long-format panel.

    Skips any series that still fails after retries (one slow FRED endpoint must
    not abort the build), then reindexes each series to a complete month-start
    grid and linearly interpolates FRED's occasional missing month. Reused by the
    monthly monitor/retrain job.
    """
    series = series or DEFAULT_SERIES
    frames: list[pl.DataFrame] = []
    for s in series:
        try:
            frames.append(_read_fred(s))
        except OSError as exc:
            console.print(f"[yellow]Skipping {s} — FRED fetch failed: {exc}[/yellow]")
    if not frames:
        raise typer.Exit(code=1)
    panel = pl.concat(frames).sort(["unique_id", "ds"])
    return (
        panel.group_by("unique_id")
        .agg(pl.col("ds").min().alias("lo"), pl.col("ds").max().alias("hi"))
        .with_columns(
            pl.datetime_ranges("lo", "hi", interval="1mo", time_unit="us", time_zone="UTC").alias(
                "ds"
            )
        )
        .explode("ds")
        .select("unique_id", "ds")
        .join(panel, on=["unique_id", "ds"], how="left")
        .with_columns(pl.col("y").interpolate().over("unique_id"))
        .sort(["unique_id", "ds"])
    )


@app.command()
def main(
    series: Annotated[
        list[str] | None, typer.Option(help="FRED series ids (default: CPI, UNRATE).")
    ] = None,
) -> None:
    """Build macro-train/test Parquet from FRED public CSV (read-only)."""
    spec = DOMAIN_SPECS[NAME]
    panel = build_macro_panel(series)
    console.print(
        f"[bold]Series:[/bold] {panel['unique_id'].n_unique()} "
        f"({', '.join(panel['unique_id'].unique().sort().to_list())}), {panel.height} points"
    )

    train, test = split_train_test(panel, horizon=spec.horizon)
    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / f"{NAME}-train.parquet")
    test.write_parquet(out_dir / f"{NAME}-test.parquet")
    console.print(f"[green]Wrote[/green] train {train.height}, test {test.height} -> {out_dir}")
    console.print("Run: uv run python scripts/run_benchmark.py --dataset domain-macro")


if __name__ == "__main__":
    app()
