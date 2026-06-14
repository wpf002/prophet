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
# Forecastable defaults (trend + seasonality + mean-reversion), not random walks.
DEFAULT_SERIES = ["CPIAUCSL", "UNRATE"]


def _read_fred(series: str) -> pl.DataFrame:
    """READ-ONLY pull of one FRED series as long format (unique_id, ds, y)."""
    raw = urllib.request.urlopen(FRED_CSV.format(series=series), timeout=30).read().decode()
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


@app.command()
def main(
    series: Annotated[
        list[str] | None, typer.Option(help="FRED series ids (default: CPI, UNRATE).")
    ] = None,
) -> None:
    """Build macro-train/test Parquet from FRED public CSV (read-only)."""
    spec = DOMAIN_SPECS[NAME]
    series = series or DEFAULT_SERIES
    panel = pl.concat([_read_fred(s) for s in series]).sort(["unique_id", "ds"])

    # FRED occasionally omits a month (a "." placeholder, e.g. a delayed release).
    # Reindex each series to a complete month-start grid and linearly interpolate
    # the gaps so the level series is regular for "MS"-frequency models.
    panel = (
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
    console.print(
        f"[bold]Series:[/bold] {panel['unique_id'].n_unique()} ({', '.join(series)}), "
        f"{panel.height} points"
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
