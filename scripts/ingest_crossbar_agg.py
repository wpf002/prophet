"""Phase-5 connector — Crossbar PLATFORM-LEVEL hourly activity -> domain Parquet.

Where ``ingest_crossbar.py`` builds *per-market* series from the public API, this
builds the *platform aggregate*: total traded volume and trade count per hour
across ALL markets. That's the forecastable, decision-relevant target (load /
liquidity provisioning), and it needs the full ``Trade`` history — deeper than the
public API's 168-hour candle window — so it reads the read-only Postgres directly.

Read-only throughout: opens the connection with
``SET default_transaction_read_only = on`` and only ever runs a single grouped
SELECT. Never writes to or alters the Crossbar database.

Two series are emitted in Nixtla long format ``(unique_id, ds, y)``:

* ``platform_volume`` — SUM(quantity) per hour
* ``platform_trades`` — COUNT(*) per hour

Both are gap-filled to a regular hourly grid (missing hours = 0 activity), then
split into train/test with ``split_train_test(horizon=24)`` and written to
``data/raw/domains/crossbar-agg-{train,test}.parquet``.

Usage:
    # DSN from Railway (read-only): railway variables --service Postgres --kv
    CROSSBAR_DSN=postgresql://... uv run python scripts/ingest_crossbar_agg.py
"""

from __future__ import annotations

import os

import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

NAME = "crossbar-agg"

# Hourly platform totals across every market. date_trunc keeps this cheap on the
# server; we gap-fill the grid client-side. createdAt is stored naive — it is
# wall-clock UTC, localized as such below.
_AGG_QUERY = (
    "SELECT date_trunc('hour', \"createdAt\") AS hour, "
    "SUM(quantity)::bigint AS volume, COUNT(*)::bigint AS trades "
    'FROM "Trade" GROUP BY 1 ORDER BY 1'
)


def _read_hourly(dsn: str) -> pl.DataFrame:
    """READ-ONLY pull of hourly (hour, volume, trades) platform totals."""
    import psycopg

    with (
        psycopg.connect(dsn, connect_timeout=20, autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(_AGG_QUERY)
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=["hour", "volume", "trades"], orient="row")


def _to_gap_filled_panel(hourly: pl.DataFrame) -> pl.DataFrame:
    """Regular hourly grid (missing hours = 0) -> two long series."""
    hourly = hourly.with_columns(
        pl.col("hour").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
        pl.col("volume").cast(pl.Float64),
        pl.col("trades").cast(pl.Float64),
    ).sort("hour")

    grid = pl.datetime_range(
        hourly["hour"].min(),
        hourly["hour"].max(),
        interval="1h",
        time_zone="UTC",
        time_unit="us",
        eager=True,
    ).alias("ds")

    filled = (
        pl.DataFrame({"ds": grid})
        .join(hourly.rename({"hour": "ds"}), on="ds", how="left")
        .with_columns(pl.col("volume").fill_null(0.0), pl.col("trades").fill_null(0.0))
    )

    volume = filled.select("ds", pl.col("volume").alias("y")).with_columns(
        pl.lit("platform_volume").alias("unique_id")
    )
    trades = filled.select("ds", pl.col("trades").alias("y")).with_columns(
        pl.lit("platform_trades").alias("unique_id")
    )
    return pl.concat([volume, trades]).select("unique_id", "ds", "y").sort(["unique_id", "ds"])


@app.command()
def main(
    dsn: str | None = typer.Option(None, help="Read-only Crossbar DSN. Default: env CROSSBAR_DSN."),
) -> None:
    """Build crossbar-agg-{train,test} Parquet from the read-only Crossbar DB."""
    dsn = dsn or os.environ.get("CROSSBAR_DSN")
    if not dsn:
        raise typer.BadParameter(
            "Set CROSSBAR_DSN (railway variables --service Postgres) or --dsn."
        )

    spec = DOMAIN_SPECS[NAME]
    hourly = _read_hourly(dsn)
    console.print(
        f"[bold]Hourly buckets:[/bold] {hourly.height} "
        f"({hourly['hour'].min()} -> {hourly['hour'].max()})"
    )
    panel = _to_gap_filled_panel(hourly)
    console.print(
        f"[bold]Series:[/bold] {panel['unique_id'].n_unique()} "
        f"({panel.height} points on a gap-filled grid)"
    )

    train, test = split_train_test(panel, horizon=spec.horizon)
    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / f"{NAME}-train.parquet")
    test.write_parquet(out_dir / f"{NAME}-test.parquet")
    console.print(
        f"[green]Wrote[/green] train {train.height}, test {test.height} -> {out_dir}\n"
        "Run: uv run python scripts/run_benchmark.py --dataset domain-crossbar-agg --models statistical"
    )


if __name__ == "__main__":
    app()
