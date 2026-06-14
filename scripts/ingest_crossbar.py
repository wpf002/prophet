"""Phase-5 connector — Crossbar prediction-market prices -> domain Parquet.

Reads the YES-contract trade price per market from Crossbar's Postgres
(READ-ONLY) and writes a Prophet domain dataset in long format
(unique_id = marketId, ds = synthetic per-trade step [Datetime us UTC],
y = price 1-99 = implied probability). The model ladder then runs via
``run_benchmark.py --dataset domain-crossbar``.

This is strictly read-only against Crossbar — it never writes to or alters the
Crossbar database or app. Set ``CROSSBAR_DSN`` (a read-only connection string)
or pass ``--dsn``.

``--synthetic`` generates a clearly-labeled stand-in (realistic prediction-market
price walks) to prove the pipeline when no DB is reachable. Synthetic data is
for wiring/verification only, never a forecastability verdict.

Usage:
    CROSSBAR_DSN=postgresql://... uv run python scripts/ingest_crossbar.py
    uv run python scripts/ingest_crossbar.py --synthetic   # pipeline proof
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

_EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
NAME = "crossbar"


def _read_trades(dsn: str) -> pl.DataFrame:
    """READ-ONLY pull of YES-contract trades: (unique_id, created_at, price)."""
    import psycopg

    query = (
        'SELECT "marketId" AS unique_id, "createdAt" AS created_at, price '
        'FROM "Trade" WHERE outcome = \'YES\' ORDER BY "marketId", "createdAt"'
    )
    with (
        psycopg.connect(dsn, connect_timeout=15, autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(query)
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=["unique_id", "created_at", "price"], orient="row")


def _synthetic_trades(n_markets: int = 30, n_trades: int = 300, seed: int = 42) -> pl.DataFrame:
    """Labeled synthetic prediction-market YES-price walks (pipeline proof only)."""
    rng = np.random.default_rng(seed)
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    rows: list[tuple[str, dt.datetime, int]] = []
    for m in range(n_markets):
        resolution = float(rng.integers(0, 2)) * 100.0  # market drifts toward 0 or 100
        price = float(rng.integers(35, 66))
        t = base
        for _ in range(n_trades):
            drift = 0.01 * (resolution - price)  # converge toward resolution
            price = float(np.clip(price + drift + rng.normal(0, 2.0), 1, 99))
            t += dt.timedelta(minutes=rng.integers(1, 30).item())
            rows.append((f"SYNTH-{m:03d}", t, round(price)))
    return pl.DataFrame(rows, schema=["unique_id", "created_at", "price"], orient="row")


def _to_long(trades: pl.DataFrame) -> pl.DataFrame:
    """Reindex each market's trades to a regular per-step sequence (synthetic ds)."""
    return (
        trades.with_columns(
            pl.col("unique_id").cast(pl.String),
            pl.col("price").cast(pl.Float64),
        )
        .sort(["unique_id", "created_at"])
        .with_columns(pl.int_range(pl.len()).over("unique_id").alias("step"))
        .with_columns(
            (pl.lit(_EPOCH) + pl.duration(hours=pl.col("step")))
            .cast(pl.Datetime("us", "UTC"))
            .alias("ds")
        )
        .select("unique_id", "ds", pl.col("price").alias("y"))
        .sort(["unique_id", "ds"])
    )


@app.command()
def main(
    dsn: str | None = typer.Option(None, help="Read-only Crossbar DSN. Default: env CROSSBAR_DSN."),
    synthetic: bool = typer.Option(False, help="Generate labeled synthetic data (pipeline proof)."),
    min_trades: int = typer.Option(40, help="Drop markets with fewer than this many trades."),
) -> None:
    """Build crossbar-train/test Parquet from Crossbar trades (read-only) or synthetic."""
    spec = DOMAIN_SPECS[NAME]
    if synthetic:
        console.print("[yellow]Synthetic mode — pipeline proof only, not a real verdict.[/yellow]")
        trades = _synthetic_trades()
    else:
        dsn = dsn or os.environ.get("CROSSBAR_DSN")
        if not dsn:
            raise typer.BadParameter("Set CROSSBAR_DSN / --dsn, or use --synthetic.")
        trades = _read_trades(dsn)
    console.print(
        f"[bold]Trades:[/bold] {trades.height} across {trades['unique_id'].n_unique()} markets"
    )

    panel = _to_long(trades)
    counts = panel.group_by("unique_id").len()
    keep = counts.filter(pl.col("len") >= min_trades)["unique_id"]
    panel = panel.filter(pl.col("unique_id").is_in(keep))
    if panel["unique_id"].n_unique() == 0:
        raise typer.Exit(code=1)

    train, test = split_train_test(panel, horizon=spec.horizon)
    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / f"{NAME}-train.parquet")
    test.write_parquet(out_dir / f"{NAME}-test.parquet")
    console.print(
        f"[green]Wrote[/green] {panel['unique_id'].n_unique()} markets | "
        f"train {train.height}, test {test.height} -> {out_dir}"
    )
    console.print(
        "Run: uv run python scripts/run_benchmark.py --dataset domain-crossbar --models statistical"
    )


if __name__ == "__main__":
    app()
