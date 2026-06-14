"""Phase-5 connector — Crossbar prediction-market prices -> domain Parquet.

Materializes the YES-contract price series per market into a Prophet domain
dataset in long format (unique_id = marketId, ds = Datetime us UTC,
y = price 1-99 = implied probability). The model ladder then runs unchanged via
``run_benchmark.py --dataset domain-crossbar``.

Three read-only sources, in order of preference:

* ``--api`` (default base ``https://crossbar.fly.dev``) — pulls the PUBLIC,
  unauthenticated ``/markets`` list and each market's ``/candles`` (YES trades
  already bucketed into a regular time grid server-side). No credentials, no DB
  access. This is the recommended path for a real verdict.
* ``CROSSBAR_DSN`` / ``--dsn`` — a read-only Postgres connection
  (``default_transaction_read_only = on``); reindexes raw trades to a regular
  per-market step. Never writes to or alters the Crossbar database.
* ``--synthetic`` — a clearly-labeled stand-in (realistic price walks) to prove
  the pipeline when no source is reachable. Never a forecastability verdict.

Usage:
    uv run python scripts/ingest_crossbar.py --api          # real, no creds
    CROSSBAR_DSN=postgresql://... uv run python scripts/ingest_crossbar.py
    uv run python scripts/ingest_crossbar.py --synthetic    # pipeline proof
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request

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
DEFAULT_API = "https://crossbar.fly.dev"


def _get_json(url: str, timeout: int = 20) -> object:
    """GET and parse JSON from a public read-only endpoint."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _read_candles_api(
    base_url: str, top_n: int, hours: int, bucket_ms: int, target: str = "close"
) -> pl.DataFrame:
    """READ-ONLY pull of bucketed YES candles via Crossbar's public API.

    Returns a long panel (unique_id, ds [Datetime us UTC], y) built from the
    busiest ``top_n`` open markets. ds is a real wall-clock grid (candle bucket
    timestamps), so no synthetic reindexing is needed. ``target`` selects the y
    column: ``close`` (YES price = implied probability) or ``volume`` (traded
    quantity per bucket — the Phase-5-style forecastable activity signal).
    """
    key = {"close": "c", "volume": "v"}.get(target)
    if key is None:
        raise typer.BadParameter("target must be 'close' or 'volume'.")
    base = base_url.rstrip("/")
    markets = _get_json(f"{base}/markets")
    if not isinstance(markets, list):
        raise typer.BadParameter(f"Unexpected /markets response from {base}.")
    markets = sorted(markets, key=lambda m: m.get("volume24h", 0) or 0, reverse=True)[:top_n]

    rows: list[tuple[str, dt.datetime, float]] = []
    for m in markets:
        mid = m["id"]
        try:
            payload = _get_json(f"{base}/markets/{mid}/candles?bucket={bucket_ms}&hours={hours}")
        except OSError:
            continue
        for c in payload.get("candles", []):  # type: ignore[union-attr]
            ts = dt.datetime.fromtimestamp(int(c["t"]) / 1000, tz=dt.UTC)
            rows.append((mid, ts, float(c[key])))

    return (
        pl.DataFrame(rows, schema=["unique_id", "ds", "y"], orient="row")
        .with_columns(pl.col("ds").dt.cast_time_unit("us"))
        .sort(["unique_id", "ds"])
    )


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
    api: bool = typer.Option(False, help="Pull real data via Crossbar's public read-only API."),
    base_url: str = typer.Option(DEFAULT_API, help="Crossbar API base URL (with --api)."),
    top_n: int = typer.Option(80, help="Busiest N open markets to pull (with --api)."),
    hours: int = typer.Option(168, help="Candle look-back window in hours (max 168)."),
    bucket_ms: int = typer.Option(3_600_000, help="Candle bucket size in ms (default 1h)."),
    target: str = typer.Option("close", help="Forecast target (with --api): close | volume."),
    dsn: str | None = typer.Option(None, help="Read-only Crossbar DSN. Default: env CROSSBAR_DSN."),
    synthetic: bool = typer.Option(False, help="Generate labeled synthetic data (pipeline proof)."),
    min_trades: int = typer.Option(40, help="Drop markets with fewer than this many points."),
) -> None:
    """Build crossbar-train/test Parquet from Crossbar (read-only API/DSN) or synthetic."""
    spec = DOMAIN_SPECS[NAME]
    if api:
        console.print(f"[bold]API mode[/bold] — public read-only pull from {base_url} (y={target})")
        panel = _read_candles_api(base_url, top_n, hours, bucket_ms, target)
    else:
        if synthetic:
            console.print("[yellow]Synthetic mode — pipeline proof only, not a verdict.[/yellow]")
            trades = _synthetic_trades()
        else:
            dsn = dsn or os.environ.get("CROSSBAR_DSN")
            if not dsn:
                raise typer.BadParameter("Use --api, set CROSSBAR_DSN / --dsn, or --synthetic.")
            trades = _read_trades(dsn)
        panel = _to_long(trades)

    console.print(
        f"[bold]Series:[/bold] {panel['unique_id'].n_unique()} markets, {panel.height} points"
    )
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
