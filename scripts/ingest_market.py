"""Phase 5 connector — Alpaca daily bars -> market domain Parquet.

Pulls daily OHLCV for the portfolio's tickers from Alpaca and writes two domain
datasets in Nixtla long format (``unique_id`` = ticker, ``ds`` = synthetic daily
[Datetime us UTC], ``y`` = Int64):

- ``market-close`` : close price in cents (money invariant).
- ``market-vol``   : daily trading volume.

Trading days are reindexed to a regular daily sequence (weekend/holiday gaps
dropped, M4-style) so the series are evenly spaced and the 5-day trading week is
the seasonal period. Each dataset holds out the last ``horizon`` steps as test.

Tickers default to the live Syntrackr ``positions`` table (via
``PROPHET_CASHFLOW_DSN``); pass ``--tickers AAPL,MSFT,...`` to override.
Alpaca credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

Usage:
    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... \\
        uv run python scripts/ingest_market.py --start 2019-01-01
"""

from __future__ import annotations

import datetime as dt
import os

import httpx
import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)


def _portfolio_tickers() -> list[str]:
    """Distinct tickers from the live Syntrackr positions table."""
    import psycopg

    dsn = os.environ.get("PROPHET_CASHFLOW_DSN")
    if not dsn:
        raise typer.BadParameter("Set PROPHET_CASHFLOW_DSN or pass --tickers.")
    with (
        psycopg.connect(dsn, connect_timeout=10, autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SET default_transaction_read_only = on")
        cur.execute("select distinct ticker from positions order by ticker")
        return [r[0] for r in cur.fetchall()]


def _fetch_bars(tickers: list[str], start: str, end: str, feed: str) -> pl.DataFrame:
    """Fetch daily bars for all tickers (paginated). Returns (unique_id, date, close, volume)."""
    headers = {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
    }
    records: list[tuple[str, str, float, int]] = []
    page_token: str | None = None
    while True:
        params: dict[str, str | int] = {
            "symbols": ",".join(tickers),
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "adjustment": "all",
            "feed": feed,
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token
        resp = httpx.get(_BARS_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for sym, bars in payload.get("bars", {}).items():
            for bar in bars:
                records.append((sym, bar["t"][:10], float(bar["c"]), int(bar["v"])))
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return pl.DataFrame(records, schema=["unique_id", "date", "close", "volume"], orient="row")


def _to_long(df: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    """Reindex trading days to a regular daily sequence and build long format."""
    seq = (
        df.with_columns(pl.col("date").str.to_date())
        .sort(["unique_id", "date"])
        .with_columns(pl.int_range(pl.len()).over("unique_id").alias("step"))
        .with_columns(
            (pl.lit(_EPOCH) + pl.duration(days=pl.col("step")))
            .cast(pl.Datetime("us", "UTC"))
            .alias("ds")
        )
    )
    return seq.select("unique_id", "ds", value.alias("y")).sort(["unique_id", "ds"])


def _write(panel: pl.DataFrame, name: str) -> tuple[int, int]:
    horizon = DOMAIN_SPECS[name].horizon
    train, test = split_train_test(panel, horizon=horizon)
    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / f"{name}-train.parquet")
    test.write_parquet(out_dir / f"{name}-test.parquet")
    return train.height, test.height


@app.command()
def main(
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers. Default: live Syntrackr positions."
    ),
    start: str = typer.Option("2019-01-01", help="History start date (YYYY-MM-DD)."),
    end: str = typer.Option("2025-06-01", help="History end date (YYYY-MM-DD)."),
    feed: str = typer.Option("iex", help="Alpaca data feed (iex on the free plan)."),
) -> None:
    """Build market-close and market-vol domain Parquet from Alpaca daily bars."""
    symbols = [t.strip().upper() for t in tickers.split(",")] if tickers else _portfolio_tickers()
    console.print(f"[bold]Tickers ({len(symbols)}):[/bold] {', '.join(symbols)}")

    bars = _fetch_bars(symbols, start=start, end=end, feed=feed)
    console.print(
        f"[bold]Fetched:[/bold] {bars.height} daily bars across {bars['unique_id'].n_unique()} tickers"
    )

    close_panel = _to_long(bars, (pl.col("close") * 100).round(0).cast(pl.Int64))
    vol_panel = _to_long(bars, pl.col("volume").cast(pl.Int64))

    ctr, cte = _write(close_panel, "market-close")
    vtr, vte = _write(vol_panel, "market-vol")
    console.print(f"[green]market-close[/green]: train {ctr}, test {cte}")
    console.print(f"[green]market-vol[/green]:   train {vtr}, test {vte}")
    console.print(
        "Run: uv run python scripts/run_benchmark.py --dataset domain-market-close --models ml"
    )


if __name__ == "__main__":
    app()
