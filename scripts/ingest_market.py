"""Phase 5 connector — Alpaca daily bars -> market domain Parquet.

Pulls daily OHLCV for the portfolio's tickers and writes two domain datasets in
Nixtla long format (``unique_id`` = ticker, ``ds`` = synthetic daily
[Datetime us UTC], ``y`` = Int64): ``market-close`` (close cents) and
``market-vol`` (volume). Fetch/reindex logic lives in ``prophet.data.market``.

Tickers default to the live Syntrackr ``positions`` table (via
``PROPHET_CASHFLOW_DSN``); pass ``--tickers AAPL,MSFT,...`` to override.
Alpaca credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

Usage:
    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... \\
        uv run python scripts/ingest_market.py --start 2019-01-01
"""

from __future__ import annotations

import os

import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.data.market import TARGETS, fetch_bars, to_long
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()


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


def _write(panel: pl.DataFrame, name: str) -> tuple[int, int]:
    train, test = split_train_test(panel, horizon=DOMAIN_SPECS[name].horizon)
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

    bars = fetch_bars(symbols, start=start, end=end, feed=feed)
    console.print(
        f"[bold]Fetched:[/bold] {bars.height} daily bars across {bars['unique_id'].n_unique()} tickers"
    )

    for name, value in TARGETS.items():
        tr, te = _write(to_long(bars, value), name)
        console.print(f"[green]{name}[/green]: train {tr}, test {te}")
    console.print(
        "Run: uv run python scripts/run_benchmark.py --dataset domain-market-close --models ml"
    )


if __name__ == "__main__":
    app()
