"""Market data fetch + reindex (Alpaca daily bars).

Shared by the ingest connector and the actuals job so both produce the *same*
``(unique_id, ds, y)`` representation: trading days reindexed to a regular daily
sequence from a fixed epoch, ``y`` as Int64 (close in cents, or volume).
"""

from __future__ import annotations

import datetime as dt
import os

import httpx
import polars as pl

_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
EPOCH = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)


def fetch_bars(tickers: list[str], *, start: str, end: str, feed: str = "iex") -> pl.DataFrame:
    """Fetch daily bars for all tickers (paginated). Columns: unique_id, date, close, volume."""
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


def to_long(bars: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    """Reindex trading days to a regular daily sequence and build long format."""
    seq = (
        bars.with_columns(pl.col("date").str.to_date())
        .sort(["unique_id", "date"])
        .with_columns(pl.int_range(pl.len()).over("unique_id").alias("step"))
        .with_columns(
            (pl.lit(EPOCH) + pl.duration(days=pl.col("step")))
            .cast(pl.Datetime("us", "UTC"))
            .alias("ds")
        )
    )
    return seq.select("unique_id", "ds", value.alias("y")).sort(["unique_id", "ds"])


# Target name -> value expression (Int64), shared by ingest and actuals.
TARGETS: dict[str, pl.Expr] = {
    "market-close": (pl.col("close") * 100).round(0).cast(pl.Int64),
    "market-vol": pl.col("volume").cast(pl.Int64),
}
