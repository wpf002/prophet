"""Forecast-vs-actual store — write served forecasts and realized values.

Row-shaping helpers are pure (and unit-tested); the DB functions wrap psycopg.
All are no-ops-friendly: callers should skip them when no monitor DSN is set.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from prophet.monitoring.schema import ROLLING_ACCURACY_SQL

ForecastRow = tuple[str, datetime, int, str, float, float | None, float | None]
ActualRow = tuple[str, datetime, float]


def forecast_rows(
    series_id: str,
    model: str,
    horizon: int,
    points: list[Any],
) -> list[ForecastRow]:
    """Shape served forecast points into ``forecasts`` rows (95% bounds only)."""
    rows: list[ForecastRow] = []
    for p in points:
        lo95 = p.lo.get("95") if getattr(p, "lo", None) else None
        hi95 = p.hi.get("95") if getattr(p, "hi", None) else None
        rows.append((series_id, p.ds, horizon, model, float(p.y_hat), lo95, hi95))
    return rows


def log_forecasts(dsn: str, rows: list[ForecastRow]) -> int:
    """Insert forecast rows. Returns the number written."""
    if not rows:
        return 0
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO forecasts "
            "(series_id, ds, horizon, model, y_hat, y_lo_95, y_hi_95) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()
    return len(rows)


def upsert_actuals(dsn: str, rows: list[ActualRow]) -> int:
    """Upsert realized values into ``actuals`` (idempotent on series_id, ds)."""
    if not rows:
        return 0
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO actuals (series_id, ds, y) VALUES (%s, %s, %s) "
            "ON CONFLICT (series_id, ds) DO UPDATE SET y = EXCLUDED.y, recorded_at = now()",
            rows,
        )
        conn.commit()
    return len(rows)


def rolling_accuracy(dsn: str, days: int = 30) -> list[dict[str, Any]]:
    """Per-series MAE and 95% interval coverage over the last ``days``."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(ROLLING_ACCURACY_SQL, {"days": days})
        cols = [c.name for c in (cur.description or [])]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
