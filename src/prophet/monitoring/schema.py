"""Forecast-vs-actual logging schema (Phase 6).

Two tables back the accuracy dashboard and drift alerts:

- ``forecasts`` — every forecast the API serves (point + 95% interval).
- ``actuals``   — the realized value once it lands.

A nightly job joins them on ``(series_id, ds)`` to compute rolling accuracy.
Money/volume values are stored as integers (project invariant).
"""

from __future__ import annotations

from typing import Any

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
    id            BIGSERIAL PRIMARY KEY,
    series_id     TEXT        NOT NULL,
    ds            TIMESTAMPTZ NOT NULL,
    horizon       INTEGER     NOT NULL,
    model         TEXT        NOT NULL,
    y_hat         DOUBLE PRECISION NOT NULL,
    y_lo_95       DOUBLE PRECISION,
    y_hi_95       DOUBLE PRECISION,
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS forecasts_series_ds_idx ON forecasts (series_id, ds);

CREATE TABLE IF NOT EXISTS actuals (
    series_id    TEXT        NOT NULL,
    ds           TIMESTAMPTZ NOT NULL,
    y            DOUBLE PRECISION NOT NULL,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, ds)
);
"""

# Rolling accuracy over the last N days: mean absolute error and 95% interval
# coverage of forecasts whose target date has a matching actual.
ROLLING_ACCURACY_SQL = """
SELECT
    f.series_id,
    count(*)                                         AS n,
    avg(abs(f.y_hat - a.y))                          AS mae,
    avg(((a.y BETWEEN f.y_lo_95 AND f.y_hi_95))::int)::float AS coverage_95
FROM forecasts f
JOIN actuals a ON a.series_id = f.series_id AND a.ds = f.ds
WHERE a.ds >= now() - (%(days)s || ' days')::interval
GROUP BY f.series_id
ORDER BY f.series_id;
"""


def create_monitoring_tables(conn: Any) -> None:
    """Create the forecasts/actuals tables (idempotent). conn is a psycopg connection."""
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
    conn.commit()
