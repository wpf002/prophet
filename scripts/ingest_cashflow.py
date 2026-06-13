"""Phase 5 connector — Syntrackr cash flow -> domain Parquet.

Materializes daily net cash flow per category into
``data/raw/domains/cashflow-{train,test}.parquet`` (Nixtla long format,
``unique_id`` = category, ``ds`` = day [Datetime us UTC], ``y`` = net cents),
then the model ladder runs via ``run_benchmark.py --dataset domain-cashflow``.

Two input modes:
- **Postgres** (default): set ``PROPHET_CASHFLOW_DSN`` or pass ``--dsn``.
- **CSV export**: pass ``--csv path.csv`` (same column names as the table).

The table/column names below are assumptions — override with the flags to match
the real Syntrackr schema.

Usage:
    PROPHET_CASHFLOW_DSN=postgresql://... uv run python scripts/ingest_cashflow.py
    uv run python scripts/ingest_cashflow.py --csv data/raw/domains/cashflow.csv \\
        --amount-units dollars --category-col category --date-col ts --amount-col amount
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()


def _read_source(
    *,
    dsn: str | None,
    csv: Path | None,
    table: str,
    category_col: str,
    date_col: str,
    amount_col: str,
) -> pl.DataFrame:
    """Return raw rows with columns (category, ts, amount)."""
    select = f"{category_col} AS category, {date_col} AS ts, {amount_col} AS amount"
    if csv is not None:
        raw = pl.read_csv(csv)
        return raw.select(
            pl.col(category_col).alias("category"),
            pl.col(date_col).alias("ts"),
            pl.col(amount_col).alias("amount"),
        )
    if not dsn:
        raise typer.BadParameter("Provide --csv or set PROPHET_CASHFLOW_DSN / --dsn.")

    # sqlalchemy + psycopg; normalize a bare postgresql:// URL to the psycopg driver.
    from sqlalchemy import create_engine

    uri = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(uri)
    with engine.connect() as conn:
        return pl.read_database(f"SELECT {select} FROM {table}", connection=conn)


@app.command()
def main(
    dsn: str | None = typer.Option(
        None, help="Postgres DSN. Defaults to env PROPHET_CASHFLOW_DSN."
    ),
    csv: str | None = typer.Option(None, help="CSV export to ingest instead of a DB."),
    table: str = typer.Option("transactions", help="Source table name (Postgres mode)."),
    category_col: str = typer.Option("category", help="Column for the series id."),
    date_col: str = typer.Option("ts", help="Column for the transaction date/timestamp."),
    amount_col: str = typer.Option("amount", help="Column for the signed amount."),
    amount_units: str = typer.Option("cents", help="Source amount units: cents or dollars."),
) -> None:
    """Build cashflow-train/test Parquet from Postgres or a CSV export."""
    dsn = dsn or os.environ.get("PROPHET_CASHFLOW_DSN")
    spec = DOMAIN_SPECS["cashflow"]

    raw = _read_source(
        dsn=dsn,
        csv=Path(csv) if csv else None,
        table=table,
        category_col=category_col,
        date_col=date_col,
        amount_col=amount_col,
    )
    console.print(f"[bold]Raw rows:[/bold] {raw.height}")

    # To integer cents (project invariant) and a UTC day key. Handle ts arriving
    # as a string (CSV), naive datetime, or tz-aware datetime (Postgres timestamptz).
    ts_dtype = raw.schema["ts"]
    if ts_dtype == pl.String:
        ts = pl.col("ts").str.to_datetime(time_unit="us", strict=False).dt.replace_time_zone("UTC")
    elif isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        ts = pl.col("ts").cast(pl.Datetime("us", ts_dtype.time_zone)).dt.convert_time_zone("UTC")
    else:
        ts = pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")

    cents = pl.col("amount") if amount_units == "cents" else pl.col("amount") * 100
    tidy = raw.with_columns(
        pl.col("category").cast(pl.String).alias("unique_id"),
        ts.dt.truncate("1d").alias("ds"),
        cents.round(0).cast(pl.Int64).alias("amount_cents"),
    )

    # Net cents per category per day.
    daily = tidy.group_by(["unique_id", "ds"]).agg(pl.col("amount_cents").sum().alias("y"))

    # Per-series upsample to a complete daily grid — no-transaction days are real
    # zeros, and the models need a regular frequency.
    grids: list[pl.DataFrame] = []
    for uid in daily["unique_id"].unique().sort():
        series = daily.filter(pl.col("unique_id") == uid).sort("ds")
        full = series.upsample("ds", every="1d").with_columns(
            pl.col("unique_id").fill_null(uid), pl.col("y").fill_null(0)
        )
        grids.append(full)
    panel = pl.concat(grids).select("unique_id", "ds", "y")

    train, test = split_train_test(panel, horizon=spec.horizon)

    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / "cashflow-train.parquet")
    test.write_parquet(out_dir / "cashflow-test.parquet")

    console.print(
        f"[green]Wrote[/green] {train['unique_id'].n_unique()} series | "
        f"train {train.height} rows, test {test.height} rows -> {out_dir}"
    )
    console.print(
        "Run: uv run python scripts/run_benchmark.py --dataset domain-cashflow --models ml"
    )


if __name__ == "__main__":
    app()
