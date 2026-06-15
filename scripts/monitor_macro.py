"""Monthly macro monitor + auto-retrain job.

Operationalizes the served ``macro`` model. Each run:

1. Re-ingests the latest FRED data and retrains the AutoETS model (so new monthly
   prints are incorporated) — skip with ``--no-retrain``.
2. Upserts the latest realized values into ``actuals`` so accuracy can be scored
   against the forecasts the API logged.
3. Reports rolling forecast accuracy and a per-series input-drift (PSI) check.

FRED updates monthly, so schedule this monthly (Railway cron). Needs
PROPHET_MONITOR_DSN; the retrain step needs no credentials (public FRED).

Usage:
    uv run python scripts/monitor_macro.py
    uv run python scripts/monitor_macro.py --no-retrain --days 180
"""

from __future__ import annotations

import subprocess
import sys

import polars as pl
import psycopg
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import load_domain
from prophet.monitoring.schema import create_monitoring_tables
from prophet.monitoring.store import rolling_accuracy, upsert_actuals

app = typer.Typer(no_args_is_help=False)
console = Console()

DATASET = "macro"
ACTUALS_MONTHS = 18  # recent months to upsert (covers any logged forecast horizon)


def _retrain() -> None:
    """Re-ingest fresh FRED and retrain the macro model (same as boot)."""
    for cmd in (
        [sys.executable, "scripts/ingest_macro.py"],
        [
            sys.executable,
            "scripts/train_production.py",
            "--dataset",
            DATASET,
            "--method",
            "statistical",
            "--n-jobs",
            "1",
        ],
    ):
        subprocess.run(cmd, check=True)


@app.command()
def main(
    retrain: bool = typer.Option(True, help="Re-ingest FRED + retrain before scoring."),
    days: int = typer.Option(400, help="Rolling-accuracy window (days)."),
) -> None:
    """Retrain on fresh FRED, upsert actuals, and report accuracy + drift."""
    dsn = settings.monitor_dsn
    if not dsn:
        console.print("[red]PROPHET_MONITOR_DSN not set.[/red]")
        raise typer.Exit(code=2)

    if retrain:
        console.print("[bold]Retraining macro on fresh FRED…[/bold]")
        _retrain()

    train, test = load_domain(DATASET, settings.data_raw)
    panel = pl.concat([train, test]).sort(["unique_id", "ds"])
    series = panel["unique_id"].unique().sort().to_list()

    # Upsert recent realized values so they overlap the forecasts the API logged.
    rows = [
        (r["unique_id"], r["ds"], float(r["y"]))
        for sid in series
        for r in panel.filter(pl.col("unique_id") == sid).tail(ACTUALS_MONTHS).to_dicts()
    ]
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        create_monitoring_tables(conn)
    written = upsert_actuals(dsn, rows)
    console.print(f"[green]Upserted[/green] {written} actuals across {len(series)} series.")

    # Monitoring signal = accuracy + interval calibration as forecasts and actuals
    # accumulate. (Input-distribution PSI is deliberately NOT used here: at monthly
    # frequency there are far too few recent points for a stable PSI, and macro
    # levels trend so PSI on the level fires perpetually. Calibration — do realized
    # values land inside the 95% interval ~95% of the time — is the right check.)
    report = rolling_accuracy(dsn, days=days)
    if not report:
        console.print(f"[yellow]No forecast/actual overlap in the last {days}d yet.[/yellow]")
        console.print("[dim]Accuracy accrues once logged forecasts reach their target month.[/dim]")
        return
    console.print(f"[bold]Rolling accuracy + calibration (last {days}d):[/bold]")
    for row in report[:20]:
        cov = row["coverage_95"]
        flag = "[red]MISCALIBRATED[/red]" if row["n"] >= 6 and cov < 0.80 else "ok"
        console.print(
            f"  {row['series_id']:9s} n={row['n']:>4} mae={row['mae']:,.3f} "
            f"coverage95={cov:.2f}  {flag}"
        )


if __name__ == "__main__":
    app()
