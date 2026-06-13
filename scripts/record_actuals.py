"""Phase 6 nightly job — record realized values and report rolling accuracy.

Re-fetches recent market bars for the served model's tickers, reindexes them to
the same synthetic ``ds`` the forecasts used, upserts them into ``actuals``, then
prints rolling accuracy + a drift check. Run on a schedule (Railway cron).

Requires PROPHET_MONITOR_DSN, APCA_API_KEY_ID, APCA_API_SECRET_KEY.

Usage:
    uv run python scripts/record_actuals.py --dataset market-vol --days 30
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.market import TARGETS, fetch_bars, to_long
from prophet.monitoring.schema import create_monitoring_tables
from prophet.monitoring.store import rolling_accuracy, upsert_actuals

app = typer.Typer(no_args_is_help=False)
console = Console()


@app.command()
def main(
    dataset: str = typer.Option("market-vol", help="Served dataset / target."),
    start: str = typer.Option("2019-01-01", help="Refetch start date."),
    end: str = typer.Option("2025-06-01", help="Refetch end date."),
    days: int = typer.Option(30, help="Rolling-accuracy window."),
) -> None:
    """Upsert actuals from fresh market data, then report rolling accuracy."""
    dsn = settings.monitor_dsn
    if not dsn:
        console.print("[red]PROPHET_MONITOR_DSN not set.[/red]")
        raise typer.Exit(code=2)
    if dataset not in TARGETS:
        console.print(f"[red]No market target for '{dataset}'. Known: {sorted(TARGETS)}[/red]")
        raise typer.Exit(code=2)

    meta_path = Path("models/production") / settings.production_model / "metadata.json"
    series = json.loads(meta_path.read_text())["series"] if meta_path.exists() else None

    bars = fetch_bars(series or [], start=start, end=end) if series else None
    if bars is None:
        console.print("[red]No production model metadata; cannot resolve series.[/red]")
        raise typer.Exit(code=2)

    panel = to_long(bars, TARGETS[dataset])
    rows = [(r["unique_id"], r["ds"], float(r["y"])) for r in panel.to_dicts()]

    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        create_monitoring_tables(conn)
    written = upsert_actuals(dsn, rows)
    console.print(f"[green]Upserted[/green] {written} actuals for {len(series)} series.")

    report = rolling_accuracy(dsn, days=days)
    if not report:
        console.print(f"[yellow]No forecast/actual overlap in the last {days} days yet.[/yellow]")
        return
    console.print(f"[bold]Rolling accuracy (last {days}d):[/bold]")
    for row in report[:20]:
        console.print(
            f"  {row['series_id']:8s} n={row['n']:>4} mae={row['mae']:,.1f} "
            f"coverage95={row['coverage_95']:.2f}"
        )


if __name__ == "__main__":
    app()
