"""Compute + log calibration for the macro model.

Two sources of resolved forecasts:

* ``--backtest`` — cross-validate AutoETS on the local macro data to generate
  (predicted, 95% interval, actual) records. Use before live actuals have
  accrued, and to validate the calibration machinery on the known-good macro
  model (AutoETS, MASE 0.169 vs naive 0.740).
* default — read resolved forecasts from the monitoring DB (PROPHET_MONITOR_DSN).

Results persist to the MLflow calibration experiment; read them via
GET /calibration/{model} and GET /calibration/drift.

Usage:
    uv run python scripts/calibrate_macro.py --backtest
    PROPHET_MONITOR_DSN=... uv run python scripts/calibrate_macro.py
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import typer
from rich.console import Console

from prophet.calibration.confidence import CalibrationRecord
from prophet.calibration.metrics import calibration_report
from prophet.calibration.service import calibrate_and_log, records_from_resolved
from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS, load_domain

app = typer.Typer(no_args_is_help=False)
console = Console()

DATASET = "macro"
LEVEL = 95


def _backtest_records(n_windows: int) -> list[CalibrationRecord]:
    """Resolved records from an expanding-window backtest of AutoETS on macro."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    spec = DOMAIN_SPECS[DATASET]
    train, test = load_domain(DATASET, settings.data_raw)
    panel = pl.concat([train, test]).sort(["unique_id", "ds"]).to_pandas()
    sf = StatsForecast(
        models=[AutoETS(season_length=spec.seasonality, alias="AutoETS")],
        freq=spec.freq,
        n_jobs=1,
    )
    cv = sf.cross_validation(
        df=panel, h=spec.horizon, n_windows=n_windows, step_size=spec.horizon, level=[LEVEL]
    )
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv = pl.from_pandas(cv)
    # True horizon = step index of each ds within its (series, cutoff) window.
    cv = cv.with_columns(
        (pl.col("ds").rank("ordinal").over(["unique_id", "cutoff"])).cast(pl.Int64).alias("h")
    )
    lo, hi = f"AutoETS-lo-{LEVEL}", f"AutoETS-hi-{LEVEL}"
    return [
        CalibrationRecord(
            predicted=float(r["AutoETS"]),
            lo=float(r[lo]),
            hi=float(r[hi]),
            level=LEVEL,
            actual=float(r["y"]),
            model=DATASET,
            series_id=str(r["unique_id"]),
            horizon=int(r["h"]),
            ds=r["ds"] if isinstance(r["ds"], dt.datetime) else None,
        )
        for r in cv.to_dicts()
    ]


@app.command()
def main(
    backtest: bool = typer.Option(False, help="Backtest AutoETS locally instead of the monitor DB."),
    n_windows: int = typer.Option(24, help="CV windows for --backtest (>=24 → empirical basis)."),
    min_group: int = typer.Option(8, help="Minimum resolved records per (series, horizon)."),
) -> None:
    """Compute + log macro calibration; print a per-series validation readout."""
    if backtest:
        console.print("[bold]Backtesting AutoETS on macro (known-good baseline)…[/bold]")
        records = _backtest_records(n_windows)
    else:
        dsn = settings.monitor_dsn
        if not dsn:
            raise typer.BadParameter("Set PROPHET_MONITOR_DSN, or use --backtest.")
        from prophet.monitoring.store import resolved_forecasts

        records = records_from_resolved(resolved_forecasts(dsn, model=DATASET))
    if not records:
        console.print("[yellow]No resolved forecasts yet.[/yellow]")
        raise typer.Exit(0)

    # Per-series validation readout (pooled over horizons), then persist per group.
    console.print(f"[bold]Calibration by series[/bold] (level {LEVEL}%, {len(records)} records):")
    for series in sorted({r.series_id for r in records if r.series_id}):
        recs = [r for r in records if r.series_id == series]
        rep = calibration_report(recs)
        console.print(
            f"  {series:9s} n={rep.n:>4} coverage={rep.coverage:.2f} "
            f"ece={rep.ece:.3f} brier={rep.brier:.3f} log_score={rep.log_score:.2f}"
        )

    logged = calibrate_and_log(DATASET, records, min_group=min_group)
    drifted = [r for r in logged if r["drifted"]]
    console.print(
        f"[green]Logged[/green] {len(logged)} (series, horizon) calibration runs to MLflow "
        f"({len(drifted)} drifted)."
    )


if __name__ == "__main__":
    app()
