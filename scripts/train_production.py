"""Phase 6 — train and persist the production forecasting model.

Fits the winning approach (LightGBM via MLForecast, the Phase 3 recipe) on a
domain dataset's full history, with conformal prediction intervals, and saves it
plus metadata to ``models/production/<dataset>/`` for the API to load on startup.

Usage:
    uv run python scripts/train_production.py --dataset market-vol
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl
import typer
from mlforecast.utils import PredictionIntervals
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS, load_domain
from prophet.experiments.tracking import benchmark_run
from prophet.models.ml import DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS, UNTUNED_COL, _build_mlf

app = typer.Typer(no_args_is_help=False)
console = Console()

PRODUCTION_DIR = Path("models/production")


@app.command()
def main(
    dataset: str = typer.Option("market-vol", help="Domain dataset key to train on."),
    n_jobs: int = typer.Option(4, help="LightGBM threads."),
    non_negative: bool = typer.Option(
        True, help="Clip forecasts and intervals at 0 (correct for counts like volume)."
    ),
) -> None:
    """Fit + persist the production model with prediction intervals."""
    if dataset not in DOMAIN_SPECS:
        console.print(f"[red]Unknown domain '{dataset}'. Known: {sorted(DOMAIN_SPECS)}[/red]")
        raise typer.Exit(code=2)
    spec = DOMAIN_SPECS[dataset]

    # Train on the full available history (train + held-out test) so the served
    # model forecasts from the most recent observation.
    train, test = load_domain(dataset, settings.data_raw)
    panel = pl.concat([train, test]).sort(["unique_id", "ds"])
    panel_pd = panel.to_pandas()
    series = sorted(panel["unique_id"].unique().to_list())
    console.print(f"[bold]{dataset}[/bold]: {len(series)} series, {panel.height} obs")

    model = lgb.LGBMRegressor(random_state=settings.random_seed, n_jobs=n_jobs, verbosity=-1)
    mlf = _build_mlf(
        {UNTUNED_COL: model},
        freq=spec.freq,
        lags=list(DEFAULT_LAGS),
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        seasonal_diff=spec.seasonality,
    )

    with benchmark_run(
        run_name=f"production-{dataset}",
        tags={"phase": "6", "dataset": f"domain-{dataset}", "models": "production"},
    ):
        mlf.fit(
            panel_pd,
            prediction_intervals=PredictionIntervals(n_windows=2, h=spec.horizon),
        )

        out_dir = PRODUCTION_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        mlf.save(str(out_dir))

        metadata = {
            "dataset": dataset,
            "model": UNTUNED_COL,
            "freq": spec.freq,
            "horizon": spec.horizon,
            "seasonality": spec.seasonality,
            "n_series": len(series),
            "series": series,
            "n_obs": panel.height,
            "non_negative": non_negative,
            "trained_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    console.print(f"[green]Saved[/green] production model -> {out_dir}")
    console.print(f"  series: {len(series)} | max horizon: {spec.horizon} | freq: {spec.freq}")


if __name__ == "__main__":
    app()
