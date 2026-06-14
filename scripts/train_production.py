"""Phase 6 — train and persist the production forecasting model.

Fits a domain dataset's full history with conformal prediction intervals and
saves the model plus metadata to ``models/production/<dataset>/`` for the API to
load on startup. Two engines:

* ``--method ml`` (default) — LightGBM via MLForecast (the Phase 3 recipe). Best
  for large panels with cross-series structure (e.g. market-vol).
* ``--method statistical`` — AutoETS via StatsForecast. Best for a few long,
  regular series where classical per-series models win (e.g. macro: AutoETS
  beats LightGBM 0.17 vs 0.40 MASE).

Usage:
    uv run python scripts/train_production.py --dataset market-vol
    uv run python scripts/train_production.py --dataset macro --method statistical
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

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
STATSFORECAST_FILE = "statsforecast.pkl"
STATISTICAL_MODEL = "AutoETS"


def _fit_ml(panel_pd: Any, out_dir: Path, spec: Any, n_jobs: int) -> tuple[str, str]:
    """Fit + save a LightGBM/MLForecast model with conformal intervals."""
    model = lgb.LGBMRegressor(random_state=settings.random_seed, n_jobs=n_jobs, verbosity=-1)
    mlf = _build_mlf(
        {UNTUNED_COL: model},
        freq=spec.freq,
        lags=list(DEFAULT_LAGS),
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        seasonal_diff=spec.seasonality,
    )
    mlf.fit(panel_pd, prediction_intervals=PredictionIntervals(n_windows=2, h=spec.horizon))
    mlf.save(str(out_dir))
    return UNTUNED_COL, "mlforecast"


def _fit_statistical(panel_pd: Any, out_dir: Path, spec: Any, n_jobs: int) -> tuple[str, str]:
    """Fit + save an AutoETS/StatsForecast model with conformal intervals."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    from statsforecast.utils import ConformalIntervals

    sf = StatsForecast(
        models=[AutoETS(season_length=spec.seasonality, alias=STATISTICAL_MODEL)],
        freq=spec.freq,
        n_jobs=n_jobs,
    )
    sf.fit(panel_pd, prediction_intervals=ConformalIntervals(n_windows=2, h=spec.horizon))
    sf.save(str(out_dir / STATSFORECAST_FILE))
    return STATISTICAL_MODEL, "statsforecast"


@app.command()
def main(
    dataset: str = typer.Option("market-vol", help="Domain dataset key to train on."),
    method: str = typer.Option("ml", help="Engine: ml (LightGBM) or statistical (AutoETS)."),
    n_jobs: int = typer.Option(4, help="Worker count (LightGBM threads / StatsForecast procs)."),
    non_negative: bool = typer.Option(
        True, help="Clip forecasts and intervals at 0 (correct for counts like volume)."
    ),
) -> None:
    """Fit + persist the production model with prediction intervals."""
    if dataset not in DOMAIN_SPECS:
        console.print(f"[red]Unknown domain '{dataset}'. Known: {sorted(DOMAIN_SPECS)}[/red]")
        raise typer.Exit(code=2)
    if method not in ("ml", "statistical"):
        console.print(f"[red]Unknown method '{method}'. Use 'ml' or 'statistical'.[/red]")
        raise typer.Exit(code=2)
    spec = DOMAIN_SPECS[dataset]

    # Train on the full available history (train + held-out test) so the served
    # model forecasts from the most recent observation.
    train, test = load_domain(dataset, settings.data_raw)
    panel = pl.concat([train, test]).sort(["unique_id", "ds"])
    panel_pd = panel.to_pandas()
    series = sorted(panel["unique_id"].unique().to_list())
    console.print(f"[bold]{dataset}[/bold] ({method}): {len(series)} series, {panel.height} obs")

    with benchmark_run(
        run_name=f"production-{dataset}",
        tags={"phase": "6", "dataset": f"domain-{dataset}", "models": f"production-{method}"},
    ):
        # Rebuild the dir so stale artifacts from a different engine can't linger.
        out_dir = PRODUCTION_DIR / dataset
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fit = _fit_statistical if method == "statistical" else _fit_ml
        model_col, engine = fit(panel_pd, out_dir, spec, n_jobs)

        metadata = {
            "dataset": dataset,
            "model": model_col,
            "engine": engine,
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

    console.print(f"[green]Saved[/green] {engine} model ({model_col}) -> {out_dir}")
    console.print(f"  series: {len(series)} | max horizon: {spec.horizon} | freq: {spec.freq}")


if __name__ == "__main__":
    app()
