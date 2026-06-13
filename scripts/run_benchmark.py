"""Run forecasting benchmarks.

Executes the model ladder against a dataset, logs results to MLflow,
and compares against published benchmark scores.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --dataset m4-hourly --models baselines
"""

from __future__ import annotations

import os

import mlflow
import typer
from rich.console import Console
from rich.table import Table

from prophet.config import settings
from prophet.data.loaders import M4Frequency, load_m4, load_synthetic
from prophet.evaluation.benchmarks import m4_target
from prophet.evaluation.cross_validation import split_train_test
from prophet.evaluation.metrics import aggregate_metrics, evaluate
from prophet.experiments.tracking import benchmark_run, log_metrics_dict

app = typer.Typer(no_args_is_help=False)
console = Console()


@app.command()
def main(
    dataset: str = typer.Option("synthetic", help="Dataset: synthetic, m4-hourly, m4-daily"),
    models: str = typer.Option("baselines", help="Model group: baselines, statistical, ml, neural"),
    sample_n: int | None = typer.Option(None, help="Cap on number of unique series"),
    rank_sample_n: int | None = typer.Option(
        None,
        help="Statistical only: rank the ensemble on this many sampled series "
        "(final forecast still covers all series).",
    ),
    n_jobs: int | None = typer.Option(
        None,
        help="Worker/thread count for models. Default leaves ~4 cores free for "
        "interactive use; pass -1 to use all cores.",
    ),
    tune: bool = typer.Option(True, help="ML only: run Optuna hyperparameter tuning."),
    n_trials: int = typer.Option(50, help="ML only: number of Optuna trials when tuning."),
    max_steps: int = typer.Option(1000, help="Neural only: training steps per model."),
    accelerator: str = typer.Option(
        "mps", help="Neural only: device (mps, cpu, gpu). Falls back to CPU per model."
    ),
) -> None:
    """Run the benchmark."""
    console.rule(f"[bold cyan]Prophet benchmark — {dataset} / {models}[/bold cyan]")

    # Be a polite neighbour by default: leave headroom so a long AutoARIMA run
    # doesn't starve the UI. Opt back into all cores with --n-jobs -1.
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 4)
    console.print(f"  workers (n_jobs): {n_jobs}")

    if dataset == "synthetic":
        df = load_synthetic(n_series=10, n_obs=500, frequency="1h", seed=settings.random_seed)
        seasonality = 24
        horizon = 24
        freq = "h"
        target = None
        # Synthetic data has no separate test split; hold out the last `horizon`.
        train_df, test_df = split_train_test(df, horizon=horizon)
    elif dataset.startswith("m4-"):
        freq_str = dataset.split("-", 1)[1].capitalize()
        m4_freq = M4Frequency(freq_str)
        # Use the official M4 train/test split so results reproduce published
        # scores. The test split is the held-out horizon Naive2 was scored on;
        # the full train series is what MASE scales against.
        train_df, test_df = load_m4(m4_freq, settings.data_raw, sample_n=sample_n)
        seasonality = m4_freq.seasonality
        horizon = m4_freq.horizon
        freq = {
            "Hourly": "h",
            "Daily": "D",
            "Monthly": "MS",
            "Quarterly": "QS",
            "Yearly": "YS",
            "Weekly": "W",
        }[m4_freq.value]
        target = m4_target(m4_freq.value, method="ETS")
    elif dataset.startswith("domain-"):
        from prophet.data.domains import DOMAIN_SPECS, load_domain

        name = dataset.split("-", 1)[1]
        if name not in DOMAIN_SPECS:
            console.print(f"[red]Unknown domain '{name}'. Known: {sorted(DOMAIN_SPECS)}[/red]")
            raise typer.Exit(code=2)
        spec = DOMAIN_SPECS[name]
        train_df, test_df = load_domain(name, settings.data_raw, sample_n=sample_n)
        seasonality = spec.seasonality
        horizon = spec.horizon
        freq = spec.freq
        target = None
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(code=2)

    console.print(f"  series: {train_df['unique_id'].n_unique()}")
    console.print(f"  train observations: {train_df.height}")
    console.print(f"  test observations: {test_df.height}")
    console.print(f"  horizon: {horizon}")
    console.print(f"  seasonality: {seasonality}")

    if models not in ("baselines", "statistical", "ml", "neural"):
        console.print(
            f"\n[yellow]Model group '{models}' not yet implemented. See ROADMAP.md.[/yellow]"
        )
        raise typer.Exit(code=1)

    phase = {"baselines": "1", "statistical": "2", "ml": "3", "neural": "4"}[models]

    with benchmark_run(
        run_name=f"{dataset}-{models}",
        tags={"phase": phase, "dataset": dataset, "models": models},
    ):
        if models == "baselines":
            from prophet.models.baselines import forecast_baselines

            forecasts = forecast_baselines(
                train_df, horizon=horizon, seasonality=seasonality, freq=freq, n_jobs=n_jobs
            )
        elif models == "statistical":
            from prophet.models.statistical import forecast_statistical

            forecasts, info = forecast_statistical(
                train_df,
                horizon=horizon,
                season_length=seasonality,
                freq=freq,
                rank_sample_n=rank_sample_n,
                n_jobs=n_jobs,
            )
            # Log the CV-based ensemble selection and weights.
            log_metrics_dict(info.cv_mase, prefix="cv_mase_")
            log_metrics_dict(info.ensemble_weights, prefix="ens_weight_")
            mlflow.set_tag("ensemble_members", ",".join(info.ensemble_members))
            mlflow.log_param("cv_windows", info.cv_windows)
            members = ", ".join(
                f"{name} ({info.ensemble_weights[name]:.2f})" for name in info.ensemble_members
            )
            console.print(f"\n[bold]Ensemble (inverse-CV-MASE weighted):[/bold] {members}")
        elif models == "ml":
            from prophet.models.ml import forecast_ml

            forecasts, ml_info = forecast_ml(
                train_df,
                horizon=horizon,
                season_length=seasonality,
                freq=freq,
                tune=tune,
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            # Log CV MASE, best params, and feature importance to MLflow.
            mlflow.log_metric("cv_mase_LightGBM", ml_info.cv_mase_untuned)
            if ml_info.cv_mase_tuned is not None:
                mlflow.log_metric("cv_mase_LightGBM_tuned", ml_info.cv_mase_tuned)
            mlflow.log_param("n_trials", ml_info.n_trials)
            for name, value in ml_info.best_params.items():
                mlflow.log_param(f"best_{name}", value)
            mlflow.log_dict(ml_info.feature_importance, "feature_importance.json")
            if ml_info.best_params:
                mlflow.log_dict(ml_info.best_params, "best_params.json")
            top_feats = sorted(
                ml_info.feature_importance.items(), key=lambda kv: kv[1], reverse=True
            )[:8]
            console.print(
                "\n[bold]Top features (gain):[/bold] "
                + ", ".join(f"{n}={v:.0f}" for n, v in top_feats)
            )
        else:  # neural
            from prophet.models.neural import forecast_neural

            forecasts, neural_info = forecast_neural(
                train_df,
                horizon=horizon,
                season_length=seasonality,
                freq=freq,
                max_steps=max_steps,
                accelerator=accelerator,
            )
            mlflow.log_param("max_steps", neural_info.max_steps)
            mlflow.log_param("input_size", neural_info.input_size)
            for mname, t in neural_info.timings.items():
                mlflow.log_metric(f"{mname}_fit_seconds", t.fit_seconds)
                mlflow.log_metric(f"{mname}_predict_seconds", t.predict_seconds)
                mlflow.set_tag(f"{mname}_device", t.device)
            if neural_info.failures:
                console.print(f"\n[red]Failed models:[/red] {neural_info.failures}")
            console.print(
                "\n[bold]Training time:[/bold] "
                + ", ".join(
                    f"{m} {t.fit_seconds:.0f}s ({t.device})" for m, t in neural_info.timings.items()
                )
            )

        model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]

        # Evaluate every model, then present ranked by MASE (best first).
        rows: list[tuple[str, dict[str, float]]] = []
        for col in model_cols:
            metrics_df = evaluate(
                train_df, test_df, forecasts, seasonality=seasonality, model_col=col
            )
            agg = aggregate_metrics(metrics_df)
            log_metrics_dict(agg, prefix=f"{col}_")
            rows.append((col, agg))
        rows.sort(key=lambda r: r[1]["mase"])

        results_table = Table(title=f"Per-model aggregate metrics ({dataset}, ranked by MASE)")
        results_table.add_column("Rank", justify="right")
        results_table.add_column("Model")
        results_table.add_column("MASE", justify="right")
        results_table.add_column("sMAPE", justify="right")
        results_table.add_column("WAPE", justify="right")
        for rank, (col, agg) in enumerate(rows, start=1):
            results_table.add_row(
                str(rank),
                col,
                f"{agg['mase']:.4f}",
                f"{agg['smape']:.4f}",
                f"{agg['wape']:.4f}",
            )
        console.print(results_table)

        if target is not None:
            console.print(
                f"\n[bold]M4 target (ETS reference):[/bold] "
                f"MASE={target['mase']}, sMAPE={target['smape']}"
            )


if __name__ == "__main__":
    app()
