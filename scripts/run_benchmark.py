"""Run forecasting benchmarks.

Executes the model ladder against a dataset, logs results to MLflow,
and compares against published benchmark scores.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --dataset m4-hourly --models baselines
"""

from __future__ import annotations

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
) -> None:
    """Run the benchmark."""
    console.rule(f"[bold cyan]Prophet benchmark — {dataset} / {models}[/bold cyan]")

    if dataset == "synthetic":
        df = load_synthetic(n_series=10, n_obs=500, frequency="1h", seed=settings.random_seed)
        seasonality = 24
        horizon = 24
        freq = "h"
        target = None
    elif dataset.startswith("m4-"):
        freq_str = dataset.split("-", 1)[1].capitalize()
        m4_freq = M4Frequency(freq_str)
        train, _test = load_m4(m4_freq, settings.data_raw, sample_n=sample_n)
        df = train  # train already excludes the held-out test
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
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(code=2)

    console.print(f"  series: {df['unique_id'].n_unique()}")
    console.print(f"  observations: {df.height}")
    console.print(f"  horizon: {horizon}")
    console.print(f"  seasonality: {seasonality}")

    if models != "baselines":
        console.print(
            f"\n[yellow]Model group '{models}' not yet implemented. See ROADMAP.md.[/yellow]"
        )
        raise typer.Exit(code=1)

    # Phase 1 baseline path
    train_df, test_df = split_train_test(df, horizon=horizon)

    from prophet.models.baselines import forecast_baselines

    with benchmark_run(
        run_name=f"{dataset}-baselines",
        tags={"phase": "1", "dataset": dataset, "models": "baselines"},
    ):
        forecasts = forecast_baselines(
            train_df, horizon=horizon, seasonality=seasonality, freq=freq
        )

        model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
        results_table = Table(title="Per-model aggregate metrics")
        results_table.add_column("Model")
        results_table.add_column("MASE", justify="right")
        results_table.add_column("sMAPE", justify="right")
        results_table.add_column("WAPE", justify="right")

        for col in model_cols:
            metrics_df = evaluate(
                train_df, test_df, forecasts, seasonality=seasonality, model_col=col
            )
            agg = aggregate_metrics(metrics_df)
            log_metrics_dict(agg, prefix=f"{col}_")
            results_table.add_row(
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
