"""Prophet CLI — single entry point for project commands."""

from __future__ import annotations

import typer
from rich.console import Console

from prophet import __version__
from prophet.config import settings

app = typer.Typer(
    name="prophet",
    help="Time-series forecasting system benchmarked against M4/M5.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"prophet [bold cyan]v{__version__}[/bold cyan]")


@app.command()
def info() -> None:
    """Print current configuration."""
    console.print("[bold]Prophet configuration[/bold]")
    console.print(f"  data_raw:              {settings.data_raw}")
    console.print(f"  data_processed:        {settings.data_processed}")
    console.print(f"  mlflow_tracking_uri:   {settings.mlflow_tracking_uri}")
    console.print(f"  mlflow_experiment:     {settings.mlflow_experiment_name}")
    console.print(f"  random_seed:           {settings.random_seed}")


@app.command()
def benchmark(
    dataset: str = typer.Option("m4-hourly", help="Dataset slug (e.g. m4-hourly, m4-daily)"),
    models: str = typer.Option(
        "baselines", help="Model group: baselines, statistical, ml, neural, all"
    ),
) -> None:
    """Run benchmark for the given dataset and model group."""
    console.print(f"[yellow]Benchmark not yet implemented for {dataset} / {models}[/yellow]")
    console.print("Implement in [cyan]scripts/run_benchmark.py[/cyan] then wire here.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
