"""Download and preprocess M4 competition dataset.

Uses the official M4 GitHub mirror (Mcompetitions/M4-methods).
Stores results as Parquet under data/raw/m4/.

Usage:
    uv run python scripts/download_m4.py
    uv run python scripts/download_m4.py --frequency Hourly
"""

from __future__ import annotations

import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.loaders import M4Frequency

app = typer.Typer(no_args_is_help=False)
console = Console()


@app.command()
def main(
    frequency: str | None = typer.Option(
        None,
        help="Single frequency to download (Hourly, Daily, etc.). Default: all.",
    ),
) -> None:
    """Download M4 data and save as Parquet."""
    target_dir = settings.data_raw / "m4"
    target_dir.mkdir(parents=True, exist_ok=True)

    frequencies = [M4Frequency(frequency)] if frequency else list(M4Frequency)

    console.print(f"[bold]Target directory:[/bold] {target_dir}")
    console.print(f"[bold]Frequencies:[/bold] {[f.value for f in frequencies]}")

    console.print(
        "\n[yellow]Download logic not yet implemented.[/yellow]\n"
        "Implementation options:\n"
        "  1. Use [cyan]utilsforecast.data.M4[/cyan] (recommended)\n"
        "  2. Fetch directly from Mcompetitions/M4-methods GitHub repo\n"
        "  3. Use the [cyan]datasetsforecast[/cyan] package\n\n"
        "See ROADMAP.md Phase 1 acceptance criteria."
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
