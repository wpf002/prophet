"""Connector — Bellwether market-intelligence signals -> domain Parquet.

Materializes Bellwether's daily signal volume into a Prophet domain dataset in
long format (unique_id = "<industry>:<metric>", ds = Datetime us UTC at midnight,
y = daily count). The model ladder then runs unchanged via
``run_benchmark.py --dataset domain-bellwether``.

Read-only sources:

* ``--api`` (default base ``http://localhost:4000``) — pulls Bellwether's PUBLIC,
  unauthenticated ``/industries`` list and each industry's
  ``/industries/<id>/trends?days=<n>`` (daily events / company-mentions /
  buyer-complaints, already gap-filled to a regular daily grid server-side).
  No credentials, no DB access. Point ``--base-url`` at the deployed API later.
* ``--synthetic`` — a clearly-labeled stand-in to prove the pipeline when the
  Bellwether API isn't reachable. Never a forecastability verdict.

Usage:
    uv run python scripts/ingest_bellwether.py --api
    uv run python scripts/ingest_bellwether.py --api --base-url https://bellwether.up.railway.app
    uv run python scripts/ingest_bellwether.py --synthetic
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

import numpy as np
import polars as pl
import typer
from rich.console import Console

from prophet.config import settings
from prophet.data.domains import DOMAIN_SPECS
from prophet.evaluation.cross_validation import split_train_test

app = typer.Typer(no_args_is_help=False)
console = Console()

NAME = "bellwether"
DEFAULT_API = "http://localhost:4000"
METRICS = ("events", "companies", "complaints")


def _get_json(url: str, timeout: int = 30) -> object:
    """GET and parse JSON from a public read-only endpoint."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _read_trends_api(base_url: str, days: int) -> pl.DataFrame:
    """READ-ONLY pull of daily signal volume via Bellwether's public API.

    Returns a long panel (unique_id, ds [Datetime us UTC], y). For each industry
    and each metric (events / companies / complaints) the /trends endpoint gives
    a continuous daily series, so no synthetic reindexing is needed.
    """
    base = base_url.rstrip("/")
    industries = _get_json(f"{base}/industries")
    if not isinstance(industries, list):
        raise typer.BadParameter(f"Unexpected /industries response from {base}.")

    rows: list[tuple[str, dt.datetime, float]] = []
    for ind in industries:
        ind_id = ind["id"]
        try:
            points = _get_json(f"{base}/industries/{ind_id}/trends?days={days}")
        except OSError:
            continue
        if not isinstance(points, list):
            continue
        for p in points:
            ds = dt.datetime.fromisoformat(p["date"]).replace(tzinfo=dt.UTC)
            for metric in METRICS:
                rows.append((f"{ind_id}:{metric}", ds, float(p.get(metric, 0) or 0)))

    return (
        pl.DataFrame(rows, schema=["unique_id", "ds", "y"], orient="row")
        .with_columns(pl.col("ds").dt.cast_time_unit("us"))
        .sort(["unique_id", "ds"])
    )


def _synthetic(days: int = 120, seed: int = 42) -> pl.DataFrame:
    """Labeled synthetic daily signal volumes (pipeline proof only)."""
    rng = np.random.default_rng(seed)
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    rows: list[tuple[str, dt.datetime, float]] = []
    for ind in (f"synth-{i:02d}" for i in range(6)):
        for metric in METRICS:
            base = float(rng.integers(2, 20))
            for d in range(days):
                ds = start + dt.timedelta(days=d)
                weekly = 1.0 + 0.4 * np.sin(2 * np.pi * d / 7)  # weekly seasonality
                y = max(0.0, round(base * weekly + rng.normal(0, 2)))
                rows.append((f"{ind}:{metric}", ds, y))
    return (
        pl.DataFrame(rows, schema=["unique_id", "ds", "y"], orient="row")
        .with_columns(pl.col("ds").dt.cast_time_unit("us"))
        .sort(["unique_id", "ds"])
    )


@app.command()
def main(
    api: bool = typer.Option(False, help="Pull real data via Bellwether's public read-only API."),
    base_url: str = typer.Option(DEFAULT_API, help="Bellwether API base URL (with --api)."),
    days: int = typer.Option(365, help="History look-back window in days (with --api, max 365)."),
    synthetic: bool = typer.Option(False, help="Generate labeled synthetic data (pipeline proof)."),
    min_points: int = typer.Option(
        15, help="Drop series with fewer than this many days (need > horizon)."
    ),
) -> None:
    """Build bellwether-train/test Parquet from Bellwether (read-only API) or synthetic."""
    spec = DOMAIN_SPECS[NAME]
    if api:
        console.print(f"[bold]API mode[/bold] — public read-only pull from {base_url} (days={days})")
        panel = _read_trends_api(base_url, days)
    elif synthetic:
        console.print("[yellow]Synthetic mode — pipeline proof only, not a verdict.[/yellow]")
        panel = _synthetic()
    else:
        raise typer.BadParameter("Use --api (default base http://localhost:4000) or --synthetic.")

    if panel.height == 0:
        console.print("[red]No data returned — is the Bellwether API running?[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Series:[/bold] {panel['unique_id'].n_unique()} (industry x metric), "
        f"{panel.height} points"
    )

    # Honesty check: how much of the signal is non-zero? A corpus that's mostly
    # zero-filled (freshly seeded) is "data too young" to forecast meaningfully.
    nonzero_share = float(panel.select((pl.col("y") > 0).mean()).item() or 0.0)
    console.print(f"[bold]Non-zero share:[/bold] {nonzero_share:.1%}")
    if nonzero_share < 0.10:
        console.print(
            "[yellow]Verdict: data too young — series are mostly zero-filled. "
            "Let Bellwether accumulate before trusting forecasts.[/yellow]"
        )

    counts = panel.group_by("unique_id").len()
    keep = counts.filter(pl.col("len") >= min_points)["unique_id"]
    panel = panel.filter(pl.col("unique_id").is_in(keep))
    if panel["unique_id"].n_unique() == 0:
        console.print(f"[red]No series with >= {min_points} days.[/red]")
        raise typer.Exit(code=1)

    train, test = split_train_test(panel, horizon=spec.horizon)
    out_dir = settings.data_raw / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / f"{NAME}-train.parquet")
    test.write_parquet(out_dir / f"{NAME}-test.parquet")
    console.print(
        f"[green]Wrote[/green] {panel['unique_id'].n_unique()} series | "
        f"train {train.height}, test {test.height} -> {out_dir}"
    )
    console.print(
        "Run: uv run python scripts/run_benchmark.py --dataset domain-bellwether --models statistical"
    )


if __name__ == "__main__":
    app()
