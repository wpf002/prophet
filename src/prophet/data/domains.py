"""Applied-domain datasets (Phase 5).

The Phase 1-4 model ladder is source-agnostic - every model consumes the Nixtla
long format ``(unique_id, ds, y)``. So an applied domain only needs two things:

1. a **connector** that materializes its source (Postgres, odds API, BI export)
   into ``data/raw/domains/<name>-{train,test}.parquet`` in that long format, and
2. a **DomainSpec** describing its frequency, horizon, and seasonality.

Everything downstream (baselines, statistical, ML) then runs unchanged via
``run_benchmark.py --dataset domain-<name>``.

The specs below are provisional starting points — adjust each once its real data
shape is known. Connectors live in ``scripts/ingest_<domain>.py`` (one per
source) and are intentionally not committed with credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = ("unique_id", "ds", "y")


@dataclass(frozen=True)
class DomainSpec:
    """Forecasting configuration for an applied domain.

    Attributes:
        name: Dataset key used as ``--dataset domain-<name>``.
        freq: Pandas frequency string (e.g. "h", "D", "MS").
        horizon: Forecast horizon in steps.
        seasonality: Seasonal period for SeasonalNaive / MASE scaling.
        description: One-line human summary.
    """

    name: str
    freq: str
    horizon: int
    seasonality: int
    description: str


# Provisional specs — confirm against real data before citing results.
DOMAIN_SPECS: dict[str, DomainSpec] = {
    "casino": DomainSpec(
        name="casino",
        freq="h",
        horizon=24,
        seasonality=24,
        description="Casino table-game revenue by hour and game (cents).",
    ),
    "betting": DomainSpec(
        name="betting",
        freq="h",
        horizon=24,
        seasonality=24,
        description="Sports betting line movement / CLV, intraday.",
    ),
    "cashflow": DomainSpec(
        name="cashflow",
        freq="D",
        horizon=30,
        seasonality=7,
        description="Personal cash flow (Syntrackr), daily with weekly seasonality.",
    ),
    # Market data on the user's portfolio tickers (Alpaca daily bars). Trading
    # days are reindexed to a regular sequence (calendar gaps dropped), so
    # seasonality is the 5-day trading week and the horizon is ~1 trading month.
    "market-close": DomainSpec(
        name="market-close",
        freq="D",
        horizon=21,
        seasonality=5,
        description="Daily close price (cents) for portfolio tickers. Expected: ~random walk.",
    ),
    "market-vol": DomainSpec(
        name="market-vol",
        freq="D",
        horizon=21,
        seasonality=5,
        description="Daily trading volume for portfolio tickers. Expected: forecastable.",
    ),
}


def load_domain(
    name: str,
    data_dir: Path,
    *,
    sample_n: int | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load an applied-domain train/test split from Parquet.

    Mirrors ``load_m4``: reads ``<data_dir>/domains/<name>-{train,test}.parquet``,
    validates the long-format schema, and optionally subsamples series.

    Args:
        name: Domain key (must exist in DOMAIN_SPECS).
        data_dir: Root data directory (parquet lives under ``domains/``).
        sample_n: Optional cap on number of unique series.
        seed: Seed for the subsample.

    Returns:
        (train, test) Polars DataFrames in long format.

    Raises:
        KeyError: If the domain is unknown.
        FileNotFoundError: If the parquet files are missing.
        ValueError: If a loaded frame lacks the required columns.
    """
    if name not in DOMAIN_SPECS:
        raise KeyError(f"Unknown domain {name!r}. Known: {sorted(DOMAIN_SPECS)}")

    domain_dir = data_dir / "domains"
    train_path = domain_dir / f"{name}-train.parquet"
    test_path = domain_dir / f"{name}-test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Domain '{name}' parquet not found at {domain_dir}. "
            f"Run its connector (scripts/ingest_{name}.py) to materialize "
            f"{name}-train.parquet and {name}-test.parquet."
        )

    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    for label, frame in (("train", train), ("test", test)):
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Domain '{name}' {label} is missing columns {missing}.")

    if sample_n is not None:
        ids = train["unique_id"].unique().sort()
        sampled = ids.sample(n=min(sample_n, len(ids)), seed=seed)
        train = train.filter(pl.col("unique_id").is_in(sampled))
        test = test.filter(pl.col("unique_id").is_in(sampled))

    return train, test
