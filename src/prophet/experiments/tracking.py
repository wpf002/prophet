"""MLflow experiment tracking helpers.

Every benchmark run must be logged as an MLflow run. No exceptions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from prophet.config import settings


def configure_mlflow() -> None:
    """Configure MLflow tracking URI and experiment from settings."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


@contextmanager
def benchmark_run(
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Context manager for an MLflow benchmark run.

    Usage:
        with benchmark_run("m4-hourly-baselines", tags={"phase": "1"}) as run:
            mlflow.log_metric("mase", 0.893)

    Args:
        run_name: Human-readable run name.
        tags: Optional dict of tags to attach.

    Yields:
        Active MLflow run.
    """
    configure_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_metrics_dict(metrics: dict[str, float], prefix: str = "") -> None:
    """Log a dict of metrics, optionally with a prefix.

    Args:
        metrics: Dict of metric name -> value.
        prefix: Optional prefix for metric names (e.g. "test_" or "cv_").
    """
    for name, value in metrics.items():
        key = f"{prefix}{name}" if prefix else name
        mlflow.log_metric(key, value)
