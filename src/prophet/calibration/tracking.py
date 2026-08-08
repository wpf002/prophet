"""MLflow integration for calibration records.

Calibration persists to the SAME MLflow store as benchmarks (``mlruns/`` — there
is no parallel store), but in a dedicated experiment
(``settings.mlflow_calibration_experiment``, default ``prophet-calibration``) so
calibration runs are queryable apart from benchmark runs.

Each calibration record is one MLflow run tagged with the model, series, and
horizon it scores. Metrics (Brier, log score, interval coverage, ECE) are logged
as run metrics; the reliability curve is logged as a JSON artifact. This mirrors
``prophet.experiments.tracking`` so both use the same conventions.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from prophet.config import settings


def configure_calibration_mlflow() -> None:
    """Point MLflow at the shared store and the calibration experiment."""
    # Recent MLflow gates the file-store backend behind this opt-out; set it
    # before any store is constructed (mirrors experiments.tracking).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_calibration_experiment)


@contextmanager
def calibration_run(
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Context manager for one calibration MLflow run.

    Usage:
        with calibration_run("macro/UNRATE/h6", tags={"model": "macro"}):
            mlflow.log_metric("brier", 0.08)

    Every run is tagged ``kind=calibration`` so calibration runs are trivially
    filterable within the experiment.
    """
    configure_calibration_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        merged = {"kind": "calibration"}
        if tags:
            merged.update(tags)
        mlflow.set_tags(merged)
        yield run
