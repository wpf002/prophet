"""Scaffold tests for the calibration module (Step 1).

Verify the package imports and that calibration records land in a *separate*
MLflow experiment within the *same* tracking store as benchmarks — no parallel
store.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from prophet.calibration import calibration_run, configure_calibration_mlflow
from prophet.config import settings


def test_package_exports_public_api() -> None:
    import prophet.calibration as cal

    assert callable(cal.calibration_run)
    assert callable(cal.configure_calibration_mlflow)


def test_calibration_experiment_is_distinct_from_benchmarks(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))
    configure_calibration_mlflow()

    exp = mlflow.get_experiment_by_name(settings.mlflow_calibration_experiment)
    assert exp is not None
    # Same store, different experiment than the benchmark experiment.
    assert settings.mlflow_calibration_experiment != settings.mlflow_experiment_name


def test_calibration_run_logs_and_tags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "mlflow_tracking_uri", str(tmp_path / "mlruns"))

    with calibration_run("scaffold-smoke", tags={"model": "macro"}) as run:
        mlflow.log_metric("brier", 0.1)
    run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.tags["kind"] == "calibration"
    assert fetched.data.tags["model"] == "macro"
    assert fetched.data.metrics["brier"] == 0.1
