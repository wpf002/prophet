"""Calibration — make forecast confidence an empirically tracked property.

Prophet scores forecasts against realized outcomes, so confidence here can be
*calibrated*, not decorated. This package turns per-run scoring into a tracked,
queryable property of the system over time.

Planned surface (built in order; see ROADMAP / the calibration README):

- ``confidence`` — structured per-forecast confidence: point estimate, interval,
  basis (empirical | model_derived | assumed), evidence count, independence
  score, last-calibrated timestamp.
- ``metrics`` — reliability curves, Brier / log score, and interval coverage,
  computed per model / series / horizon.
- ``drift`` — a calibration-drift detector: flag a model that *was* well
  calibrated and stopped being so, over a rolling window.
- ``gate`` — the four-criteria forecastability gate as code (is a time series,
  beats naive, drives a decision, has lead time), recording which criterion
  failed.
- ``tracking`` — persistence to the existing MLflow store (a dedicated
  experiment; no parallel store).

Empirical calibration is the whole point — there are deliberately no
Dempster-Shafer / imprecise-probability / belief-function formalisms here.
"""

from __future__ import annotations

from prophet.calibration.confidence import Basis, CalibrationRecord, Confidence
from prophet.calibration.drift import CalibrationDriftReport, calibration_drift
from prophet.calibration.gate import (
    Criterion,
    GateResult,
    check_is_time_series,
    forecastability_gate,
)
from prophet.calibration.metrics import CalibrationReport, calibration_report
from prophet.calibration.service import (
    calibrate_and_log,
    drifted_calibrations,
    latest_calibration,
    records_from_resolved,
)
from prophet.calibration.tracking import calibration_run, configure_calibration_mlflow

__all__ = [
    "Basis",
    "CalibrationDriftReport",
    "CalibrationRecord",
    "CalibrationReport",
    "Confidence",
    "Criterion",
    "GateResult",
    "calibrate_and_log",
    "calibration_drift",
    "calibration_report",
    "calibration_run",
    "check_is_time_series",
    "configure_calibration_mlflow",
    "drifted_calibrations",
    "forecastability_gate",
    "latest_calibration",
    "records_from_resolved",
]
