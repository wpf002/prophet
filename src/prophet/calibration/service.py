"""Calibration service — compute, persist to MLflow, and read back (Steps 3 & 7).

Ties the pieces together: take resolved forecasts (from the monitoring store),
compute a calibration report + drift + structured confidence per
(model, series, horizon), and log each as an MLflow run in the calibration
experiment. The API reads those runs back — the store of record is MLflow, not a
parallel table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import mlflow

from prophet.calibration.confidence import Basis, CalibrationRecord
from prophet.calibration.drift import calibration_drift
from prophet.calibration.metrics import calibration_report
from prophet.calibration.tracking import calibration_run, configure_calibration_mlflow
from prophet.config import settings

# Effective evidence needed before calibration is treated as empirical.
MIN_EMPIRICAL_EVIDENCE = 20.0
DEFAULT_LEVEL = 95  # the interval level the monitoring store persists


def records_from_resolved(
    rows: list[dict[str, Any]], *, level: int = DEFAULT_LEVEL
) -> list[CalibrationRecord]:
    """Map ``store.resolved_forecasts`` rows into calibration records."""
    return [
        CalibrationRecord(
            predicted=float(r["y_hat"]),
            lo=float(r["y_lo_95"]),
            hi=float(r["y_hi_95"]),
            level=level,
            actual=float(r["actual"]),
            model=r.get("model"),
            series_id=r.get("series_id"),
            horizon=int(r["horizon"]) if r.get("horizon") is not None else None,
            ds=r.get("ds"),
        )
        for r in rows
    ]


def _evidence(records: list[CalibrationRecord]) -> tuple[int, float, Basis]:
    """Structured-confidence stats: count, independence, and the resulting basis.

    Independence discounts repeated targets (the same outcome forecast at several
    origins/horizons is not fresh evidence). Basis is empirical once enough
    *effective* evidence has accrued, else the interval is only model-derived.
    """
    n = len(records)
    unique_targets = len({r.ds for r in records if r.ds is not None}) or n
    independence = min(1.0, unique_targets / n)
    effective = n * independence
    basis = Basis.EMPIRICAL if effective >= MIN_EMPIRICAL_EVIDENCE else Basis.MODEL_DERIVED
    return n, independence, basis


def calibrate_and_log(
    model: str,
    records: list[CalibrationRecord],
    *,
    min_group: int = 10,
    drift_min_per_window: int = 30,
) -> list[dict[str, Any]]:
    """Compute + log calibration per (series, horizon) group; return the reports."""
    groups: dict[tuple[str, int], list[CalibrationRecord]] = defaultdict(list)
    for r in records:
        groups[(r.series_id or "series", r.horizon or 0)].append(r)

    reports: list[dict[str, Any]] = []
    for (series, horizon), recs in sorted(groups.items()):
        if len(recs) < min_group:
            continue
        report = calibration_report(recs)
        ev_count, independence, basis = _evidence(recs)

        drift = None
        # Not enough per window yet → drift stays unknown.
        with suppress(ValueError):
            drift = calibration_drift(recs, min_per_window=drift_min_per_window)

        run_name = f"{model}/{series}/h{horizon}"
        tags = {
            "model": model,
            "series_id": series,
            "horizon": str(horizon),
            "level": str(report.nominal_level),
            "basis": basis.value,
            "evidence_count": str(ev_count),
            "drifted": ("true" if drift.drifted else "false") if drift else "unknown",
            "regressed_from_good": (
                ("true" if drift.regressed_from_good else "false") if drift else "unknown"
            ),
        }
        with calibration_run(run_name, tags=tags):
            mlflow.log_metrics(
                {
                    "n": float(report.n),
                    "coverage": report.coverage,
                    "ece": report.ece,
                    "brier": report.brier,
                    "log_score": report.log_score,
                    "independence_score": independence,
                    "effective_evidence": ev_count * independence,
                }
            )
            mlflow.log_param("reliability", json.dumps(report.reliability))
            if drift:
                mlflow.log_metrics(
                    {
                        "reference_ece": drift.reference_ece,
                        "recent_ece": drift.recent_ece,
                        "ece_delta": drift.delta,
                    }
                )

        reports.append(
            {
                "model": model,
                "series_id": series,
                "horizon": horizon,
                "level": report.nominal_level,
                "n": report.n,
                "coverage": report.coverage,
                "ece": report.ece,
                "brier": report.brier,
                "log_score": report.log_score,
                "basis": basis.value,
                "evidence_count": ev_count,
                "independence_score": independence,
                "effective_evidence": ev_count * independence,
                "reliability": report.reliability,
                "drifted": drift.drifted if drift else None,
                "regressed_from_good": drift.regressed_from_good if drift else None,
                "last_calibrated": datetime.now(tz=UTC).isoformat(),
            }
        )
    return reports


def _run_to_dict(run: Any) -> dict[str, Any]:
    m, t, p = run.data.metrics, run.data.tags, run.data.params
    reliability = json.loads(p["reliability"]) if "reliability" in p else []
    drifted = {"true": True, "false": False}.get(t.get("drifted", "unknown"))
    regressed = {"true": True, "false": False}.get(t.get("regressed_from_good", "unknown"))
    return {
        "model": t.get("model"),
        "series_id": t.get("series_id"),
        "horizon": int(t["horizon"]) if t.get("horizon") else None,
        "level": int(t["level"]) if t.get("level") else None,
        "n": int(m.get("n", 0)),
        "coverage": m.get("coverage"),
        "ece": m.get("ece"),
        "brier": m.get("brier"),
        "log_score": m.get("log_score"),
        "basis": t.get("basis"),
        "evidence_count": int(t["evidence_count"]) if t.get("evidence_count") else None,
        "independence_score": m.get("independence_score"),
        "effective_evidence": m.get("effective_evidence"),
        "reliability": [tuple(pt) for pt in reliability],
        "drifted": drifted,
        "regressed_from_good": regressed,
        "last_calibrated": datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat(),
    }


def latest_calibration(model: str | None = None) -> list[dict[str, Any]]:
    """Latest calibration report per (model, series, horizon) from MLflow."""
    configure_calibration_mlflow()
    exp = mlflow.get_experiment_by_name(settings.mlflow_calibration_experiment)
    if exp is None:
        return []
    filt = "tags.kind = 'calibration'"
    if model:
        filt += f" and tags.model = '{model}'"
    runs = mlflow.search_runs(
        [exp.experiment_id],
        filter_string=filt,
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in runs:  # DESC → first seen per key is the newest
        t = run.data.tags
        key = (t.get("model", ""), t.get("series_id", ""), t.get("horizon", ""))
        if key not in latest:
            latest[key] = _run_to_dict(run)
    return list(latest.values())


def drifted_calibrations(model: str | None = None) -> list[dict[str, Any]]:
    """Latest calibration reports that are currently flagged as drifted."""
    return [c for c in latest_calibration(model) if c.get("drifted") is True]
