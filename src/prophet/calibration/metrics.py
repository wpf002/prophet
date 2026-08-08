"""Calibration metrics (Step 4).

Given resolved forecasts (predicted point + nominal interval vs. realized value),
answer: *are the intervals honest?* An 80% interval should contain the outcome
~80% of the time. We reconstruct a predictive Gaussian ``N(point, sigma)`` from
each record's interval (``sigma = half-width / z_level``) and score it:

- **reliability curve** — for a grid of nominal levels, the empirical central
  coverage. Perfect calibration lies on the diagonal.
- **ECE** (expected calibration error) — mean |nominal - empirical| over the
  grid. 0 = perfectly calibrated. This is the "0 is perfect" number.
- **Brier** — proper score of the coverage indicators against their nominal
  probabilities (lower is better; calibration + resolution).
- **log score** — Gaussian negative log-likelihood (lower is better; rewards
  sharp *and* calibrated). A proper scoring rule.
- **coverage** — raw fraction of outcomes inside the stated interval.

No scipy: quantiles and the normal CDF come from ``statistics.NormalDist``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from prophet.calibration.confidence import CalibrationRecord

# Nominal levels the reliability curve is evaluated at (central two-sided).
DEFAULT_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_EPS = 1e-9


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration summary for a group of resolved forecasts."""

    n: int
    nominal_level: int  # the interval level the records were served at
    coverage: float  # raw fraction inside the stated interval
    ece: float  # expected calibration error (0 = perfect)
    brier: float  # proper score of coverage indicators (lower better)
    log_score: float  # Gaussian NLL (lower better)
    reliability: list[tuple[float, float]]  # (nominal, empirical) points


def _z(level_fraction: float) -> float:
    """Two-sided z for a central interval, e.g. 0.8 -> 1.2816."""
    return NormalDist().inv_cdf((1.0 + level_fraction) / 2.0)


def _sigmas(records: list[CalibrationRecord]) -> npt.NDArray[np.float64]:
    """Reconstruct predictive sigma per record from its interval half-width."""
    out = np.empty(len(records), dtype=float)
    for i, r in enumerate(records):
        z = _z(r.level / 100.0)
        sigma = (r.hi - r.lo) / (2.0 * z) if z > 0 else 0.0
        # Degenerate (zero-width) intervals get a tiny floor so scores stay finite.
        out[i] = max(sigma, _EPS * (abs(r.predicted) + 1.0))
    return out


def reliability_curve(
    records: list[CalibrationRecord], grid: tuple[float, ...] = DEFAULT_GRID
) -> list[tuple[float, float]]:
    """Empirical central coverage at each nominal level (the reliability curve)."""
    if not records:
        raise ValueError("Need at least one record.")
    point = np.array([r.predicted for r in records], dtype=float)
    actual = np.array([r.actual for r in records], dtype=float)
    sigma = _sigmas(records)
    dev = np.abs(actual - point)
    return [(p, float(np.mean(dev <= _z(p) * sigma))) for p in grid]


def expected_calibration_error(
    records: list[CalibrationRecord], grid: tuple[float, ...] = DEFAULT_GRID
) -> float:
    """Mean absolute gap between nominal and empirical coverage over the grid."""
    curve = reliability_curve(records, grid)
    return float(np.mean([abs(nominal - empirical) for nominal, empirical in curve]))


def brier_score(
    records: list[CalibrationRecord], grid: tuple[float, ...] = DEFAULT_GRID
) -> float:
    """Brier score of coverage indicators vs. their nominal probabilities."""
    point = np.array([r.predicted for r in records], dtype=float)
    actual = np.array([r.actual for r in records], dtype=float)
    sigma = _sigmas(records)
    dev = np.abs(actual - point)
    terms = [np.mean((p - (dev <= _z(p) * sigma).astype(float)) ** 2) for p in grid]
    return float(np.mean(terms))


def log_score(records: list[CalibrationRecord]) -> float:
    """Mean Gaussian negative log-likelihood of the outcomes (lower is better)."""
    point = np.array([r.predicted for r in records], dtype=float)
    actual = np.array([r.actual for r in records], dtype=float)
    sigma = _sigmas(records)
    nll = 0.5 * np.log(2.0 * math.pi * sigma**2) + (actual - point) ** 2 / (2.0 * sigma**2)
    return float(np.mean(nll))


def interval_coverage(records: list[CalibrationRecord]) -> float:
    """Raw fraction of outcomes that fell inside the stated interval."""
    return float(np.mean([r.covered() for r in records]))


def calibration_report(
    records: list[CalibrationRecord], grid: tuple[float, ...] = DEFAULT_GRID
) -> CalibrationReport:
    """Bundle every calibration metric for a group of resolved forecasts."""
    if not records:
        raise ValueError("Need at least one record.")
    levels = {r.level for r in records}
    nominal = next(iter(levels)) if len(levels) == 1 else max(levels)
    return CalibrationReport(
        n=len(records),
        nominal_level=nominal,
        coverage=interval_coverage(records),
        ece=expected_calibration_error(records, grid),
        brier=brier_score(records, grid),
        log_score=log_score(records),
        reliability=reliability_curve(records, grid),
    )
