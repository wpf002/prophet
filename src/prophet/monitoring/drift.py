"""Drift detection — input distribution (PSI) and rolling accuracy.

Two production guards (Phase 6):

- **PSI** (Population Stability Index): compares a recent input distribution to
  the training distribution. PSI > 0.25 conventionally signals significant drift.
- **Rolling MASE**: compares recent forecast accuracy to the model's training-time
  CV MASE. A sustained ratio above ~1.5x signals the model has degraded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PSI_ALERT_THRESHOLD = 0.25
MASE_ALERT_RATIO = 1.5


def population_stability_index(
    expected: FloatArray,
    actual: FloatArray,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between a reference and a current sample.

    Bins are the reference distribution's quantiles (open-ended tails), so the
    metric is robust to scale. 0 means identical; rule-of-thumb breakpoints are
    0.1 (minor) and 0.25 (significant) drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("expected and actual must be non-empty.")

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = np.clip(e_counts / expected.size, eps, None)
    a_prop = np.clip(a_counts / actual.size, eps, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


@dataclass(frozen=True)
class DriftReport:
    """Outcome of a drift check."""

    psi: float
    psi_alert: bool
    rolling_mase: float | None
    mase_ratio: float | None
    mase_alert: bool

    @property
    def alert(self) -> bool:
        return self.psi_alert or self.mase_alert


def check_drift(
    train_inputs: FloatArray,
    recent_inputs: FloatArray,
    *,
    rolling_mase: float | None = None,
    train_cv_mase: float | None = None,
    psi_threshold: float = PSI_ALERT_THRESHOLD,
    mase_ratio: float = MASE_ALERT_RATIO,
) -> DriftReport:
    """Combine input-distribution PSI and rolling-MASE checks into one report.

    Args:
        train_inputs: Reference (training) values of the monitored input.
        recent_inputs: Recent values of the same input.
        rolling_mase: Recent rolling-window MASE (None to skip the accuracy check).
        train_cv_mase: The model's training-time CV MASE baseline.
        psi_threshold: PSI value above which input drift fires.
        mase_ratio: rolling/baseline MASE ratio above which accuracy drift fires.
    """
    psi = population_stability_index(train_inputs, recent_inputs)
    ratio: float | None = None
    mase_alert = False
    if rolling_mase is not None and train_cv_mase:
        ratio = rolling_mase / train_cv_mase
        mase_alert = ratio > mase_ratio
    return DriftReport(
        psi=psi,
        psi_alert=psi > psi_threshold,
        rolling_mase=rolling_mase,
        mase_ratio=ratio,
        mase_alert=mase_alert,
    )
