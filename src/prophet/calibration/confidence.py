"""Structured confidence and resolved-forecast records (Step 3, data model).

Confidence here is *structured*, not a scalar. A forecast carries not just a point
and an interval, but where that interval's width came from (``basis``), how much
resolved evidence backs it (``evidence_count``), how independent that evidence is
(``independence_score``), and when it was last checked against outcomes
(``last_calibrated``). That's what lets a caller tell a 90% interval built from
300 independent resolved outcomes apart from a 90% interval that is a model
default no outcome has ever tested.

Storage reuses MLflow (see ``prophet.calibration.service`` /
``prophet.calibration.tracking``); these are the in-memory shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Basis(StrEnum):
    """Where an interval's width comes from — how much to trust it.

    * ``empirical``     — calibrated from resolved forecast/actual outcomes
                          (conformal). The interval has been tested against reality.
    * ``model_derived`` — the model's own predictive interval, not yet checked
                          against outcomes for this series/horizon.
    * ``assumed``       — a default used when there is no evidence at all.
    """

    EMPIRICAL = "empirical"
    MODEL_DERIVED = "model_derived"
    ASSUMED = "assumed"


@dataclass(frozen=True)
class Confidence:
    """Structured confidence attached to a single forecast step."""

    point_estimate: float
    interval: tuple[float, float]  # (lo, hi)
    level: int  # nominal confidence level of the interval, e.g. 80
    basis: Basis
    evidence_count: int  # resolved forecast/actual pairs backing the calibration
    independence_score: float  # 0..1 — discounts autocorrelated / overlapping evidence
    last_calibrated: datetime | None

    def __post_init__(self) -> None:
        lo, hi = self.interval
        if hi < lo:
            raise ValueError(f"interval hi ({hi}) < lo ({lo}).")
        if not 0 < self.level < 100:
            raise ValueError(f"level must be in (0, 100), got {self.level}.")
        if not 0.0 <= self.independence_score <= 1.0:
            raise ValueError(f"independence_score must be in [0, 1], got {self.independence_score}.")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative.")

    @property
    def effective_evidence(self) -> float:
        """Evidence count discounted by independence — the count that actually counts."""
        return self.evidence_count * self.independence_score


@dataclass(frozen=True)
class CalibrationRecord:
    """One resolved forecast: what was predicted vs. what happened.

    The atom calibration metrics are computed from. ``lo``/``hi`` are the bounds
    of the nominal ``level`` central interval; ``actual`` is the realized value.
    """

    predicted: float
    lo: float
    hi: float
    level: int
    actual: float
    # Optional provenance for grouping (per model / series / horizon).
    model: str | None = None
    series_id: str | None = None
    horizon: int | None = None
    ds: datetime | None = None

    def covered(self) -> bool:
        """Did the realized value fall inside the predicted interval?"""
        return self.lo <= self.actual <= self.hi
