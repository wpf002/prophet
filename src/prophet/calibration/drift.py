"""Calibration-drift detection (Step 5).

The actual deliverable: catch a model that *was* well-calibrated and stopped
being so. Input-distribution drift (``prophet.monitoring.drift``) and accuracy
drift are already covered; this is different — a model can stay accurate on the
point forecast while its intervals quietly become dishonest.

Split resolved forecasts into an older *reference* window and a newer *recent*
window, score calibration on each (ECE), and flag drift when the recent window's
calibration has degraded past a threshold. ``was_calibrated`` records whether the
reference window was good in the first place, so the report distinguishes
"regressed from good" (the alarming case) from "never calibrated".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from prophet.calibration.confidence import CalibrationRecord
from prophet.calibration.metrics import calibration_report

# ECE at or below this counts as well calibrated; a recent ECE this much worse
# than the reference counts as degraded.
GOOD_ECE = 0.10
ECE_DEGRADATION = 0.10


@dataclass(frozen=True)
class CalibrationDriftReport:
    """Outcome of a calibration-drift check over reference vs. recent windows."""

    n_reference: int
    n_recent: int
    reference_ece: float
    recent_ece: float
    delta: float  # recent_ece - reference_ece (positive = worse)
    reference_coverage: float
    recent_coverage: float
    was_calibrated: bool  # reference window was well calibrated
    drifted: bool  # recent calibration degraded past the threshold

    @property
    def regressed_from_good(self) -> bool:
        """The alarming case: it used to be calibrated and no longer is."""
        return self.was_calibrated and self.drifted


def _ds_key(record: CalibrationRecord) -> datetime:
    assert record.ds is not None
    return record.ds


def _split(
    records: list[CalibrationRecord],
    cutoff: datetime | None,
    recent_fraction: float,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    """Partition into (reference, recent). By ``cutoff`` datetime if given, else
    by taking the last ``recent_fraction`` of the time-ordered records."""
    all_dated = all(r.ds is not None for r in records)
    ordered = sorted(records, key=_ds_key) if all_dated else list(records)
    if cutoff is not None:
        reference = [r for r in ordered if r.ds is not None and r.ds < cutoff]
        recent = [r for r in ordered if r.ds is not None and r.ds >= cutoff]
        return reference, recent
    k = max(1, round(len(ordered) * (1.0 - recent_fraction)))
    return ordered[:k], ordered[k:]


def calibration_drift(
    records: list[CalibrationRecord],
    *,
    cutoff: datetime | None = None,
    recent_fraction: float = 0.5,
    min_per_window: int = 30,
    good_ece: float = GOOD_ECE,
    ece_degradation: float = ECE_DEGRADATION,
) -> CalibrationDriftReport:
    """Flag calibration drift between an older and a newer window of outcomes.

    Args:
        records: resolved forecasts (ideally carrying ``ds`` for time ordering).
        cutoff: split point; records before it are the reference window.
        recent_fraction: if no cutoff, the newest this fraction is the recent window.
        min_per_window: minimum records each window needs to be scored.
        good_ece: ECE at or below which a window is "well calibrated".
        ece_degradation: recent-minus-reference ECE above which drift fires.

    Raises:
        ValueError: either window has fewer than ``min_per_window`` records.
    """
    reference, recent = _split(records, cutoff, recent_fraction)
    if len(reference) < min_per_window or len(recent) < min_per_window:
        raise ValueError(
            f"Need >= {min_per_window} records per window "
            f"(reference={len(reference)}, recent={len(recent)})."
        )

    ref = calibration_report(reference)
    rec = calibration_report(recent)
    delta = rec.ece - ref.ece
    was_calibrated = ref.ece <= good_ece
    # Drift = recent calibration is both worse than reference AND actually bad now.
    drifted = delta > ece_degradation and rec.ece > good_ece

    return CalibrationDriftReport(
        n_reference=len(reference),
        n_recent=len(recent),
        reference_ece=ref.ece,
        recent_ece=rec.ece,
        delta=delta,
        reference_coverage=ref.coverage,
        recent_coverage=rec.coverage,
        was_calibrated=was_calibrated,
        drifted=drifted,
    )
