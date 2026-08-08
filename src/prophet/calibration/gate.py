"""Forecastability gate (Step 6).

The four criteria that decide whether a series is worth forecasting, enforced as
code that runs *before* a series is admitted — and that records which criterion
failed, so a rejection is explainable:

1. **is a time series** — regularly-spaced timestamped values.
2. **beats naive** — the best model beats a naive baseline (from the ad-hoc
   verdict). ``None`` (too short to establish) fails: absence of evidence is not
   a pass.
3. **drives a decision** — a downstream decision actually consumes the forecast.
4. **has lead time** — there is time to act between forecast and outcome.

Criteria 1 and 2 are checkable from data (helpers below); 3 and 4 are declared by
the integrator. All four must hold to admit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Criterion(StrEnum):
    """The four forecastability criteria."""

    IS_TIME_SERIES = "is_time_series"
    BEATS_NAIVE = "beats_naive"
    DRIVES_DECISION = "drives_decision"
    HAS_LEAD_TIME = "has_lead_time"


@dataclass(frozen=True)
class GateResult:
    """Outcome of the forecastability gate."""

    admitted: bool
    failed: list[Criterion]
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> list[Criterion]:
        failed = set(self.failed)
        return [c for c in Criterion if c not in failed]


def check_is_time_series(
    timestamps: Sequence[datetime], *, tolerance: float = 0.25
) -> tuple[bool, str]:
    """Are these timestamps a regularly-spaced series?

    Requires >= 3 points and gaps within ``tolerance`` of the median gap (so the
    odd DST / month-end wobble is fine, but an irregular event log is not).
    """
    if len(timestamps) < 3:
        return False, f"only {len(timestamps)} points; need >= 3."
    ts = sorted(timestamps)
    gaps = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    if any(g <= 0 for g in gaps):
        return False, "duplicate or non-increasing timestamps."
    median = sorted(gaps)[len(gaps) // 2]
    if median <= 0:
        return False, "degenerate spacing."
    irregular = sum(1 for g in gaps if abs(g - median) > tolerance * median)
    if irregular > len(gaps) * 0.1:
        return False, f"{irregular}/{len(gaps)} gaps deviate > {tolerance:.0%} from the median."
    return True, "regularly spaced."


def forecastability_gate(
    *,
    is_time_series: bool,
    beats_naive: bool | None,
    drives_decision: bool,
    has_lead_time: bool,
    details: dict[str, str] | None = None,
) -> GateResult:
    """Admit a series only if all four criteria hold; record which failed.

    Args:
        is_time_series: is the input a regular time series (see ``check_is_time_series``).
        beats_naive: does the best model beat naive? ``None`` = couldn't establish → fails.
        drives_decision: does a downstream decision consume this forecast?
        has_lead_time: is there lead time to act on it?
        details: optional per-criterion notes to merge into the recorded reasons.
    """
    reasons: dict[str, str] = {}
    failed: list[Criterion] = []

    def record(crit: Criterion, ok: bool, why: str) -> None:
        reasons[crit.value] = why
        if not ok:
            failed.append(crit)

    record(
        Criterion.IS_TIME_SERIES,
        is_time_series,
        "regular time series." if is_time_series else "not a regular time series.",
    )
    record(
        Criterion.BEATS_NAIVE,
        beats_naive is True,
        "beats naive."
        if beats_naive is True
        else ("does not beat naive." if beats_naive is False else "could not establish (too short)."),
    )
    record(
        Criterion.DRIVES_DECISION,
        drives_decision,
        "drives a downstream decision." if drives_decision else "no downstream decision.",
    )
    record(
        Criterion.HAS_LEAD_TIME,
        has_lead_time,
        "lead time to act." if has_lead_time else "no lead time to act.",
    )

    if details:
        reasons.update(details)
    return GateResult(admitted=not failed, failed=failed, reasons=reasons)
