"""Published benchmark scores from the M4 and M5 forecasting competitions.

These are the targets every model in Prophet must be evaluated against.
Sources: official M4 and M5 competition writeups (Makridakis et al.).
"""

from __future__ import annotations

from typing import TypedDict


class M4Score(TypedDict):
    """Score record for an M4 frequency."""

    smape: float
    mase: float


# M4 competition official scores by frequency (sMAPE and MASE).
# Source: Makridakis, Spiliotis, Assimakopoulos (2020) - The M4 Competition.
M4_BENCHMARKS: dict[str, dict[str, M4Score]] = {
    "Yearly": {
        "Naive2": {"smape": 16.342, "mase": 3.974},
        "SES": {"smape": 16.396, "mase": 3.981},
        "Theta": {"smape": 14.593, "mase": 3.382},
        "ETS": {"smape": 15.356, "mase": 3.444},
        "ARIMA": {"smape": 15.168, "mase": 3.365},
        "Smyl_winner": {"smape": 13.176, "mase": 2.980},
    },
    "Quarterly": {
        "Naive2": {"smape": 11.012, "mase": 1.371},
        "SES": {"smape": 10.601, "mase": 1.342},
        "Theta": {"smape": 10.310, "mase": 1.231},
        "ETS": {"smape": 10.291, "mase": 1.231},
        "ARIMA": {"smape": 10.431, "mase": 1.165},
        "Smyl_winner": {"smape": 9.679, "mase": 1.118},
    },
    "Monthly": {
        "Naive2": {"smape": 14.427, "mase": 1.063},
        "SES": {"smape": 13.610, "mase": 1.040},
        "Theta": {"smape": 13.002, "mase": 0.970},
        "ETS": {"smape": 13.525, "mase": 1.005},
        "ARIMA": {"smape": 13.443, "mase": 0.971},
        "Smyl_winner": {"smape": 12.126, "mase": 0.884},
    },
    "Weekly": {
        "Naive2": {"smape": 9.161, "mase": 2.777},
        "SES": {"smape": 9.012, "mase": 2.685},
        "Theta": {"smape": 9.093, "mase": 2.637},
        "ETS": {"smape": 8.727, "mase": 2.527},
        "ARIMA": {"smape": 8.653, "mase": 2.359},
        "Smyl_winner": {"smape": 7.817, "mase": 2.356},
    },
    "Daily": {
        "Naive2": {"smape": 3.045, "mase": 3.278},
        "SES": {"smape": 3.046, "mase": 3.252},
        "Theta": {"smape": 3.053, "mase": 3.262},
        "ETS": {"smape": 3.046, "mase": 3.253},
        "ARIMA": {"smape": 3.193, "mase": 3.410},
        "Smyl_winner": {"smape": 2.836, "mase": 3.045},
    },
    "Hourly": {
        # Naive2 MASE corrected from 11.608 to 2.395 in Phase 1: the original
        # value was the plain Naive(1) score, not Naive2. Verified by scoring the
        # official M4 Naive2 submission (datasetsforecast mirror) with our metric
        # against the held-out test, clean train scaling -> MASE 2.395, sMAPE
        # 18.383. See docs/phase-1-results.md.
        "Naive2": {"smape": 18.383, "mase": 2.395},
        # NOTE: SES/Theta Hourly MASE below are unverified and look anomalous
        # (~11.5, the Naive(1) magnitude rather than the 2-3 range Naive2 lands
        # in). Treat as suspect until checked against their official submissions.
        "SES": {"smape": 18.094, "mase": 11.426},
        "Theta": {"smape": 18.138, "mase": 11.504},
        "ETS": {"smape": 17.307, "mase": 3.443},
        "ARIMA": {"smape": 13.980, "mase": 2.452},
        "Smyl_winner": {"smape": 9.328, "mase": 0.893},
    },
}


def m4_target(frequency: str, method: str = "ETS") -> M4Score:
    """Lookup a published M4 score.

    Args:
        frequency: One of Yearly, Quarterly, Monthly, Weekly, Daily, Hourly.
        method: Method name. Defaults to ETS (strong statistical baseline).
            Use 'Smyl_winner' for the competition-winning score.

    Returns:
        Dict with 'smape' and 'mase' for the requested method.

    Raises:
        KeyError: If frequency or method not found.
    """
    return M4_BENCHMARKS[frequency][method]


def within_tolerance(
    our_score: float,
    target_score: float,
    *,
    tolerance_pct: float = 10.0,
) -> bool:
    """Check whether our score is within tolerance of a target.

    Args:
        our_score: Our model's score (lower = better for sMAPE/MASE).
        target_score: Published benchmark score.
        tolerance_pct: Allowed relative deviation in percent.

    Returns:
        True if our_score <= target_score * (1 + tolerance_pct / 100).
    """
    return our_score <= target_score * (1.0 + tolerance_pct / 100.0)
