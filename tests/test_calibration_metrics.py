"""Calibration metric tests (Step 4).

Build records with a *known* predictive distribution and check the metrics can
tell a well-calibrated forecaster from a miscalibrated one.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from prophet.calibration.confidence import CalibrationRecord
from prophet.calibration.metrics import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    interval_coverage,
    log_score,
    reliability_curve,
)

LEVEL = 80
_Z80 = NormalDist().inv_cdf((1 + LEVEL / 100) / 2)


def _records(*, true_sigma: float, stated_sigma: float, n: int = 4000, seed: int = 42):
    """Outcomes ~ N(point, true_sigma); intervals built from stated_sigma.

    stated_sigma == true_sigma  → well calibrated.
    stated_sigma  < true_sigma  → over-confident (intervals too narrow).
    """
    rng = np.random.default_rng(seed)
    point = 100.0
    half = _Z80 * stated_sigma
    recs = []
    for _ in range(n):
        actual = point + rng.normal(0, true_sigma)
        recs.append(
            CalibrationRecord(
                predicted=point, lo=point - half, hi=point + half, level=LEVEL, actual=actual
            )
        )
    return recs


def test_well_calibrated_has_low_ece_and_nominal_coverage() -> None:
    recs = _records(true_sigma=1.0, stated_sigma=1.0)
    assert expected_calibration_error(recs) < 0.03
    # 80% intervals should contain ~80% of outcomes.
    assert abs(interval_coverage(recs) - 0.80) < 0.03


def test_reliability_curve_tracks_the_diagonal_when_calibrated() -> None:
    recs = _records(true_sigma=1.0, stated_sigma=1.0)
    for nominal, empirical in reliability_curve(recs):
        assert abs(nominal - empirical) < 0.05


def test_overconfident_intervals_are_flagged() -> None:
    good = _records(true_sigma=1.0, stated_sigma=1.0)
    bad = _records(true_sigma=1.0, stated_sigma=1 / 3)  # intervals 3x too narrow

    # Miscalibration shows up in every metric.
    assert expected_calibration_error(bad) > 0.20
    assert expected_calibration_error(bad) > expected_calibration_error(good)
    assert interval_coverage(bad) < 0.5  # 80% intervals catch far less than 80%
    assert brier_score(bad) > brier_score(good)
    assert log_score(bad) > log_score(good)


def test_underconfident_intervals_are_flagged() -> None:
    good = _records(true_sigma=1.0, stated_sigma=1.0)
    wide = _records(true_sigma=1.0, stated_sigma=3.0)  # intervals 3x too wide
    assert expected_calibration_error(wide) > expected_calibration_error(good)
    assert interval_coverage(wide) > 0.95  # 80% intervals swallow almost everything


def test_report_bundles_everything() -> None:
    rep = calibration_report(_records(true_sigma=1.0, stated_sigma=1.0))
    assert rep.n == 4000
    assert rep.nominal_level == 80
    assert 0.0 <= rep.coverage <= 1.0
    assert rep.ece < 0.05
    assert len(rep.reliability) == 9
