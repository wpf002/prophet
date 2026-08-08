"""Calibration-drift tests (Step 5)."""

from __future__ import annotations

import datetime as dt
from statistics import NormalDist

import numpy as np
import pytest

from prophet.calibration.confidence import CalibrationRecord
from prophet.calibration.drift import calibration_drift

LEVEL = 80
_Z = NormalDist().inv_cdf((1 + LEVEL / 100) / 2)
_BASE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _window(n, *, true_sigma, stated_sigma, start_i, seed):
    """n dated records; outcomes ~ N(100, true_sigma), intervals from stated_sigma."""
    rng = np.random.default_rng(seed)
    half = _Z * stated_sigma
    recs = []
    for k in range(n):
        actual = 100.0 + rng.normal(0, true_sigma)
        recs.append(
            CalibrationRecord(
                predicted=100.0,
                lo=100.0 - half,
                hi=100.0 + half,
                level=LEVEL,
                actual=actual,
                ds=_BASE + dt.timedelta(days=start_i + k),
            )
        )
    return recs


def test_stable_calibration_is_not_flagged() -> None:
    recs = _window(200, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1) + _window(
        200, true_sigma=1.0, stated_sigma=1.0, start_i=200, seed=2
    )
    report = calibration_drift(recs)
    assert report.was_calibrated is True
    assert report.drifted is False
    assert report.regressed_from_good is False


def test_regression_from_good_is_flagged() -> None:
    # Reference window well calibrated; recent window overconfident (intervals 3x narrow).
    reference = _window(300, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1)
    recent = _window(300, true_sigma=1.0, stated_sigma=1 / 3, start_i=300, seed=2)
    report = calibration_drift(reference + recent)
    assert report.was_calibrated is True
    assert report.drifted is True
    assert report.regressed_from_good is True
    assert report.recent_ece > report.reference_ece
    assert report.recent_coverage < report.reference_coverage


def test_cutoff_split_by_timestamp() -> None:
    reference = _window(300, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1)
    recent = _window(300, true_sigma=1.0, stated_sigma=1 / 3, start_i=300, seed=2)
    cutoff = _BASE + dt.timedelta(days=300)
    report = calibration_drift(reference + recent, cutoff=cutoff)
    assert report.n_reference == 300
    assert report.n_recent == 300
    assert report.regressed_from_good is True


def test_too_few_records_raises() -> None:
    recs = _window(20, true_sigma=1.0, stated_sigma=1.0, start_i=0, seed=1)
    with pytest.raises(ValueError, match="per window"):
        calibration_drift(recs)
