"""Forecastability-gate tests (Step 6)."""

from __future__ import annotations

import datetime as dt

from prophet.calibration.gate import (
    Criterion,
    check_is_time_series,
    forecastability_gate,
)

_BASE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def test_all_criteria_pass_admits() -> None:
    r = forecastability_gate(
        is_time_series=True, beats_naive=True, drives_decision=True, has_lead_time=True
    )
    assert r.admitted is True
    assert r.failed == []
    assert set(r.passed) == set(Criterion)


def test_records_which_criterion_failed() -> None:
    r = forecastability_gate(
        is_time_series=True, beats_naive=False, drives_decision=True, has_lead_time=True
    )
    assert r.admitted is False
    assert r.failed == [Criterion.BEATS_NAIVE]
    assert "does not beat naive" in r.reasons["beats_naive"]


def test_beats_naive_none_fails_as_unestablished() -> None:
    r = forecastability_gate(
        is_time_series=True, beats_naive=None, drives_decision=True, has_lead_time=True
    )
    assert r.admitted is False
    assert Criterion.BEATS_NAIVE in r.failed
    assert "could not establish" in r.reasons["beats_naive"]


def test_multiple_failures_all_recorded() -> None:
    r = forecastability_gate(
        is_time_series=False, beats_naive=True, drives_decision=False, has_lead_time=True
    )
    assert set(r.failed) == {Criterion.IS_TIME_SERIES, Criterion.DRIVES_DECISION}


def test_check_is_time_series_regular() -> None:
    ts = [_BASE + dt.timedelta(days=i) for i in range(30)]
    ok, why = check_is_time_series(ts)
    assert ok is True
    assert "regular" in why


def test_check_is_time_series_irregular() -> None:
    ts = [_BASE, _BASE + dt.timedelta(days=1), _BASE + dt.timedelta(days=40), _BASE + dt.timedelta(days=41)]
    ok, _ = check_is_time_series(ts)
    assert ok is False


def test_check_is_time_series_too_few() -> None:
    ok, _ = check_is_time_series([_BASE, _BASE + dt.timedelta(days=1)])
    assert ok is False


def test_gate_wires_from_helper() -> None:
    ts = [_BASE + dt.timedelta(days=i) for i in range(30)]
    is_ts, _ = check_is_time_series(ts)
    r = forecastability_gate(
        is_time_series=is_ts, beats_naive=True, drives_decision=True, has_lead_time=True
    )
    assert r.admitted is True
