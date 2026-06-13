"""Tests for drift detection (PSI + rolling-MASE)."""

from __future__ import annotations

import numpy as np

from prophet.monitoring.drift import check_drift, population_stability_index


class TestPSI:
    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 5000)
        assert population_stability_index(x, x.copy()) < 0.01

    def test_shifted_distribution_flags_drift(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 5000)
        shifted = rng.normal(2.0, 1, 5000)  # large mean shift
        assert population_stability_index(ref, shifted) > 0.25

    def test_empty_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="non-empty"):
            population_stability_index(np.array([]), np.array([1.0]))


class TestCheckDrift:
    def test_no_alert_when_stable(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 2000)
        report = check_drift(ref, ref.copy(), rolling_mase=1.0, train_cv_mase=1.0)
        assert not report.alert
        assert report.mase_ratio == 1.0

    def test_mase_alert_fires(self) -> None:
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 2000)
        report = check_drift(ref, ref.copy(), rolling_mase=2.0, train_cv_mase=1.0)
        assert report.mase_alert
        assert report.alert

    def test_psi_alert_fires_on_input_drift(self) -> None:
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, 2000)
        recent = rng.normal(3.0, 1, 2000)
        report = check_drift(ref, recent)
        assert report.psi_alert
        assert report.alert
