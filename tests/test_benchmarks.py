"""Tests for benchmark lookup utilities."""

from __future__ import annotations

import pytest

from prophet.evaluation.benchmarks import (
    M4_BENCHMARKS,
    m4_target,
    within_tolerance,
)


class TestM4Benchmarks:
    def test_all_frequencies_present(self) -> None:
        expected = {"Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"}
        assert set(M4_BENCHMARKS.keys()) == expected

    def test_each_frequency_has_naive_and_winner(self) -> None:
        for freq, methods in M4_BENCHMARKS.items():
            assert "Naive2" in methods, f"{freq} missing Naive2"
            assert "Smyl_winner" in methods, f"{freq} missing Smyl_winner"

    def test_winner_beats_naive_on_smape(self) -> None:
        for freq in M4_BENCHMARKS:
            naive_smape = M4_BENCHMARKS[freq]["Naive2"]["smape"]
            winner_smape = M4_BENCHMARKS[freq]["Smyl_winner"]["smape"]
            assert winner_smape < naive_smape, f"{freq}: winner not better than Naive2"

    def test_m4_target_lookup_hourly_ets(self) -> None:
        target = m4_target("Hourly", "ETS")
        assert target["smape"] == 17.307
        assert target["mase"] == 3.443

    def test_m4_target_raises_on_unknown_method(self) -> None:
        with pytest.raises(KeyError):
            m4_target("Hourly", "NotARealModel")


class TestWithinTolerance:
    def test_score_better_than_target_is_within_tolerance(self) -> None:
        assert within_tolerance(our_score=1.0, target_score=2.0, tolerance_pct=10.0)

    def test_score_equal_to_target_is_within_tolerance(self) -> None:
        assert within_tolerance(our_score=2.0, target_score=2.0, tolerance_pct=10.0)

    def test_score_just_under_tolerance(self) -> None:
        # target 2.0, tolerance 10% => upper bound 2.2
        assert within_tolerance(our_score=2.19, target_score=2.0, tolerance_pct=10.0)

    def test_score_over_tolerance_fails(self) -> None:
        assert not within_tolerance(our_score=2.21, target_score=2.0, tolerance_pct=10.0)
