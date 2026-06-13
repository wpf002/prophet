"""Tests for the applied-domain loader (Phase 5).

Verifies the source-agnostic Parquet contract without any live data source: write
a synthetic long-format split to the expected path, then load it back.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from prophet.data.domains import DOMAIN_SPECS, load_domain
from prophet.data.loaders import load_synthetic
from prophet.evaluation.cross_validation import split_train_test


def _materialize(data_dir: Path, name: str) -> None:
    df = load_synthetic(n_series=4, n_obs=200, frequency="1h", seed=42)
    train, test = split_train_test(df, horizon=24)
    domain_dir = data_dir / "domains"
    domain_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(domain_dir / f"{name}-train.parquet")
    test.write_parquet(domain_dir / f"{name}-test.parquet")


class TestLoadDomain:
    def test_roundtrip(self, tmp_path: Path) -> None:
        _materialize(tmp_path, "casino")
        train, test = load_domain("casino", tmp_path)
        assert set(train.columns) >= {"unique_id", "ds", "y"}
        assert train["unique_id"].n_unique() == 4
        assert test["unique_id"].n_unique() == 4

    def test_sample_n_caps_series(self, tmp_path: Path) -> None:
        _materialize(tmp_path, "casino")
        train, _ = load_domain("casino", tmp_path, sample_n=2)
        assert train["unique_id"].n_unique() == 2

    def test_unknown_domain_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            load_domain("does-not-exist", tmp_path)

    def test_missing_files_raise(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_domain("casino", tmp_path)

    def test_validates_required_columns(self, tmp_path: Path) -> None:
        domain_dir = tmp_path / "domains"
        domain_dir.mkdir(parents=True)
        bad = pl.DataFrame({"unique_id": ["a"], "ds": [1], "value": [2.0]})
        bad.write_parquet(domain_dir / "casino-train.parquet")
        bad.write_parquet(domain_dir / "casino-test.parquet")
        with pytest.raises(ValueError, match="missing columns"):
            load_domain("casino", tmp_path)


def test_specs_are_well_formed() -> None:
    for name, spec in DOMAIN_SPECS.items():
        assert spec.name == name
        assert spec.horizon >= 1
        assert spec.seasonality >= 1
        assert spec.freq
