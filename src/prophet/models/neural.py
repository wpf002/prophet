"""Neural forecasting models (Phase 4) — the kill-or-keep decision phase.

NHITS, Temporal Fusion Transformer (TFT), and PatchTST via NeuralForecast.
Each is trained as its own NeuralForecast so we can time fit/predict per model
and fall back from MPS (Apple GPU) to CPU independently. The point of the phase
is an honest comparison against the Phase 3 LightGBM (MASE 0.934): do neural
architectures justify their training cost on this data shape?

Forecasts come back as Polars long format with ``ds`` as ``Datetime("us","UTC")``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl

MODEL_NAMES = ["NHITS", "TFT", "PatchTST"]


@dataclass(frozen=True)
class NeuralTiming:
    """Per-model timing and the device training actually ran on."""

    fit_seconds: float
    predict_seconds: float
    device: str


@dataclass(frozen=True)
class NeuralInfo:
    """Metadata from a neural forecast run."""

    timings: dict[str, NeuralTiming]
    input_size: int
    max_steps: int
    failures: dict[str, str] = field(default_factory=dict)


def _fit_one(
    name: str,
    model_cls: Any,
    train_pd: Any,
    *,
    horizon: int,
    input_size: int,
    max_steps: int,
    scaler_type: str,
    freq: str,
    seed: int,
    accelerator: str,
) -> tuple[pl.DataFrame | None, NeuralTiming | None, str | None]:
    """Train and predict a single neural model. Returns (forecast, timing, error).

    Falls back to CPU if the preferred accelerator (e.g. MPS) raises — some
    transformer ops aren't implemented on every backend.
    """
    from neuralforecast import NeuralForecast

    devices_to_try = [accelerator] if accelerator == "cpu" else [accelerator, "cpu"]
    last_err: str | None = None
    for device in devices_to_try:
        try:
            model = model_cls(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                scaler_type=scaler_type,
                random_seed=seed,
                accelerator=device,
                enable_progress_bar=False,
                logger=False,
            )
            nf = NeuralForecast(models=[model], freq=freq)
            t0 = time.perf_counter()
            nf.fit(df=train_pd)
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            forecast_pd = nf.predict()
            predict_s = time.perf_counter() - t1
            forecast = pl.from_pandas(forecast_pd).select(
                pl.col("unique_id").cast(pl.String),
                pl.col("ds").cast(pl.Datetime("us", "UTC")),
                pl.col(name).cast(pl.Float64),
            )
            return forecast, NeuralTiming(fit_s, predict_s, device), None
        except Exception as exc:  # report and try the next device
            last_err = f"{type(exc).__name__}: {exc}"
    return None, None, last_err


def forecast_neural(
    train_df: pl.DataFrame,
    *,
    horizon: int,
    season_length: int,
    freq: str,
    input_size: int | None = None,
    max_steps: int = 1000,
    scaler_type: str = "standard",
    accelerator: str = "mps",
    seed: int = 42,
) -> tuple[pl.DataFrame, NeuralInfo]:
    """Train NHITS, TFT, and PatchTST and return their forecasts plus timings.

    Args:
        train_df: Training data (unique_id, ds, y) in Polars long format.
        horizon: Number of steps to forecast.
        season_length: Seasonal period (24 for hourly).
        freq: Pandas frequency string (e.g. "h").
        input_size: Look-back window. Defaults to 7 * season_length (a weekly
            context for hourly data).
        max_steps: Training steps per model.
        scaler_type: NeuralForecast per-series scaler ("standard", "robust", ...).
        accelerator: Preferred device ("mps", "cpu", "gpu"); falls back to CPU.
        seed: Random seed for reproducibility.

    Returns:
        (forecasts, info). forecasts has columns unique_id, ds, and one column per
        successfully-trained model; info carries per-model timing + any failures.
    """
    from neuralforecast.models import NHITS, TFT, PatchTST

    if input_size is None:
        input_size = 7 * season_length

    specs: dict[str, Any] = {"NHITS": NHITS, "TFT": TFT, "PatchTST": PatchTST}
    train_pd = train_df.sort(["unique_id", "ds"]).to_pandas()

    forecasts: pl.DataFrame | None = None
    timings: dict[str, NeuralTiming] = {}
    failures: dict[str, str] = {}

    for name, cls in specs.items():
        forecast, timing, error = _fit_one(
            name,
            cls,
            train_pd,
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
            scaler_type=scaler_type,
            freq=freq,
            seed=seed,
            accelerator=accelerator,
        )
        if forecast is None or timing is None:
            failures[name] = error or "unknown error"
            continue
        timings[name] = timing
        forecasts = (
            forecast
            if forecasts is None
            else forecasts.join(forecast, on=["unique_id", "ds"], how="inner")
        )

    if forecasts is None:
        raise RuntimeError(f"All neural models failed to train: {failures}")

    info = NeuralInfo(
        timings=timings,
        input_size=input_size,
        max_steps=max_steps,
        failures=failures,
    )
    return forecasts, info
