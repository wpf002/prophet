# Macro (Bloomberg / FRED) — the Step-4 win

**Date:** 2026-06-14 · **Source:** FRED public CSV (no API key) · **Connector:**
`scripts/ingest_macro.py` · **Domain:** `macro`

This is the ecosystem-plan Step-4 gate, **cleared**: a real app data source where
forecasting clearly beats naive *and* drives a decision. Found by surveying every
app under `~/Documents/GitHub` — Bloomberg already pulls FRED macro series
(`backend/data/sources/fred_source.py`), which have decades of history.

## Data

- **CPIAUCSL** (CPI, monthly, from 1947) and **UNRATE** (unemployment rate,
  monthly, from 1948) — ~950 points each. Pulled read-only from FRED's public
  CSV. FRED occasionally omits a month (a `.` placeholder, e.g. a delayed
  release — Oct 2025 was missing); the connector reindexes to a complete
  month-start grid and linearly interpolates the gap.

## Verdict (the gate)

MASE, 5-window expanding CV, horizon 12 months (lower is better):

| Model            |   MASE | vs Naive |
|------------------|-------:|---------:|
| Naive            | 0.7399 | floor    |
| RWD              | 0.4672 | −37%     |
| **AutoETS**      | **0.1690** | **−77%** |
| AutoARIMA        | 0.2011 | −73%     |
| LightGBM (tuned) | 0.4029 | −46%     |

Forecasting **decisively** beats naive — the opposite of Crossbar (martingale)
and raw stock prices (random walk). All four filter questions pass: real monthly
series, decades of history, beats naive by 77%, and inflation/unemployment
nowcasts have obvious lead-time decision value.

## What's deployed

- `macro` DomainSpec (freq `MS`, horizon 12, seasonality 12) in
  `src/prophet/data/domains.py`.
- A trained production model (`models/production/macro/`) served through the
  multi-model API — verified end-to-end: `GET /models` lists `macro`, and
  `forecast_series` returns UNRATE projections with 80% prediction intervals.
- `scripts/entrypoint.sh` builds the macro model on container boot (no
  credentials needed), so the hosted API serves it automatically.

The served model is **AutoETS** (MASE 0.17 — the benchmark winner). The registry
now supports two engines: MLForecast/LightGBM (default, e.g. market-vol) and
StatsForecast/AutoETS (`--method statistical`, persisted via `StatsForecast.save`
and loaded by `engine: "statsforecast"` in metadata). StatsForecast's conformal
intervals are calibrated for the full trained horizon, so serving predicts at
that horizon and slices to the request.

## How Bloomberg consumes it

Bloomberg calls the Prophet API (or `@prophet/client`) with
`model: "macro"`, `series: "CPIAUCSL" | "UNRATE"` to get a forward projection +
intervals for its macro dashboard.

## How to reproduce

```bash
uv run python scripts/ingest_macro.py
uv run python scripts/run_benchmark.py --dataset domain-macro --models statistical
uv run python scripts/train_production.py --dataset macro
```
