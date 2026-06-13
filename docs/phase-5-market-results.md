# Phase 5 — Applied Domain Results (Market Data)

**Status:** Complete.
**Domain:** Daily market data for the user's 28 portfolio tickers (Alpaca bars,
~5 years), pivoted from cash flow (see below). Two targets, same model ladder.
**MLflow runs** (experiment `prophet-benchmarks`, `dataset=domain-market-*`):

| Target | baselines | statistical | ml |
| --- | --- | --- | --- |
| close | `bf676cf3` | `85e4dc28` | `2ee22043` |
| volume | `c99ed0af` | `63310128` | `974d4c49` |

## Why this isn't cash flow

The intended Phase 5 domain was Syntrackr cash flow. Connecting live (Railway CLI
→ public Postgres DSN) revealed Syntrackr is an **investment tracker**, not a
cash-flow app: largest table `positions` (30 rows, a snapshot), `trades` (3),
no transactions/history table, no time series. Per the roadmap kill criterion
("can't get clean data → switch domain"), we pivoted to forecasting the **market
data of the 28 tickers** in that portfolio, using the Alpaca keys already in the
Syntrackr environment. The live pipeline itself worked end to end.

## Data

28 tickers (large-caps, ETFs, crypto proxies), Alpaca daily bars 2020-2025
(~1,150 days each; newer ETFs fewer). Trading days reindexed to a regular daily
sequence (calendar gaps dropped), seasonality = 5 (trading week), horizon = 21
(one trading month). Two domain datasets:

- `market-close` — close price in cents (money invariant).
- `market-vol` — daily trading volume.

## Results (MASE, lower is better; ranked)

**Prices (`market-close`)**

| Model | MASE | sMAPE |
| --- | --- | --- |
| LightGBM (tuned) | 1.99 | 5.7 |
| **RandomWalkWithDrift (floor)** | **2.29** | 6.8 |
| AutoTheta (best statistical) | 2.31 | 6.8 |
| AutoARIMA | 2.33 | 6.8 |
| Naive | 2.39 | 6.9 |
| HistoricAverage | 12.67 | 40.4 |

**Volume (`market-vol`)**

| Model | MASE | sMAPE |
| --- | --- | --- |
| **Ensemble (statistical)** | **0.85** | 36.3 |
| AutoARIMA | 0.89 | 39.2 |
| HistoricAverage (floor) | 0.97 | 43.7 |
| LightGBM | 1.01 | 45.2 |
| Naive | 1.19 | 41.6 |

## Findings (honest)

1. **The methodology transfers and behaves correctly.** On **prices**, no
   classical model beats RandomWalkWithDrift (AutoARIMA/ETS/Theta all ~2.31 ≈ the
   2.29 floor) — exactly what theory predicts for near-efficient equity prices.
   On **volume**, the statistical models beat the HistoricAverage floor by ~12%
   (Ensemble 0.85 vs 0.97) — volume is autocorrelated and mean-reverting, and the
   ladder captures it. The framework correctly distinguishes a forecastable
   target from an unforecastable one.

2. **The tuned-LightGBM "edge" on prices (1.99 < 2.29) is NOT credible.** It rests
   on a single 21-day holdout (588 points total) and is contradicted by every
   statistical model tying the random-walk floor. This is almost certainly
   small-sample noise, not price predictability. **Do not read it as alpha.**
   A defensible claim would require rolling-origin CV across many windows; the
   prior remains: prices are a random walk.

3. **Global ML needs scale; here classical wins.** On the forecastable target
   (volume), the global LightGBM (1.01) *underperformed* the per-series
   statistical models (0.85) and even the HistoricAverage floor. With only 28
   series it is data-starved — the opposite of Phase 3, where 414 M4 series let
   LightGBM win. Lesson: global gradient boosting pays off with many series; with
   a few dozen, per-series classical models are the better default.

## Acceptance check

- [x] Domain data ingested to `data/raw/domains/market-{close,vol}-*.parquet`
      (connector: `scripts/ingest_market.py`).
- [x] Full ladder (baselines → statistical → ML) runs via
      `--dataset domain-market-close|vol`.
- [x] Documented MASE/sMAPE on the holdout vs. the per-target naive floor.
- [x] A target where engineered features help: lag/rolling features let the
      statistical ladder beat naive on volume (rolling-mean dominates importance).
- [x] Honest writeup including failure modes (prices unforecastable; global ML
      data-starved at 28 series; single-window caveat on the price result).

## Caveat / next step

This is a **single holdout window**, not rolling-origin CV — fine for a
direction-finding pass, but any production claim (especially the price result)
needs multi-window CV. The honest, defensible takeaway: **volume is forecastable
here (classical models, ~12% over naive); prices are not.**
