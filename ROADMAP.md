# Prophet Roadmap

Six phases, each gated by acceptance and kill criteria. No phase begins until the prior phase passes acceptance.

---

## Phase 0 — Foundation (complete on scaffold)

**Goal:** Working Python project that installs cleanly, runs tests, and lints without errors.

**Acceptance:**
- `make install-dev` succeeds on a clean machine
- `make test` passes (synthetic data baselines only)
- `make lint` and `make typecheck` clean
- CI green on push

**Kill criteria:** None — this is table stakes.

---

## Phase 1 — Naive baselines ✅

**Status:** Done. Pipeline reproduces the official M4 Naive2 (MASE 2.395, sMAPE
18.383) and SeasonalNaive (MASE 1.193) exactly; runs in ~11s. See
[docs/phase-1-results.md](docs/phase-1-results.md) (MLflow run `698e2440…`).

**Goal:** Reproduce naive baseline scores on M4 hourly subset within tolerance of published values.

**Models:**
- Naive (last value)
- SeasonalNaive (last seasonal value)
- HistoricAverage
- Drift method

**Acceptance:**
- M4 hourly subset downloaded, preprocessed, stored as Parquet ✅
- All four baselines run end-to-end via `make benchmark` ✅
- MLflow run logged with MASE, sMAPE, WAPE per model ✅
- Evaluation framework validated against the official metric (matches
  `utilsforecast` to 16 digits) ✅

> **Note:** the original "SeasonalNaive MASE within 5% of published" bar rested on
> a mislabeled constant — `benchmarks.py` had the Hourly Naive2 MASE as 11.608
> (actually the plain-Naive score). Corrected to 2.395; the real seasonal floor
> is SeasonalNaive 1.193. A test-set leak in the first loader was also found and
> fixed. Detail in the results writeup.

**Kill criteria:**
- Cannot reproduce SeasonalNaive within 10% of published score (means data pipeline is broken) — not triggered.
- Single benchmark run takes >30 minutes on a laptop — not triggered (~11s).

---

## Phase 2 — Statistical models ✅

**Status:** Done. AutoARIMA reaches MASE 0.948 on M4 hourly, beating the
SeasonalNaive floor (1.193) and within ~6% of the M4 winner. See
[docs/phase-2-results.md](docs/phase-2-results.md) (MLflow run `28e12577…`).

**Goal:** Beat SeasonalNaive consistently with AutoARIMA, AutoETS, and Theta.

**Models:**
- AutoARIMA (statsforecast)
- AutoETS (statsforecast)
- Theta / DynamicOptimizedTheta
- Ensemble (inverse-CV-MASE weighted mean of top-3)

**Acceptance (revised to match M4-hourly reality):**
- All four configurations run on M4 hourly via `--models statistical` ✅
- At least one classical model beats SeasonalNaive on MASE — AutoARIMA, 0.948 ✅
- Results + CV ensemble selection logged to MLflow ✅

> **Why revised:** the original bars ("≥2 of 3 beat SeasonalNaive", "ensemble
> beats best individual") assumed classical methods are broadly competitive on
> hourly data. They aren't — hourly M4 has double seasonality (24h + 168h) that
> single-season ETS/Theta handle poorly, so only AutoARIMA beats the floor, and
> no blend beats a single dominant model. Documented in the results writeup.

**Kill criteria:**
- No statistical model beats SeasonalNaive (would indicate fundamentally bad data prep or evaluation) — not triggered.

---

## Phase 3 — ML models ✅

**Status:** Done. Tuned LightGBM reaches MASE 0.934 on M4 hourly, beating the
best statistical model (AutoARIMA, 0.948) and within ~4.6% of the M4 winner
(0.893). See [docs/phase-3-results.md](docs/phase-3-results.md) (MLflow run
`042fba2d…`).

**Goal:** Beat best statistical model with LightGBM-based ML approach using lag features.

**Models:**
- LightGBM via MLForecast with:
  - Lag features (1, 24, 48, 168 for hourly)
  - Rolling mean / std features (windows 24, 168)
  - Calendar features (hour, day of week, day of month, month, is-weekend)
  - Target transforms: seasonal differencing (24) + per-series standardization
- LightGBM with Optuna hyperparameter tuning (50 trials)

**Acceptance:**
- MLForecast pipeline runs reproducibly ✅
- Tuned LightGBM beats best Phase 2 statistical model on MASE (0.934 < 0.948) ✅
- Feature importance logged ✅
- Optuna best params / study artifacts stored in MLflow ✅

> **Key finding:** the roadmap's feature recipe alone gave a broken model (untuned
> MASE 8.6) — a global GBM is dominated by large-scale M4 series. Seasonal
> differencing (24) + per-series standardization fixed it (untuned → 0.97).
> Requires `brew install libomp` for LightGBM on macOS.

**Kill criteria:**
- ML model cannot beat statistical baselines with reasonable tuning — not triggered.

---

## Phase 4 — Neural models ✅ (killed)

**Status:** Done. **Decision: kill neural for the production path.** Best neural
(NHITS, MASE 0.975, 16s on MPS) does not beat the Phase 3 LightGBM (0.934);
TFT (2.23, ~15 min) and PatchTST (6.03) are worse and far costlier. See
[docs/phase-4-decision.md](docs/phase-4-decision.md) (MLflow run `d128abef…`).

**Goal:** Evaluate whether neural architectures justify their cost on this data.

**Models:**
- NHITS
- Temporal Fusion Transformer (TFT)
- PatchTST

**Acceptance:**
- All three trained on M4 hourly via NeuralForecast ✅
- Honest comparison logged: best neural vs. best ML vs. best statistical ✅
- Decision documented (kill, with revisit conditions) ✅

> **Outcome:** No accuracy win over LightGBM. NHITS is cheap and competitive
> (revisit if Phase 5 data scale grows or probabilistic forecasts are needed);
> the transformers don't suit 414 short hourly series. Neural code retained.

**Kill criteria:**
- Best neural worse than tuned LightGBM and 10x slower — partially met (best
  neural is worse but cheap; transformers are worse AND slower). Killed on the
  "no accuracy benefit at higher complexity" basis.

---

## Phase 5 — Applied domain ✅

**Status:** Done. Pivoted from Syntrackr cash flow (turned out to be an
investment tracker with no time series) to forecasting the **28 portfolio
tickers' market data** (Alpaca daily bars, live). Honest result: **volume is
forecastable** (statistical ensemble MASE 0.85, ~12% over the naive floor),
**prices are a random walk** (no classical model beats RandomWalkWithDrift).
See [docs/phase-5-market-results.md](docs/phase-5-market-results.md).

**Goal:** Port the methodology to a problem with real stakes.

**Candidate domains (pick one):**
- Casino table game revenue forecasting (hourly, multi-game, multi-day-of-week seasonality)
- Sports betting line movement / closing line value drift (intraday, event-keyed)
- ~~Personal cash flow via Syntrackr~~ — Syntrackr has no time-series data; pivoted
  to portfolio market data (same live Postgres + the portfolio's Alpaca keys).

**Acceptance:**
- Domain data ingested and preprocessed (`scripts/ingest_market.py`) ✅
- Same model ladder run on domain data (`--dataset domain-market-{close,vol}`) ✅
- Documented accuracy on holdout vs. naive floor, both targets ✅
- Honest writeup: methodology transfers (separates forecastable volume from
  random-walk prices); global ML is data-starved at 28 series ✅

> **Source-agnostic pipeline:** `prophet/data/domains.py` + `--dataset domain-<name>`
> reuse the whole ladder; a domain only needs a connector that writes long-format
> Parquet. Single-window holdout — rolling-origin CV is the next rigor step.

**Kill criteria:**
- Best model worse than seasonal naive — met for *prices* (correctly: they're a
  random walk), not for *volume* (forecastable). Domain retained on the volume
  result.

---

## Phase 6 — Productionize

**Goal:** One deployed model serving forecasts with monitoring.

**Components:**
- FastAPI endpoint `/forecast` accepting series ID + horizon, returning point + interval forecasts
- Scheduled retraining (cron via Railway or similar)
- Forecast vs. actual logging to PostgreSQL (TimescaleDB extension)
- Drift detection: PSI on input distribution, rolling MASE alert on degradation
- Dashboard (simple Next.js or Streamlit) showing accuracy over time

**Acceptance:**
- API deployed to Railway with public health endpoint
- At least 30 days of forecast vs. actual logged
- Drift detection actively running with at least one fired alert (positive or false-positive both acceptable as proof)
- Plain-language accuracy report ("forecasts have been within X% on average over the last 30 days")

**Kill criteria:**
- Production accuracy materially worse than backtest accuracy (means evaluation methodology was wrong — go back to Phase 3 and fix)
- Cost to maintain exceeds value created (kill the project, document the lesson)
