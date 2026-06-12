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

## Phase 1 — Naive baselines

**Goal:** Reproduce naive baseline scores on M4 hourly subset within tolerance of published values.

**Models:**
- Naive (last value)
- SeasonalNaive (last seasonal value)
- HistoricAverage
- Drift method

**Acceptance:**
- M4 hourly subset downloaded, preprocessed, stored as Parquet
- All four baselines run end-to-end via `make benchmark`
- MLflow run logged with MASE, sMAPE, WAPE per model
- SeasonalNaive MASE within 5% of published M4 baseline scores

**Kill criteria:**
- Cannot reproduce SeasonalNaive within 10% of published score (means data pipeline is broken)
- Single benchmark run takes >30 minutes on a laptop (means we have a scale problem before adding real models)

---

## Phase 2 — Statistical models

**Goal:** Beat SeasonalNaive consistently with AutoARIMA, AutoETS, and Theta.

**Models:**
- AutoARIMA (statsforecast)
- AutoETS (statsforecast)
- Theta / DynamicOptimizedTheta
- Simple ensemble (mean of top-3)

**Acceptance:**
- All three models run on M4 hourly subset via `make benchmark`
- At least two of three beat SeasonalNaive on MASE
- Ensemble beats best individual model
- Results logged to MLflow with full hyperparameters

**Kill criteria:**
- No statistical model beats SeasonalNaive (would indicate fundamentally bad data prep or evaluation)

---

## Phase 3 — ML models

**Goal:** Beat best statistical model with LightGBM-based ML approach using lag features.

**Models:**
- LightGBM via MLForecast with:
  - Lag features (1, 7, 14, 28 for daily; 1, 24, 168 for hourly)
  - Rolling mean / std features
  - Calendar features (day of week, day of month, month, quarter)
- LightGBM with Optuna hyperparameter tuning

**Acceptance:**
- MLForecast pipeline runs reproducibly
- Tuned LightGBM beats best Phase 2 statistical model on MASE
- Feature importance logged
- Optuna study artifacts stored in MLflow

**Kill criteria:**
- ML model cannot beat statistical baselines with reasonable tuning (means the problem is genuinely statistical-favorable and we should not invest in neural)

---

## Phase 4 — Neural models

**Goal:** Evaluate whether neural architectures justify their cost on this data.

**Models:**
- NHITS
- Temporal Fusion Transformer (TFT)
- PatchTST

**Acceptance:**
- All three trained on M4 hourly via NeuralForecast
- Honest comparison logged: best neural vs. best ML vs. best statistical
- Decision documented in ROADMAP.md: do neurals justify production cost or not?

**Kill criteria:**
- Best neural model worse than tuned LightGBM and slower by 10x (means kill neural for this domain — log the result and move on, don't sink more time)

---

## Phase 5 — Applied domain

**Goal:** Port the methodology to a problem with real stakes.

**Candidate domains (pick one):**
- Casino table game revenue forecasting (hourly, multi-game, multi-day-of-week seasonality)
- Sports betting line movement / closing line value drift (intraday, event-keyed)
- Personal cash flow forecasting via Syntrackr data (monthly, intervention scenarios)

**Acceptance:**
- Domain data ingested and preprocessed
- Same model ladder run on domain data
- Documented accuracy on holdout period with realistic forecast horizon
- Honest writeup: did this work? Where does it fail?

**Kill criteria:**
- Best model worse than seasonal naive on domain data (means the data is too noisy or the problem isn't tractable with available features)

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
