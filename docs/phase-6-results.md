# Phase 6 — Productionize (Results)

**Status:** Code-complete and verified locally. The deployable artifact exists:
a persisted model served behind a real `/forecast` API with prediction intervals,
plus drift-detection and forecast-vs-actual logging. The remaining items are
inherently external/time-bound (live Railway deploy, 30 days of accumulated
accuracy) — see "What's left".

## What was built

- **Production model + registry.** `scripts/train_production.py` fits the winning
  approach (LightGBM via MLForecast, the Phase 3 recipe) on a domain's full
  history with **conformal prediction intervals** and persists it + metadata to
  `models/production/<dataset>/`. `prophet/serving/registry.py` loads it once,
  caches it in memory, and serves a single series with `forecast_series`.
- **Real `POST /forecast`.** Accepts `series_id`, `horizon`, optional `level`;
  returns point forecast and lo/hi bounds per level. 503 if no model loaded, 404
  for an unknown series, 400 if horizon exceeds the model's calibrated horizon,
  422 for invalid input. Non-negative targets (volume) are clipped at 0. The
  model warms on app startup (`lifespan`).
- **Monitoring.** `prophet/monitoring/drift.py` — PSI (input-distribution drift,
  alert > 0.25) and a rolling-MASE check (alert when recent/baseline > 1.5x),
  combined in `check_drift`. `prophet/monitoring/schema.py` — `forecasts` /
  `actuals` tables (DDL), a rolling-accuracy + 95%-coverage query, and a
  table-creation helper.
- **Tests** (68 total, all green): serving happy-path + 404/400/503, PSI and
  drift alerts, API wiring.

## Verified locally

Served the `market-vol` model (28 portfolio tickers, horizon 21, conformal
intervals). Live HTTP:

```
POST /forecast {"series_id":"AAPL","horizon":3,"level":[95]}  -> 200
  yhat=2,056,652  lo95=1,510,643  hi95=2,602,661   (model market-vol:LightGBM)
POST /forecast {"series_id":"ZZZZ", ...}                       -> 404
POST /forecast {"series_id":"AAPL","horizon":50}               -> 400
GET  /health                                                   -> 200
```

## What's left (external / time-bound)

1. **Deploy to Railway.** Prophet needs a Dockerfile + `railway.toml` (start:
   `uvicorn prophet.api.main:app`). The model artifact and its data are gitignored,
   so the image must either bake in a release-time `ingest_market` + `train_production`
   step (needs the Alpaca keys as Railway vars) or load the artifact from object
   storage / the MLflow registry on startup.
2. **30 days of forecast-vs-actual.** Log every served forecast to `forecasts`,
   record realized values into `actuals`, run the nightly join, and accumulate the
   rolling-accuracy history. This is calendar time, not code.
3. **Dashboard.** A small Streamlit/Next.js page over `ROLLING_ACCURACY_SQL`
   (rolling MASE + coverage + latest drift status).

## Honest caveats

- **Forecast dates are synthetic.** The market datasets reindex trading days to a
  regular sequence from a 2000 epoch (calendar gaps dropped), so `/forecast`
  returns 2000-based `ds` representing *future trading-day steps*, not real
  calendar dates. Mapping back to real dates (store the last real date per series,
  offset by business days) is a small, documented follow-up.
- **The served model is `market-vol`** (the forecastable Phase 5 target). On that
  target the statistical ensemble was marginally better than LightGBM, but the
  global MLForecast LightGBM is the production-shaped choice (one fast artifact +
  calibrated conformal intervals).

## Definition-of-done status

Per the roadmap, "Prophet is done" needs: a deployed API (code ready; deploy
pending), 30+ days within bounds (needs calendar time), and survival of
practitioner review (correct CV, honest metrics, defensible choices, documented
failures — met across phases 1-5). The software is built and tested; the final
two are deploy + elapsed time.
