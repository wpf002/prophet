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

## Deploy & monitoring scaffolding (built)

- **Railway deploy.** `Dockerfile` (installs `libgomp1` for LightGBM, uv-frozen
  deps), `.dockerignore`, `railway.toml` (healthcheck `/health`), and
  `scripts/entrypoint.sh` — on boot it ingests data + trains the model if the
  artifact is absent, then serves. Set `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`
  (+ optional `PROPHET_TICKERS`, `PROPHET_MONITOR_DSN`) as Railway vars and push.
- **Forecast-vs-actual logging.** `/forecast` logs each served forecast to the
  `forecasts` table in a background task when `PROPHET_MONITOR_DSN` is set
  (non-fatal). `scripts/record_actuals.py` (run on a Railway cron) re-fetches
  market data, upserts `actuals`, and reports rolling accuracy + coverage.
- **Dashboard.** `dashboard/app.py` (Streamlit, optional `dashboard` extra) over
  `ROLLING_ACCURACY_SQL`: per-series MAE, 95% coverage, pair counts.
  `uv run --extra dashboard streamlit run dashboard/app.py`.

## What's genuinely left (time-bound, not code)

1. **Run the deploy** — `railway up` with the env vars above (your account).
2. **30 days of accumulation** — let the forecast/actual loop run; calendar time.
3. The plain-language accuracy report then falls out of the dashboard.

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
