# Phase 3 — ML Models, LightGBM via MLForecast (Results)

**Status:** Complete — acceptance criteria met.
**Dataset:** M4 Hourly (414 series, horizon 48, seasonality 24).
**MLflow run:** `042fba2d2f644407a0dca58d0441be96` (experiment `prophet-benchmarks`,
tags `phase=3`, `dataset=m4-hourly`). Artifacts: `feature_importance.json`,
`best_params.json`.

## What was built

- `forecast_ml` — one global LightGBM via MLForecast trained across all series on:
  - **lags** `[1, 24, 48, 168]`
  - **rolling mean & std** over windows `[24, 168]` (lagged by 1, no leakage)
  - **calendar features** hour, day-of-week, day-of-month, month, is-weekend
  - **target transforms** seasonal differencing at 24 + per-series standardization
  - Untuned (library defaults) and **Optuna-tuned** (50 trials, minimize CV MASE).
- `run_benchmark.py --models ml` (phase=3): logs CV MASE, Optuna best params, and
  feature importance to MLflow; `--tune/--no-tune`, `--n-trials`, `--n-jobs` flags.

## Headline results (M4 Hourly, official test split, ranked by MASE)

| Rank | Model | MASE | sMAPE | WAPE |
| --- | --- | --- | --- | --- |
| 1 | **LightGBM_tuned** | **0.9336** | 12.99 | 12.15 |
| 2 | LightGBM (untuned) | 0.9718 | 14.35 | 12.89 |

Reference points: AutoARIMA 0.948 (best Phase 2), SeasonalNaive floor 1.193,
M4 winner (Smyl) 0.893. CV MASE of the tuned model: 0.930.

**Tuned LightGBM (0.9336) beats the best statistical model (AutoARIMA 0.948)** and
lands within **~4.6%** of the competition winner — comfortably inside the 20%
target. Tuning improved on the untuned model (0.972 → 0.934).

Best params (Optuna, 50 trials): `n_estimators=326`, `learning_rate=0.0172`,
`num_leaves=237`, `min_child_samples=38`, `subsample=0.917`,
`colsample_bytree=0.575`, `reg_alpha=2.36`, `reg_lambda=5.17`.

Top features by gain: `lag24`, `rolling_mean_lag1_window_size168`,
`rolling_mean_lag1_window_size24`, `lag48`, `rolling_std_lag1_window_size24`,
`lag168`, `hour`, `rolling_std_lag1_window_size168` — seasonal lags and rolling
stats do the work, as expected.

## The real lesson: a global GBM needs the right target transforms

The roadmap's feature recipe (lags + rolling + calendar) alone produced a broken
model — untuned MASE **8.6** (raw targets) because a single global GBM is
dominated by large-magnitude M4 series. Two target transforms fixed it:

| Recipe (untuned, full 414) | MASE |
| --- | --- |
| raw target | 8.56 |
| LocalStandardScaler | 2.00 |
| **Differences([24]) + LocalStandardScaler** | **0.972** |
| Differences([1]) + scaler | 1.29 |
| Differences([1, 24]) + scaler | 1.13 |

Per-series **standardization** removes the scale heterogeneity; **seasonal
differencing at 24** removes the dominant daily cycle so the model only has to
predict the residual. This recipe took the untuned model from unusable (8.6) to
already-beating-SeasonalNaive (0.972), and is the substantive Phase 3 finding.

## Setup note

LightGBM needs the OpenMP runtime, which isn't bundled on macOS — installed via
`brew install libomp` (a system library, hence brew not uv). Required once per
machine for `--models ml`.

## Caveats

- **Calendar features are synthetic.** M4 series carry only an integer step index;
  our UTC timestamps encode position-modulo-24h/168h, not real clock time. So
  `hour`/`dayofweek` are consistent within-series phase markers, not real calendar
  effects — the lag features carry the real seasonal signal. Harmless, and the
  model uses them where they help.
- The tuned vs untuned gap is modest (0.972 → 0.934); most of the win comes from
  the feature/transform recipe, not the hyperparameter search.

## Acceptance check

- [x] LightGBM (untuned) runs reproducibly via `--models ml --no-tune`.
- [x] **Tuned LightGBM beats the best Phase 2 statistical model** (0.9336 < 0.948).
- [x] Optuna study (50 trials) run; best params + feature importance logged to MLflow.
- [x] Tuned CV/test MASE within 20% of the M4 winner (0.9336 vs 0.893 → ~4.6%).
- [x] Lint, typecheck, tests green (50 passed).
