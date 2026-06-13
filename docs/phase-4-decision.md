# Phase 4 — Neural Models: Kill-or-Keep Decision

**Status:** Complete. **Decision: kill neural for Prophet's production path** (with
a documented revisit condition).
**Dataset:** M4 Hourly (414 series, horizon 48, seasonality 24).
**MLflow run:** `d128abefab6d4ee8a3c8b0a5a0322fda` (tags `phase=4`,
`dataset=m4-hourly`). Trained on Apple MPS (GPU), `max_steps=1000`,
`input_size=168`.

## What was run

NHITS, TFT, and PatchTST via NeuralForecast, each trained as its own model so
fit/predict time is measured per model. All three trained on MPS (no CPU
fallback needed). `forecast_neural` falls back MPS→CPU per model if a backend op
is unsupported.

## Results vs. the rest of the ladder

| Model | MASE | Training time | Phase |
| --- | --- | --- | --- |
| **LightGBM (tuned)** | **0.9336** | ~5 min (50 Optuna trials) | 3 |
| AutoARIMA | 0.948 | (per-series) | 2 |
| **NHITS** | **0.9748** | **16 s** | 4 |
| SeasonalNaive (floor) | 1.193 | — | 1 |
| TFT | 2.226 | **874 s (~15 min)** | 4 |
| PatchTST | 6.030 | 101 s | 4 |

M4 winner reference: 0.893.

## The decision

**No neural model beats the Phase 3 LightGBM (0.9336).** The picture by model:

- **NHITS** is the surprise: **MASE 0.975 in just 16 seconds** on the GPU. It
  beats the SeasonalNaive floor and is within ~4% of LightGBM — genuinely
  competitive and *cheap*. But it does not beat LightGBM or AutoARIMA, so it
  earns no place in production where a simpler, better model already exists.
- **TFT** is the clearest reject: **MASE 2.23 and ~15 minutes to train** — worse
  than the naive-ish baselines and ~55× slower than NHITS for a far worse result.
- **PatchTST** is worse still (MASE 6.03), below even SeasonalNaive — M4 hourly's
  414 short series don't give a patching transformer enough to work with.

Against the roadmap's kill criterion ("best neural worse than tuned LightGBM AND
10× slower → kill"): the *best* neural (NHITS) is worse than LightGBM but **not**
10× slower — it's actually fast. The transformers are both worse *and* far
slower. Net: neural offers **no accuracy win** here, and the only competitive
neural model (NHITS) merely ties-ish at higher operational complexity (a torch +
Lightning + GPU stack vs. a single LightGBM). **For Prophet's production path,
LightGBM stays the model to deploy; neural is killed for now.**

This matches the roadmap's stated expectation — for traditional business
forecasting on modest data, neural usually isn't worth the engineering and
inference cost.

## Revisit conditions

Keep the `forecast_neural` code; reconsider neural if any of these change:

- **Data scale grows.** Global neural models shine with many series / long
  history. The Phase 5 applied domain, if it has thousands of series, could flip
  this — NHITS's 16s/strong-result profile makes it the first to retry.
- **Probabilistic forecasts needed.** NeuralForecast's distributional heads are a
  real advantage if Prophet needs calibrated intervals at scale.
- **Longer horizons.** PatchTST/transformers are designed for long-horizon; the
  48-step M4 horizon doesn't play to that strength.

## Engineering notes

- **MPS keeps the CPU free.** Training ran on the Apple GPU; CPU load stayed ~3
  throughout — far gentler than the CPU-bound StatsForecast runs.
- **Test isolation:** torch and LightGBM ship conflicting OpenMP runtimes and
  segfault in one process, so the neural smoke test runs the train/predict path
  in a clean subprocess (`tests/test_neural.py`).

## Acceptance check

- [x] All three neural models trained and evaluated on M4 hourly via NeuralForecast.
- [x] Per-model MASE, training time, inference time logged to MLflow.
- [x] Comparison table against statistical (Phase 2) and ML (Phase 3).
- [x] Decision documented with reasoning and revisit conditions.
- [x] Lint, typecheck, tests green (51 passed).
