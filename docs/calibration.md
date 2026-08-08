# Calibration

Most systems attach a confidence number that nothing ever checks. Prophet scores
forecasts against realized outcomes, so here confidence can actually be
**calibrated** — measured, tracked, and flagged when it decays. This module turns
per-run scoring into a queryable property of the system over time.

`src/prophet/calibration/` — persists to the existing MLflow store (a dedicated
`prophet-calibration` experiment), not a parallel database.

## What "calibrated" means here

An 80% interval should contain the outcome ~80% of the time. If your 95%
intervals only catch 85% of outcomes, they're **overconfident** — the "95%" is a
fiction. Calibration measures exactly that gap, per model / series / horizon,
from resolved forecast-vs-actual pairs.

## Structured confidence, not a scalar

Each forecast carries a `Confidence` (`confidence.py`) — enough to tell a trusted
interval from a decorative one:

| field | meaning |
|---|---|
| `point_estimate`, `interval`, `level` | the forecast and its stated interval |
| `basis` | `empirical` (calibrated from outcomes) · `model_derived` (model's own interval, untested) · `assumed` (default, no evidence) |
| `evidence_count` | resolved pairs backing the calibration |
| `independence_score` | 0–1; discounts repeated/overlapping targets so evidence isn't overcounted |
| `last_calibrated` | when it was last checked against reality |

`effective_evidence = evidence_count × independence_score`. Basis becomes
`empirical` once enough *effective* evidence has accrued.

## The metrics (`metrics.py`)

From each resolved record we reconstruct a predictive Gaussian `N(point, σ)`
(σ from the interval half-width) and score it:

- **coverage** — raw fraction of outcomes inside the stated interval. Compare to
  the nominal level: 95% interval, 0.86 coverage → overconfident.
- **reliability curve** — empirical coverage across a grid of nominal levels.
  Perfect calibration sits on the diagonal.
- **ECE** (expected calibration error) — mean |nominal − empirical| over the
  grid. **This is the "0 = perfect" number.** < 0.05 is well calibrated.
- **Brier** — proper score of coverage indicators (lower is better).
- **log score** — Gaussian negative log-likelihood (lower is better; rewards
  intervals that are sharp *and* calibrated). Scale-dependent — compare a series
  to itself over time, not across series.

## Drift detection (`drift.py`) — the deliverable

A model can stay accurate on the point forecast while its intervals quietly go
dishonest. `calibration_drift` splits resolved forecasts into an older
*reference* window and a newer *recent* window, scores ECE on each, and flags
drift when the recent window degrades past a threshold. `regressed_from_good`
isolates the alarming case: **it used to be calibrated and no longer is.**

## Forecastability gate (`gate.py`)

The four admit-or-reject criteria, as code that records *which* failed:

1. is a time series (regular spacing — `check_is_time_series`)
2. beats naive (from the ad-hoc verdict; `None`/too-short fails — absence of
   evidence is not a pass)
3. drives a downstream decision (declared)
4. has lead time to act (declared)

`forecastability_gate(...)` returns `admitted` plus the `failed` criteria and a
reason per criterion.

## Storage + API

Calibration is computed from resolved pairs (`store.resolved_forecasts`, joining
`forecasts`+`actuals`) and logged to MLflow — metrics as run metrics, the
reliability curve as a param, drift flags as tags — one run per
(model, series, horizon).

- `GET /calibration/{model}` — latest calibration per series/horizon (coverage,
  ECE, Brier, log score, basis, evidence, reliability curve). 404 if none.
- `GET /calibration/drift?model=` — groups currently flagged as drifted.

Compute/seed it with `scripts/calibrate_macro.py` (`--backtest` for the local
known-good baseline, or from `PROPHET_MONITOR_DSN`).

## Reading the outputs — the macro baseline

Backtesting the known-good macro model (AutoETS, MASE 0.169 vs naive 0.740):

```
series     n   coverage  ece    brier
CPIAUCSL  120   0.86     0.066  0.213
PCEPI     120   0.79     0.074  0.218
RSAFS     120   0.88     0.112  0.183
TOTALSA   120   0.81     0.140  0.233
UNRATE    120   0.88     0.209  0.243
```

Read it: the 95% intervals actually cover ~79–88% — the macro model is **mildly
overconfident** (a known property of ETS *analytic* intervals, which ignore
parameter uncertainty; the served model's *conformal* intervals are tighter to
nominal). Low ECE (CPI/PCE ~0.07) = well calibrated; higher ECE (UNRATE ~0.21) =
watch it. None drifted across the backtest — a stable model. This is what the
module is for: not "confidence: high", but "your 95% is really 86%, here's the
curve, and it hasn't degraded."

Deliberately **not** here: Dempster-Shafer, imprecise probabilities, belief
functions, epistemic-status vectors. Empirical calibration is possible, so that's
what this does.
