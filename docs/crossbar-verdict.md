# Crossbar — forecastability verdict (real data)

**Date:** 2026-06-14 · **Source:** Crossbar public read-only API
(`https://crossbar.fly.dev`, no credentials) · **Connector:**
`scripts/ingest_crossbar.py --api`

This is the Step-4 gate from `ecosystem-plan.md`: wire one real app to Prophet
and get an honest answer to "does forecasting add value here?" — run through the
four-question filter on **real** Crossbar trade data, not synthetic.

## What was measured

- 88 busiest open markets, YES-contract trades bucketed server-side into a
  regular hourly grid (`/markets/:id/candles`), ~11.2k train points / 528 test.
- Horizon 6h, 5-window expanding CV, MASE primary (seasonality=1). Two targets:
  YES **price** (implied probability) and per-bucket **volume**.

## Results

**Target = YES price** (MASE, lower better):

| Model                 |   MASE | vs Naive |
|-----------------------|-------:|---------:|
| Naive / SeasonalNaive | 1.1250 | floor    |
| DynamicOptimizedTheta | 1.1173 | −0.7%    |
| AutoARIMA             | 1.1189 | −0.5%    |
| AutoETS / AutoTheta   | ~1.13  | ~tie     |

**Target = per-bucket volume** (MASE):

| Model                 |   MASE | vs Naive |
|-----------------------|-------:|---------:|
| Naive / SeasonalNaive | 0.4907 | floor    |
| AutoTheta             | 0.6006 | +22% (worse) |
| DynamicOptimizedTheta | 0.6079 | +24% (worse) |

(Global LightGBM crashed on this panel — series are too short for the lag-feature
window. Consistent with the Phase-5 finding that few/short series favour
per-series classical models.)

## Verdict (four-question filter)

1. **Is it a time series?** Yes — real hourly YES-price and volume series.
2. **Is it forecastable (beats naive)?** **No, not meaningfully.**
   - Price is a near-martingale: the best model beats last-value carry-forward by
     **<1% MASE** — within noise. This is the efficient-market signature, the
     same result stock close prices gave in Phase 5. (sMAPE ~2.8% looks small
     only because prices barely move hour-to-hour, not because the model adds
     skill.)
   - Volume is bursty/intermittent (sMAPE ~195%); **naive wins outright** — no
     statistical model beats carrying the last bucket forward.
3. **Downstream decision?** In principle (strategy management), but a sub-1% edge
   over "assume it stays put" is not decision-grade.
4. **Lead time?** Yes (6h).

So on the **current open-markets snapshot** — mostly short-lived sports markets
(sub-week lifespan) within the public API's 7-day candle window — neither
per-market price nor per-market hourly volume is a strong forecasting target.

## This is the filter working, not a dead end

The integration itself is **built and proven on real data**: a read-only
connector (public API / DSN / synthetic), a `crossbar` DomainSpec, and the full
model ladder running unchanged via `--dataset domain-crossbar`. Nothing in
Crossbar was touched (read-only throughout).

The target that's plausibly forecastable *and* decision-relevant is **not**
per-market hourly series — it's **platform-level aggregate activity** (total
daily volume / active-market count), which Phase 5 showed is the forecastable
cousin of price, and which drives a real ops decision (liquidity / capacity
provisioning ahead of load). Testing that needs a longer history than the public
API's 7-day candle window exposes — i.e. a **read-only `CROSSBAR_DSN`** to reach
the full `Trade` history. That is the concrete next step.

## Update — platform aggregate tested against the real DB (read-only)

Reached the full `Trade` history via `fly proxy` to `crossbar-db` (read-only):
**197,814 trades, 1,249 markets, but only ~9 days** (2026-06-05 → 06-14). Built
the platform hourly aggregate (total volume + trade count, gap-filled to a
regular grid) as `crossbar-agg` and ran the ladder:

| Model           |   MASE |
|-----------------|-------:|
| HistoricAverage | 1.0254 |
| SeasonalNaive   | 1.4511 |
| Naive           | 1.8284 |
| best statistical (DynOptTheta) | 1.9253 |

**"Predict the mean" wins** — no model beats HistoricAverage. With ~7 days of
training, daily seasonality can't be learned and the series is burst-dominated.

## Bottom line

The integration is real, read-only, and correct (per-market via public API,
aggregate via read-only DSN). The blocker is **data age**: Crossbar is ~9 days
old, too young for forecasting to beat trivial baselines at any target tested.
Value here is time-gated — re-run as history accumulates (weeks→months); the
wiring is already done.

## How to reproduce

```bash
uv run python scripts/ingest_crossbar.py --api --top-n 120 --min-trades 48            # price
uv run python scripts/ingest_crossbar.py --api --top-n 120 --min-trades 48 --target volume
uv run python scripts/run_benchmark.py --dataset domain-crossbar --models baselines
uv run python scripts/run_benchmark.py --dataset domain-crossbar --models statistical
```
