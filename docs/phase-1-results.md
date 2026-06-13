# Phase 1 — Naive Baselines on M4 (Results)

**Status:** Complete.
**Dataset:** M4 Hourly (414 series, horizon 48, seasonality 24).
**MLflow run:** `698e24404f094757933c843cea7fb542` (experiment `prophet-benchmarks`, tags `phase=1`, `dataset=m4-hourly`).

## What was built

- `scripts/download_m4.py` — pulls M4 via `datasetsforecast`, synthesises UTC
  timestamps (M4 series carry only an integer step index), holds out the last
  `horizon` points per series, and writes `data/raw/m4/{freq}-{train,test}.parquet`
  in Nixtla long format (`unique_id`, `ds: Datetime[us, UTC]`, `y: f64`).
- `scripts/run_benchmark.py` — wired to evaluate against the **official M4 test
  split** (it previously re-split the training data) and log all five metrics per
  model to MLflow.
- `forecast_baselines` now returns `ds` as `Datetime("us", "UTC")` per the project
  invariant; `load_synthetic` likewise emits UTC timestamps.

## Headline results (M4 Hourly, official test split)

| Model | MASE | sMAPE | WAPE |
| --- | --- | --- | --- |
| Naive | 11.6077 | 43.00 | 35.77 |
| **SeasonalNaive** | **1.1932** | **13.91** | 13.52 |
| HistoricAverage | 11.6862 | 34.16 | 31.98 |
| RandomWalkWithDrift | 11.4550 | 43.47 | 35.76 |

SeasonalNaive is the floor for later phases: **MASE 1.193**. It exploits the clean
24-hour cycle and dominates the non-seasonal baselines (~11.5), exactly as
expected for hourly data.

## Evaluation framework is correct (verified three ways)

Our `metrics.mase` was cross-checked against the official Nixtla
`utilsforecast.losses.mase` and reproduces it to 16 significant figures
(SeasonalNaive: 0.476613… on the contaminated data, 1.1932 on the clean split).
Scoring the **official M4 Naive2 submission** with our metric and clean train
scaling yields **MASE 2.3950, sMAPE 18.3829** — matching the published Naive2
sMAPE (18.383) and confirming our scale matches the M4 organisers'.

## Two bugs found and fixed

1. **Test-set leak in the M4 loader (fixed).** `datasetsforecast.M4.load()` returns
   each series as the *full* sequence (train **concatenated with** the held-out
   test, `ds` made continuous), not the training portion alone. The first
   `download_m4.py` used that output directly as training data, baking the test
   period into `hourly-train.parquet`. The symptom was an impossibly good
   SeasonalNaive (MASE **0.477**, which would beat the M4 winner). Fixed by holding
   out the last `horizon` points per series via `split_train_test` — by
   construction this is exactly the official M4 test split, with zero train/test
   overlap. A regression guard lives in `tests/test_baselines.py`
   (`test_no_train_test_leak`).

2. **Mislabeled published constant (fixed).** `benchmarks.py` hard-coded Hourly
   `Naive2` MASE as **11.608**. That figure is actually the plain **Naive(1)**
   score — our Naive reproduces **11.6077**. The real Naive2 MASE is **2.395**
   (verified above). Corrected the constant; the Naive2 sMAPE (18.383) was already
   right.

## Open items / caveats

- **Roadmap acceptance criteria need updating.** Phase 1 in `ROADMAP.md` says
  "confirm SeasonalNaive MASE ≈ 11.608" and "within 5% of published Naive2." Both
  rest on the mislabeled constant. The correct, verified targets are: Naive ≈
  11.608, SeasonalNaive ≈ 1.193, published Naive2 ≈ 2.395.
- **Unverified Hourly constants.** `benchmarks.py` Hourly `SES` (11.426) and
  `Theta` (11.504) MASE are suspect — same ~11.5 magnitude as the mislabeled
  Naive2, where the 2–3 range is expected. Left unchanged and flagged in-code;
  verify against their official submissions before relying on them.

## Acceptance check

- [x] All four baselines run end-to-end via `uv run python scripts/run_benchmark.py --dataset m4-hourly`.
- [x] MLflow run logged with all five metrics per model, tagged `phase=1`.
- [x] Full hourly benchmark completes in **~11 s** on a laptop (budget: <10 min).
- [x] M4 hourly Parquet committed to gitignored `data/raw/m4/`.
- [x] Evaluation framework validated: reproduces official Naive2 (MASE 2.395,
      sMAPE 18.383) and official SeasonalNaive (MASE 1.193) exactly.
- [n/a] "SeasonalNaive ≈ 11.608" — target was based on a mislabeled constant; see
      Open items.
