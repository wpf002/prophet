# Prophet

Production-grade time-series forecasting system benchmarked against M4/M5 competition data, deployable as a monitored API service.

> **Note on the name:** This project is named Prophet but does **not** wrap Meta's `prophet` library. The internal modeling stack is built on Nixtla's forecasting suite (StatsForecast, MLForecast, NeuralForecast).

---

## Core thesis

A forecasting system is only as good as the evaluation around it. Prophet is built on three commitments:

1. **Benchmark against known winners.** Every model is evaluated against published M4 and M5 competition scores. If we can't get within 10% of competition winners on a comparable subset, we don't have a model — we have a notebook.
2. **Rigorous time-series cross-validation.** No random splits. Expanding-window or rolling-origin CV on every model. Multiple horizons. Multiple metrics (MASE primary).
3. **Deploy or it doesn't exist.** The final artifact is a monitored API service with drift detection and forecast-vs-actual logging, not a Jupyter notebook.

---

## Stack

| Layer | Tool |
| --- | --- |
| Forecasting | StatsForecast, MLForecast, NeuralForecast, HierarchicalForecast |
| Data wrangling | Polars, DuckDB, PyArrow |
| ML utilities | LightGBM, scikit-learn, Optuna |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Package management | uv |
| Lint / format | Ruff |
| Types | mypy (strict) |
| Tests | pytest |

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone https://github.com/wpf002/prophet.git
cd prophet

# Install deps (including dev)
make install-dev

# Copy env template
cp .env.example .env

# Verify install — runs naive baseline on synthetic data
uv run pytest

# Download M4 competition data
make download-m4

# Run baseline benchmark
make benchmark

# Start API server
make serve
```

---

## Project layout

```
prophet/
├── src/prophet/
│   ├── config.py              # Pydantic settings
│   ├── cli.py                 # Typer CLI
│   ├── data/
│   │   ├── loaders.py         # M4/M5 dataset loaders
│   │   └── preprocessors.py   # Cleaning, resampling
│   ├── models/
│   │   ├── baselines.py       # Naive, SeasonalNaive, Drift
│   │   ├── statistical.py     # AutoARIMA, AutoETS, Theta
│   │   ├── ml.py              # LightGBM via MLForecast
│   │   └── neural.py          # NHITS, TFT, PatchTST
│   ├── evaluation/
│   │   ├── metrics.py         # MASE, sMAPE, WAPE, pinball
│   │   ├── cross_validation.py
│   │   └── benchmarks.py      # M4/M5 winner scores (constants)
│   ├── experiments/
│   │   └── tracking.py        # MLflow helpers
│   └── api/
│       ├── main.py            # FastAPI app
│       └── routes.py          # /health, /forecast endpoints
├── tests/                     # pytest suite
├── notebooks/                 # exploratory work (not for production)
├── scripts/
│   ├── download_m4.py
│   ├── download_m5.py
│   └── run_benchmark.py
├── data/
│   ├── raw/                   # gitignored — raw competition data
│   └── processed/             # gitignored — preprocessed parquet
└── mlruns/                    # gitignored — MLflow tracking
```

---

## Evaluation methodology

Every forecast is evaluated using:

- **MASE** (Mean Absolute Scaled Error) — scale-free, primary metric. Comparable across series.
- **sMAPE** — for M4 alignment.
- **WAPE** — for business-readable accuracy.
- **Pinball loss** — for probabilistic forecasts.

Cross-validation uses Nixtla's `cross_validation` with expanding windows. No random splits. Ever.

Benchmarks compare against the published winning scores for the M4 and M5 competitions stored in `src/prophet/evaluation/benchmarks.py`.

---

## Roadmap

See `ROADMAP.md` for full phase breakdown with acceptance and kill criteria.

| Phase | Status | Summary |
| --- | --- | --- |
| 0 | Done | Foundation: scaffold, deps, CI, tests skeleton |
| 1 | Done | Naive baselines on M4 hourly — SeasonalNaive MASE 1.19 ([results](docs/phase-1-results.md)) |
| 2 | Done | Statistical — AutoARIMA MASE 0.948 beats the SeasonalNaive floor ([results](docs/phase-2-results.md)) |
| 3 | Open | ML: LightGBM via MLForecast |
| 4 | Open | Neural: NHITS, TFT, PatchTST |
| 5 | Open | Applied domain (casino ops / sports betting) |
| 6 | Open | API + monitoring (drift, forecast vs actual) |

---

## Contributing

Single-developer project. No external contribution workflow.

## License

MIT
