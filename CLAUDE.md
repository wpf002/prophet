# Prophet — Claude Code Context

This file is loaded by Claude Code on every session to maintain context across local, cloud, and mobile sessions.

## Project identity

Prophet is a time-series forecasting system. Its goal is to produce forecasts that beat strong baselines on standard benchmarks (M4, M5), evaluated with proper cross-validation, then deploy the winning model as a monitored API.

It is **not** a wrapper around Meta's `prophet` library. The internal stack is Nixtla's forecasting suite.

## House rules

- **Python 3.11+ only.** No 3.10 compat.
- **uv for package management.** Never use pip directly in this project.
- **Polars over pandas.** Only fall back to pandas when a library forces it.
- **No random splits, ever.** Time-series CV with expanding or rolling windows is the only valid evaluation.
- **MASE is the primary metric.** sMAPE, WAPE, pinball loss are secondary.
- **Every benchmark run is an MLflow run.** No exceptions.
- **Lint and type-check must pass before commit.** `make lint` and `make typecheck` are non-negotiable.

## Architecture invariants (locked)

- `src/prophet/models/` modules return Nixtla-compatible forecast DataFrames (Polars), never raw numpy arrays.
- All datetime handling uses Polars `Datetime("us", "UTC")`. No naive timestamps anywhere.
- Money or revenue domain data uses integer minor units (cents), never floats — consistent with Crossbar/Furlong conventions.
- Cross-validation defaults: 5 windows, horizon matches frequency (24 for hourly, 7 for daily, 12 for monthly).
- Random seeds: always set `RANDOM_SEED = 42` for reproducibility. Pass to every model that accepts a seed.

## Phase status

See `ROADMAP.md`. Update phase status there when transitioning between phases. Phase transitions require:
1. All acceptance criteria documented as passing
2. MLflow run IDs cited for benchmark results
3. Updated status table in README.md and ROADMAP.md

## What to do at the start of a session

1. Pull latest: `git pull --rebase`
2. Sync deps: `make install-dev`
3. Run tests: `make test`
4. Check current phase in `ROADMAP.md`
5. Check open TODOs: `grep -rn "TODO\|FIXME" src/ scripts/`

## What to do at the end of a session

1. Run `make lint format typecheck test`
2. Commit with conventional commit message
3. Push to main as session checkpoint
