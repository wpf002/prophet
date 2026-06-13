# Phase 5 — Applied Domain (Plan & Data Access)

**Status:** Superseded — see [phase-5-market-results.md](docs/phase-5-market-results.md).
Syntrackr turned out to be an investment tracker with no time series, so the
applied domain pivoted to **portfolio market data** (forecastable volume vs.
random-walk prices). This doc is retained as the original plan + the reusable
data-access playbook for the casino/betting domains if pursued later.

**Decision (original):** Run all three candidate domains (casino, sports betting,
cash flow) through the same model ladder, executed **sequentially** via a shared
source-agnostic pipeline — not three bespoke builds.

## Why one pipeline, three domains

The Phase 1-4 ladder (baselines → statistical → LightGBM) only ever consumes the
Nixtla long format `(unique_id, ds, y)`. So each domain needs just two pieces:

1. a **connector** that materializes its source into
   `data/raw/domains/<name>-{train,test}.parquet`, and
2. a **DomainSpec** (freq, horizon, seasonality) in `src/prophet/data/domains.py`.

Everything else is reused. Run any domain with:

```bash
uv run python scripts/run_benchmark.py --dataset domain-<name> --models ml
```

Scaffolded and tested this phase (no live data needed yet):
`prophet/data/domains.py` (`load_domain`, `DOMAIN_SPECS`), the `domain-<name>`
path in `run_benchmark.py`, and `tests/test_domains.py`.

## The blocker: I need real data

This environment can't reach a live Postgres/API (no network egress, no
credentials, and the real schemas are unknown). So the connectors can't be
written accurately yet. To unblock, for whichever domain we start with, I need
**one** of:

- a **connection string + read-only creds** in `.env` (if reachable from here), or
- a **schema + a small sample export** (CSV/Parquet) I can read, or
- the **output of a query** you run, dropped under `data/raw/domains/`.

## Per-domain status (provisional specs — confirm against real data)

| Domain | Source | Spec (freq / horizon / seasonality) | What I need |
| --- | --- | --- | --- |
| `cashflow` (Syntrackr) | Postgres you control | D / 30 / 7 | Connection string or table schema + sample. **Recommended first** — you own it, no licensing. |
| `casino` | BI export / KQL pull | h / 24 / 24 | An export or KQL result; confirm confidentiality before it leaves your env. |
| `betting` | Pinnacle / odds API (GateSmart) | h / 24 / 24 | API access or a GateSmart export; confirm licensing. |

## Recommended order

1. **`cashflow`** first — most accessible (you control the Postgres, no
   licensing). It's a small-data regime (few series, months of history), so it
   doubles as a test of whether the methodology survives leaving clean
   competition data. Expect SeasonalNaive to be a tough floor again.
2. **`casino`** — closest to the hourly shape we tuned for; should transfer best.
3. **`betting`** — most novel; intraday volatility is the interesting case.

## Per-domain acceptance (each, when its data lands)

- Domain data ingested to `data/raw/domains/<name>-{train,test}.parquet`.
- Full ladder runs via `--dataset domain-<name>`.
- Documented MASE/WAPE on the holdout, vs. the SeasonalNaive floor.
- At least one domain-specific feature shown to help (e.g. game type, day-of-pay-
  period, event volatility).
- Honest writeup including failure modes in `docs/phase-5-<name>-results.md`.

## Next action

Pick the first domain (recommend `cashflow`) and hand me one of the access
options above. I'll write `scripts/ingest_<name>.py`, materialize the parquet,
and run the ladder.
