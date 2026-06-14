# Prophet in the Ecosystem — Integration Plan

**Status:** In progress — Steps 1–2 done. Confidence is deliberately
"prove it before believing it" — Step 4 (wire a real app) is the gate.

- ✅ **Step 1** — multi-model API (`/forecast` routes by model, `GET /models`); live.
- ✅ **Step 2** — `@prophet/client` (TS, mirrors `@flint/core`) built, tested,
  published to Verdaccio; verified end-to-end (fresh app installed it and
  forecasted against the live service). Code in `clients/typescript/`.
- ✅ **Step 3** — Prophet MCP tool in Trident (`prophet_forecast`, `prophet_models`);
  typecheck clean, verified against the live service. (Committed in the trident repo.)
- ⬜ **Step 4** — wire one real app (gate_smart / vantage). **The gate** — needs
  picking the app + its data + the decision the forecast improves.
- ⬜ **Step 5** — UI / polish.

## The question this answers

What is Prophet *for*, and how does it fit the rest of the apps? Prophet today is
a complete-but-purposeless forecasting service (it forecasts trading volume for
28 tickers — accurate, not useful). The portfolio has a cluster of apps that need
a "where is this number heading" layer: vantage, bellwether, project_hype,
bloomberg, strata, crossbar, gate_smart, furlong, ice_sight, syntrackr.

## Key finding (from reading flint + trident)

- **Flint is a package, not a service.** Apps `import { Flint } from '@flint/core'`
  (published to a local Verdaccio registry) and call it; the AI provider hides
  behind it. "App code never calls a vendor SDK directly."
- **Trident exposes capabilities as MCP tools** (`packages/mcp-server/src/tools/`,
  e.g. `perplexity.ts`) so the AIs can call them mid-chain.

## Decision: do NOT merge Prophet and Flint

Different language (Flint = TS lib, Prophet = Python service), different nature
(stateless AI proxy vs. stateful trained-model service with heavy ML deps,
training jobs, monitoring). Merging would chain a heavy batch system to the
lightweight AI layer every app depends on.

Instead: make Prophet a **sibling on the same shelf** —
`@flint/core` (language) + `@prophet/client` (numbers). Unify at the *consumption*
layer, not the engine.

## The plan

1. **Multi-model Prophet API** — `/forecast` routes by model; add `/train` +
   `/datasets` so any app's data becomes a served model (today it's hardcoded to
   `market-vol`). Prerequisite for everything else.
2. **`@prophet/client`** — a TS package mirroring `@flint/core`'s `Flint` class:
   `new Prophet().forecast({ series, horizon })`. Publish to the Verdaccio
   registry. The seam every app integrates against.
3. **Prophet MCP tool in Trident** — `tools/prophet.ts` (like `perplexity.ts`) so
   the AIs can forecast mid-chain.
4. **Prove it on ONE app** — wire gate_smart or vantage to `@prophet/client` for a
   real forecast, end to end. The test that it's a product, not a demo. **If this
   doesn't produce something noticeably useful, stop and reconsider.**
5. **UI / polish** — only after Step 4 works.

## Reality checks baked in

- Forecasting only adds value where the target is forecastable AND drives a
  decision with lead time (the four-question filter). Stock prices failed this
  (random walk); volume passed but isn't actionable. Run each candidate through
  the test — don't assume.
- With few series, classical per-series models beat the global LightGBM (the
  Phase 5 finding). Model choice depends on the host app's data shape.
