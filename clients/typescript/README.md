# @prophet/client

Typed client for the Prophet forecasting service — the **numbers** sibling of
[`@flint/core`](https://github.com/wpf002/flint). App code never calls the raw
forecasting API; it calls this, the same way it calls Flint for the AI layer.

## Install

Published to the local Verdaccio registry (shared with `@flint/*`). Point the
`@prophet` scope at it in your app's `.npmrc`:

```
@prophet:registry=http://localhost:4873/
```

```bash
pnpm add @prophet/client@~0.1.0
```

## Use

```ts
import { Prophet } from "@prophet/client";

const prophet = new Prophet({
  baseUrl: "https://prophet-api-production.up.railway.app",
});

// What can it forecast?
const { default: defaultModel, models } = await prophet.models();

// Forecast a series, with a 95% interval.
const result = await prophet.forecast({
  seriesId: "NVDA",
  horizon: 5,
  level: [95],
  // model: "market-vol",  // optional; omit for the default
});

for (const p of result.forecasts) {
  console.log(p.ds, p.yHat, p.lo?.["95"], p.hi?.["95"]);
}
```

## Errors

Failures throw a `ProphetError` with a stable `code`
(`bad_request` | `not_found` | `unavailable` | `http_error` | `network_error` |
`invalid_response`) and the HTTP `status`. Use `isProphetError(e)` to narrow.

## Develop

```bash
pnpm install
pnpm build       # tsup -> dist (ESM + CJS + d.ts)
pnpm typecheck
pnpm test        # vitest
```
