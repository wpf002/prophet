# Using Prophet from any app

Prophet is a standalone forecasting service. Any app — in any language — can send
a time series and get back a forecast **plus a forecastability verdict** (does the
best model beat a naive baseline on your own data?). No pre-trained model, no
per-source server code.

## Run it

It's live at `https://prophet-api-production.up.railway.app`, or run your own:

```bash
docker build -t prophet .
docker run -p 8000:8000 prophet
```

The container serves `POST /forecast/adhoc` with **zero configuration** — no
credentials, no pre-trained models. (Managed models like `macro` and
`market-vol` are a bonus that build on boot when their credentials are present.)

## The universal call: bring your own series

```bash
curl -s localhost:8000/forecast/adhoc -H 'content-type: application/json' -d '{
  "series": [{"ds":"2019-01-01","y":100}, {"ds":"2019-02-01","y":103}, ...],
  "horizon": 12, "freq": "MS", "level": [80]
}'
```

```json
{
  "model": "AutoETS",
  "beats_naive": true,          // ← the verdict. false = not forecastable; null = too short to tell
  "naive_mase": 1.00, "model_mase": 0.42,
  "forecasts": [{"ds":"...","y_hat":4.3,"lo":{"80":4.1},"hi":{"80":4.5}}, ...]
}
```

Prophet runs a fast model ladder (baselines + AutoETS/Theta) with expanding-window
cross-validation on *your* data, serves the winner with conformal intervals, and
tells you whether forecasting even helps. It degrades gracefully — short, tiny,
and flat series all get a sensible answer.

## Language clients (same shape everywhere)

**TypeScript** (`@prophet/client`):
```ts
import { Prophet } from "@prophet/client";
const prophet = new Prophet({ baseUrl: PROPHET_URL });
const out = await prophet.forecastAdhoc({ series, horizon: 12, freq: "MS", level: [80] });
if (out.beatsNaive) console.log(out.model, out.forecasts);
```

**Python** (`prophet-client`):
```python
from prophet_client import Prophet
out = Prophet(PROPHET_URL).forecast_adhoc(series=series, horizon=12, freq="MS", level=[80])
if out.beats_naive:
    print(out.model, out.forecasts)
```

**AI chains**: the `prophet_forecast` / `prophet_models` MCP tools in Trident.

## Managed models (pre-trained, curated)

For series Prophet owns end-to-end (macro indicators, portfolio volume), it hosts
warm models: `GET /models` to discover, `POST /forecast` with `model` + `series_id`.
Bloomberg consumes the `macro` model this way.

## The two modes

| | Managed | Ad-hoc |
|---|---|---|
| Data | curated on the server | brought by the caller |
| Training | pre-trained, warm | selected per request |
| New source | needs server code | **zero** — just call |
| Use when | Prophet curates the series | any app has a series to forecast |

Ad-hoc is what makes Prophet plug-and-play with anything.
