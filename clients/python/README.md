# prophet-client (Python)

The Python seam every app forecasts against — the sibling of `@prophet/client`
(TypeScript). Construct once with the service URL, then call. App code never
touches the raw HTTP API.

```bash
uv add prophet-client          # or: pip install prophet-client
```

## Bring your own series (stateless — works with anything)

No pre-trained model, no registration. Send a raw series, get a forecast **and**
a forecastability verdict:

```python
from prophet_client import Prophet

prophet = Prophet("https://prophet-api-production.up.railway.app")

out = prophet.forecast_adhoc(
    series=[(ts, value), ...],   # (ds, y) pairs or {"ds": ..., "y": ...} dicts
    horizon=12,
    freq="MS",                   # pandas offset alias: MS, D, h, W, ...
    level=[80],
)

if out.beats_naive:              # the four-question filter, automated
    print(f"{out.model} beats naive ({out.model_mase:.2f} vs {out.naive_mase:.2f})")
    for p in out.forecasts:
        print(p.ds, p.y_hat, p.lo, p.hi)
else:
    print("Not forecastable — naive wins.", out.beats_naive)  # False, or None if too short
```

## Managed models (pre-trained on the server)

```python
r = prophet.forecast(model="macro", series_id="UNRATE", horizon=6, level=[80])
print(r.model, r.forecasts)

for m in prophet.models():
    print(m.name, m.model, m.n_series)
```

## Notes

- Synchronous, httpx-based, thread-safe. Use as a context manager to close the
  connection: `with Prophet(url) as prophet: ...`.
- Errors raise `ProphetError` (with `.status` for HTTP errors).
- Inject a custom `httpx.Client` via `client=` for auth, retries, or testing.
