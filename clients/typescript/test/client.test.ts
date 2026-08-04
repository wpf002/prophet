import { describe, expect, it } from "vitest";

import { isProphetError, Prophet, ProphetError } from "../src/index.js";

/** Build a fake fetch that returns one canned response. */
function fakeFetch(status: number, body: unknown): typeof fetch {
  return (async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    })) as unknown as typeof fetch;
}

const client = (f: typeof fetch) => new Prophet({ baseUrl: "http://x", fetch: f });

describe("Prophet client", () => {
  it("forecast maps wire snake_case to camelCase", async () => {
    const f = fakeFetch(200, {
      series_id: "NVDA",
      horizon: 2,
      model: "market-vol:LightGBM",
      generated_at: "2026-06-14T00:00:00Z",
      forecasts: [{ ds: "2003-05-03T00:00:00Z", y_hat: 100.5, lo: { "95": 80 }, hi: { "95": 120 } }],
    });
    const res = await client(f).forecast({ seriesId: "NVDA", horizon: 2, level: [95] });
    expect(res.seriesId).toBe("NVDA");
    expect(res.model).toBe("market-vol:LightGBM");
    expect(res.forecasts[0]?.yHat).toBe(100.5);
    expect(res.forecasts[0]?.lo).toEqual({ "95": 80 });
  });

  it("forecastAdhoc maps the verdict + points", async () => {
    const f = fakeFetch(200, {
      model: "AutoETS",
      seasonality: 12,
      horizon: 3,
      n_obs: 72,
      beats_naive: true,
      naive_mase: 1.0,
      model_mase: 0.42,
      generated_at: "2026-06-15T00:00:00Z",
      forecasts: [{ ds: "2026-07-01T00:00:00Z", y_hat: 4.3, lo: { "80": 4.1 }, hi: { "80": 4.5 } }],
    });
    const res = await client(f).forecastAdhoc({
      series: [
        { ds: "2020-01-01T00:00:00Z", y: 1 },
        { ds: "2020-02-01T00:00:00Z", y: 2 },
      ],
      horizon: 3,
      freq: "MS",
      level: [80],
    });
    expect(res.model).toBe("AutoETS");
    expect(res.beatsNaive).toBe(true);
    expect(res.nObs).toBe(72);
    expect(res.forecasts[0]?.lo).toEqual({ "80": 4.1 });
  });

  it("forecastAdhoc leaves beatsNaive undefined when null", async () => {
    const f = fakeFetch(200, {
      model: "Naive",
      seasonality: 1,
      horizon: 2,
      n_obs: 10,
      beats_naive: null,
      generated_at: "2026-06-15T00:00:00Z",
      forecasts: [{ ds: "2026-07-01T00:00:00Z", y_hat: 5 }],
    });
    const res = await client(f).forecastAdhoc({
      series: [
        { ds: "2020-01-01T00:00:00Z", y: 1 },
        { ds: "2020-02-01T00:00:00Z", y: 2 },
      ],
      horizon: 2,
      freq: "MS",
    });
    expect(res.beatsNaive).toBeUndefined();
    expect(res.model).toBe("Naive");
  });

  it("models maps n_series -> nSeries", async () => {
    const f = fakeFetch(200, {
      default: "market-vol",
      models: [{ name: "market-vol", model: "LightGBM", horizon: 21, n_series: 28 }],
    });
    const res = await client(f).models();
    expect(res.default).toBe("market-vol");
    expect(res.models[0]?.nSeries).toBe(28);
  });

  it("404 surfaces as ProphetError code not_found with the detail", async () => {
    const f = fakeFetch(404, { detail: "Unknown model 'x'. See GET /models." });
    await expect(client(f).forecast({ seriesId: "s", horizon: 1, model: "x" })).rejects.toSatisfy(
      (e: unknown) => isProphetError(e) && e.code === "not_found" && e.status === 404,
    );
  });

  it("503 surfaces as unavailable", async () => {
    const f = fakeFetch(503, { detail: "no model" });
    await expect(client(f).forecast({ seriesId: "s", horizon: 1 })).rejects.toSatisfy(
      (e: unknown) => isProphetError(e) && e.code === "unavailable",
    );
  });

  it("malformed body is invalid_response", async () => {
    const f = fakeFetch(200, { nope: true });
    await expect(client(f).models()).rejects.toSatisfy(
      (e: unknown) => e instanceof ProphetError && e.code === "invalid_response",
    );
  });

  it("requires a baseUrl", () => {
    expect(() => new Prophet({ baseUrl: "" })).toThrow(ProphetError);
  });
});
