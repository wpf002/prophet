import { codeForStatus, ProphetError } from "./errors.js";
import {
  ForecastResponseWire,
  HealthResponseWire,
  ModelsResponseWire,
  type ForecastInput,
  type ForecastResult,
  type HealthResult,
  type ModelsResult,
  type ProphetConfig,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * The Prophet client. Construct once per app with the service URL, then call
 * `forecast` / `models` / `health`. App code never touches the raw HTTP API —
 * it calls this, exactly like `@flint/core` for the AI layer.
 */
export class Prophet {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(config: ProphetConfig) {
    if (!config.baseUrl) {
      throw new ProphetError("Prophet baseUrl is required.", { code: "bad_request" });
    }
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.fetchImpl = config.fetch ?? globalThis.fetch;
    if (!this.fetchImpl) {
      throw new ProphetError("No fetch available; pass config.fetch.", { code: "network_error" });
    }
  }

  /** Forecast a single series, with optional prediction intervals. */
  async forecast(input: ForecastInput): Promise<ForecastResult> {
    const body = {
      series_id: input.seriesId,
      horizon: input.horizon,
      level: input.level,
      model: input.model,
    };
    const raw = await this.request("POST", "/forecast", body);
    const parsed = ForecastResponseWire.safeParse(raw);
    if (!parsed.success) {
      throw new ProphetError("Unexpected /forecast response shape.", {
        code: "invalid_response",
        cause: parsed.error,
      });
    }
    const r = parsed.data;
    return {
      seriesId: r.series_id,
      horizon: r.horizon,
      model: r.model,
      generatedAt: r.generated_at,
      forecasts: r.forecasts.map((p) => ({
        ds: p.ds,
        yHat: p.y_hat,
        lo: p.lo ?? undefined,
        hi: p.hi ?? undefined,
      })),
    };
  }

  /** List the models this service can forecast with. */
  async models(): Promise<ModelsResult> {
    const raw = await this.request("GET", "/models");
    const parsed = ModelsResponseWire.safeParse(raw);
    if (!parsed.success) {
      throw new ProphetError("Unexpected /models response shape.", {
        code: "invalid_response",
        cause: parsed.error,
      });
    }
    return {
      default: parsed.data.default,
      models: parsed.data.models.map((m) => ({
        name: m.name,
        model: m.model ?? undefined,
        freq: m.freq ?? undefined,
        horizon: m.horizon ?? undefined,
        seasonality: m.seasonality ?? undefined,
        nSeries: m.n_series ?? undefined,
        trainedAt: m.trained_at ?? undefined,
      })),
    };
  }

  /** Liveness check. */
  async health(): Promise<HealthResult> {
    const raw = await this.request("GET", "/health");
    const parsed = HealthResponseWire.safeParse(raw);
    if (!parsed.success) {
      throw new ProphetError("Unexpected /health response shape.", {
        code: "invalid_response",
        cause: parsed.error,
      });
    }
    return parsed.data;
  }

  private async request(method: string, path: string, body?: unknown): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers: Record<string, string> = { accept: "application/json" };
    if (body !== undefined) headers["content-type"] = "application/json";
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (cause) {
      throw new ProphetError(`Request to ${path} failed.`, { code: "network_error", cause });
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      const detail = await this.safeDetail(response);
      throw new ProphetError(detail ?? `${path} returned ${response.status}.`, {
        code: codeForStatus(response.status),
        status: response.status,
      });
    }

    try {
      return await response.json();
    } catch (cause) {
      throw new ProphetError(`Could not parse JSON from ${path}.`, {
        code: "invalid_response",
        cause,
      });
    }
  }

  private async safeDetail(response: Response): Promise<string | undefined> {
    try {
      const data = (await response.json()) as { detail?: unknown };
      return typeof data.detail === "string" ? data.detail : undefined;
    } catch {
      return undefined;
    }
  }
}
