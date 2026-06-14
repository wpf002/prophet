import { z } from "zod";

// ---------------------------------------------------------------------------
// Wire schemas — the snake_case shapes the Prophet HTTP API actually returns.
// Kept internal; the client maps them to the camelCase public types below.
// ---------------------------------------------------------------------------

export const ForecastPointWire = z.object({
  ds: z.string(),
  y_hat: z.number(),
  lo: z.record(z.number()).nullable().optional(),
  hi: z.record(z.number()).nullable().optional(),
});

export const ForecastResponseWire = z.object({
  series_id: z.string(),
  horizon: z.number(),
  model: z.string(),
  generated_at: z.string(),
  forecasts: z.array(ForecastPointWire),
});

export const ModelSummaryWire = z.object({
  name: z.string(),
  model: z.string().nullable().optional(),
  freq: z.string().nullable().optional(),
  horizon: z.number().nullable().optional(),
  seasonality: z.number().nullable().optional(),
  n_series: z.number().nullable().optional(),
  trained_at: z.string().nullable().optional(),
});

export const ModelsResponseWire = z.object({
  default: z.string(),
  models: z.array(ModelSummaryWire),
});

export const HealthResponseWire = z.object({
  status: z.string(),
  version: z.string(),
  timestamp: z.string(),
});

// ---------------------------------------------------------------------------
// Public types — camelCase, what app code consumes.
// ---------------------------------------------------------------------------

/** One forecast step. `lo`/`hi` are keyed by confidence level, e.g. {"95": 1234}. */
export interface ForecastPoint {
  ds: string;
  yHat: number;
  lo?: Record<string, number>;
  hi?: Record<string, number>;
}

export interface ForecastResult {
  seriesId: string;
  horizon: number;
  model: string;
  generatedAt: string;
  forecasts: ForecastPoint[];
}

export interface ModelSummary {
  name: string;
  model?: string;
  freq?: string;
  horizon?: number;
  seasonality?: number;
  nSeries?: number;
  trainedAt?: string;
}

export interface ModelsResult {
  default: string;
  models: ModelSummary[];
}

export interface HealthResult {
  status: string;
  version: string;
  timestamp: string;
}

/** Inputs for a forecast request. */
export interface ForecastInput {
  seriesId: string;
  horizon: number;
  /** Confidence levels for prediction intervals, e.g. [80, 95]. */
  level?: number[];
  /** Which served model to use. Omit for the service's default. */
  model?: string;
}

export interface ProphetConfig {
  /** Base URL of the Prophet service, e.g. "https://prophet-api-production.up.railway.app". */
  baseUrl: string;
  /** Optional bearer token, sent as `Authorization: Bearer <apiKey>`. */
  apiKey?: string;
  /** Per-request timeout in ms (default 30000). */
  timeoutMs?: number;
  /** Injectable fetch (for tests / non-global-fetch runtimes). Defaults to global fetch. */
  fetch?: typeof fetch;
}
