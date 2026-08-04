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

export const AdhocForecastResponseWire = z.object({
  model: z.string(),
  seasonality: z.number(),
  horizon: z.number(),
  n_obs: z.number(),
  beats_naive: z.boolean().nullable().optional(),
  naive_mase: z.number().nullable().optional(),
  model_mase: z.number().nullable().optional(),
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

/** One observation of a caller-supplied series. */
export interface SeriesObservation {
  /** ISO timestamp. */
  ds: string;
  y: number;
}

/** Inputs for a stateless bring-your-own-series forecast. */
export interface AdhocForecastInput {
  /** Regularly-spaced observations (at least 2). */
  series: SeriesObservation[];
  horizon: number;
  /** Pandas offset alias, e.g. "MS", "D", "h". */
  freq: string;
  /** Seasonal period. Inferred from freq when omitted. */
  seasonality?: number;
  /** Confidence levels for prediction intervals, e.g. [80]. */
  level?: number[];
}

/** Ad-hoc forecast result, including the forecastability verdict. */
export interface AdhocForecastResult {
  /** The model chosen for this series (e.g. "AutoETS", "Naive"). */
  model: string;
  seasonality: number;
  horizon: number;
  nObs: number;
  /** Did the chosen model beat naive on the caller's data? `undefined` if too short to validate. */
  beatsNaive?: boolean;
  naiveMase?: number;
  modelMase?: number;
  generatedAt: string;
  forecasts: ForecastPoint[];
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
