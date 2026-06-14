/** Canonical client error — apps catch this, never a raw fetch/HTTP failure. */
export type ProphetErrorCode =
  | "bad_request" // 400 — e.g. horizon beyond the model's calibration
  | "not_found" // 404 — unknown model or unknown series_id
  | "unavailable" // 503 — no model loaded yet
  | "http_error" // other non-2xx
  | "network_error" // fetch threw / timed out
  | "invalid_response"; // body didn't match the expected schema

export class ProphetError extends Error {
  readonly code: ProphetErrorCode;
  readonly status?: number;

  constructor(
    message: string,
    options: { code: ProphetErrorCode; status?: number; cause?: unknown },
  ) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "ProphetError";
    this.code = options.code;
    this.status = options.status;
  }
}

export function isProphetError(value: unknown): value is ProphetError {
  return value instanceof ProphetError;
}

/** Map an HTTP status to a stable client error code. */
export function codeForStatus(status: number): ProphetErrorCode {
  if (status === 400) return "bad_request";
  if (status === 404) return "not_found";
  if (status === 503) return "unavailable";
  return "http_error";
}
