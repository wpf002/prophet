#!/usr/bin/env bash
# Prophet container entrypoint: ensure a production model exists, then serve.
#
# The model artifact and its data are not baked into the image, so on first boot
# (or when the dataset is absent) we ingest market data and train the model.
# Requires APCA_API_KEY_ID / APCA_API_SECRET_KEY for the ingest, and optionally
# PROPHET_CASHFLOW_DSN to read tickers from the live portfolio (else --tickers).
set -euo pipefail

echo "[diag] PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "[diag] /app/src/prophet contents:"; ls /app/src/prophet 2>&1 || true
echo "[diag] /app/src/prophet/models contents:"; ls /app/src/prophet/models 2>&1 || true
python - <<'PY' || true
import prophet
print("[diag] prophet.__path__:", list(getattr(prophet, "__path__", [])))
try:
    import prophet.models.ml  # noqa: F401
    print("[diag] import prophet.models.ml -> OK")
except Exception as e:
    print(f"[diag] import prophet.models.ml -> {type(e).__name__}: {e}")
PY

DATASET="${PROPHET_PRODUCTION_MODEL:-market-vol}"
MODEL_DIR="models/production/${DATASET}"

if [[ ! -f "${MODEL_DIR}/metadata.json" ]]; then
  echo "[entrypoint] No production model at ${MODEL_DIR} — building it."
  if [[ ! -f "data/raw/domains/${DATASET}-train.parquet" ]]; then
    echo "[entrypoint] Ingesting market data..."
    python scripts/ingest_market.py ${PROPHET_TICKERS:+--tickers "$PROPHET_TICKERS"}
  fi
  echo "[entrypoint] Training production model..."
  python scripts/train_production.py --dataset "${DATASET}"
else
  echo "[entrypoint] Found production model at ${MODEL_DIR}."
fi

PORT="${PORT:-8000}"
echo "[entrypoint] Serving on 0.0.0.0:${PORT}"
exec uvicorn prophet.api.main:app --host 0.0.0.0 --port "${PORT}"
