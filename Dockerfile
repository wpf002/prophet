# Prophet API — production image.
# OpenMP (libgomp) is required by LightGBM.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer) from the lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# App source. The project itself is NOT pip-installed — an editable install
# registers an import finder that shadowed PYTHONPATH and hid prophet.models.
# Instead `prophet` is imported straight from the source tree via PYTHONPATH.
COPY src ./src
COPY scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PROPHET_API_HOST=0.0.0.0

EXPOSE 8000

# Default command serves the API (ensures a model exists first). Using CMD (not
# ENTRYPOINT) lets a cron service override it with e.g. the record_actuals job.
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
