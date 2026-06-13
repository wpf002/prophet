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

# App source.
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PROPHET_API_HOST=0.0.0.0

EXPOSE 8000

# The entrypoint ensures a production model exists, then serves.
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
