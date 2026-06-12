.PHONY: help install install-dev test test-cov lint format typecheck clean download-m4 benchmark serve

help:
	@echo "Prophet — common commands"
	@echo ""
	@echo "  make install         Install runtime dependencies (uv)"
	@echo "  make install-dev     Install runtime + dev dependencies"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage report"
	@echo "  make lint            Lint with ruff"
	@echo "  make format          Format with ruff"
	@echo "  make typecheck       Type-check with mypy"
	@echo "  make download-m4     Download M4 competition dataset"
	@echo "  make benchmark       Run baseline benchmark on M4 hourly subset"
	@echo "  make serve           Start FastAPI dev server"
	@echo "  make clean           Remove caches and build artifacts"

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest

test-cov:
	uv run pytest --cov-report=html

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy src

download-m4:
	uv run python scripts/download_m4.py

benchmark:
	uv run python scripts/run_benchmark.py

serve:
	uv run uvicorn prophet.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
