"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Prophet runtime settings.

    Values are loaded from .env file or environment variables.
    All paths are resolved relative to the project root if not absolute.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="PROPHET_",
        extra="ignore",
    )

    # Data paths
    data_raw: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_processed: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    # MLflow — note env var is MLFLOW_TRACKING_URI without prefix
    mlflow_tracking_uri: str = Field(
        default=str(PROJECT_ROOT / "mlruns"),
        validation_alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="prophet-benchmarks",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Reproducibility
    random_seed: int = Field(default=42)

    # Serving — which persisted production model the API loads and serves.
    production_model: str = Field(default="market-vol")


settings = Settings()
