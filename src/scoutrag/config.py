"""Environment-based application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings with a dedicated SCOUTRAG_ namespace."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SCOUTRAG_",
        extra="ignore",
    )

    app_name: str = "ScoutRAG"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_prefix: str = "/api/v1"
    default_result_count: int = Field(default=10, ge=1, le=100)
    max_result_count: int = Field(default=50, ge=1, le=100)
    candidate_pool_size: int = Field(default=40, ge=1, le=500)


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings object per process."""
    return Settings()
