"""Environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
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
    profiles_path: Path = Path("data/processed/bundesliga-2023-2024/player_season_profiles.parquet")
    metric_evidence_path: Path = Path(
        "data/processed/bundesliga-2023-2024/player_metric_evidence.parquet"
    )
    dense_index_path: Path = Path("data/processed/bundesliga-2023-2024/dense_index.json")
    dense_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enable_dense_retrieval: bool = True
    local_files_only: bool = False
    answer_mode: Literal["template", "openai"] = "template"
    openai_model: str = "gpt-5.6-terra"
    openai_max_output_tokens: int = Field(default=800, ge=100, le=4_000)


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings object per process."""
    return Settings()
