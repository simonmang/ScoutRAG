"""Transport-specific response models."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Stable health response for probes and smoke tests."""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
