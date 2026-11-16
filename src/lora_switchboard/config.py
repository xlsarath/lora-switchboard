"""Runtime configuration, read from ``LSB_*`` environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the switchboard service.

    Every field maps to an environment variable with the ``LSB_`` prefix,
    e.g. ``vllm_url`` -> ``LSB_VLLM_URL``.
    """

    model_config = SettingsConfigDict(env_prefix="LSB_", extra="ignore")

    vllm_url: str = Field(
        default="http://127.0.0.1:8001",
        description="Base URL of the vLLM OpenAI-compatible server (no trailing /v1).",
    )
    vllm_api_key: SecretStr | None = Field(
        default=None,
        description="Bearer token for vLLM if it was started with --api-key.",
    )
    db_path: Path = Field(default=Path("switchboard.db"), description="SQLite registry file.")

    admin_token: SecretStr | None = Field(
        default=None,
        description="If set, the /admin API requires 'Authorization: Bearer <token>'.",
    )

    load_timeout_s: float = Field(default=120.0, gt=0, description="Timeout for a vLLM load.")
    unload_timeout_s: float = Field(default=30.0, gt=0)
    request_timeout_s: float = Field(
        default=600.0, gt=0, description="Upstream timeout for proxied inference requests."
    )

    drain_delay_s: float = Field(
        default=10.0,
        ge=0,
        description="Minimum wait after an alias moves away from a version before it may unload.",
    )
    drain_timeout_s: float = Field(
        default=300.0,
        ge=0,
        description="Maximum extra wait for in-flight requests on a version to finish.",
    )
    keep_loaded: int = Field(
        default=2,
        ge=1,
        description="Versions per alias kept loaded in vLLM (current + N-1 rollback targets).",
    )
    auto_gc: bool = Field(
        default=True, description="Unload versions beyond keep_loaded after each promote."
    )
    reconcile_interval_s: float = Field(
        default=30.0,
        ge=0,
        description="How often to re-load alias targets missing from vLLM (0 disables).",
    )

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit one JSON object per log line.")

    @field_validator("vllm_url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if value.endswith("/v1"):
            value = value[: -len("/v1")]
        return value
