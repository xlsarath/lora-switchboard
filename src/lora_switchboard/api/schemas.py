"""Request bodies for the admin API. Responses reuse the domain/service models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import NAME_PATTERN
from ..smoke import SmokeSpec


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterRequest(_Body):
    uri: str = Field(min_length=1, description="Path or URI vLLM can load the adapter from.")
    base_model: str | None = Field(default=None, description="Required for a new adapter.")
    version: int | None = Field(default=None, gt=0, description="Defaults to the next number.")
    sha256: str | None = None
    notes: str | None = None
    description: str | None = None


class SmokeInput(_Body):
    """Either a parsed suite or raw YAML text (what the CLI sends)."""

    smoke: SmokeSpec | None = None
    smoke_yaml: str | None = None

    def smoke_spec(self) -> SmokeSpec | None:
        if self.smoke_yaml is not None:
            return SmokeSpec.from_yaml(self.smoke_yaml)
        return self.smoke

    @model_validator(mode="after")
    def _one_source(self) -> SmokeInput:
        if self.smoke is not None and self.smoke_yaml is not None:
            raise ValueError("pass either 'smoke' or 'smoke_yaml', not both")
        return self


class DeployRequest(SmokeInput):
    adapter: str = Field(pattern=NAME_PATTERN)
    version: int | None = Field(default=None, gt=0, description="Deploy an existing version...")
    uri: str | None = Field(default=None, description="...or register the next one from a URI.")
    base_model: str | None = None
    sha256: str | None = None
    notes: str | None = None
    alias: str | None = Field(default=None, pattern=NAME_PATTERN)
    skip_smoke: bool = Field(default=False, description="Explicitly deploy without smoke tests.")

    @model_validator(mode="after")
    def _version_or_uri(self) -> DeployRequest:
        if (self.version is None) == (self.uri is None):
            raise ValueError("pass exactly one of 'version' or 'uri'")
        has_smoke = self.smoke is not None or self.smoke_yaml is not None
        if not has_smoke and not self.skip_smoke:
            raise ValueError("a smoke suite is required; pass skip_smoke=true to deploy without")
        if has_smoke and self.skip_smoke:
            raise ValueError("skip_smoke=true conflicts with a smoke suite")
        return self


class PromoteRequest(_Body):
    version: int = Field(gt=0)
    adapter: str | None = Field(default=None, pattern=NAME_PATTERN)
    force: bool = False


class RollbackRequest(_Body):
    to_version: int | None = Field(default=None, gt=0)


class UnloadRequest(_Body):
    drain: bool = True


class GcRequest(_Body):
    keep: int | None = Field(default=None, ge=1)
