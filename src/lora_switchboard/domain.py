"""Core domain types shared by the registry, services and API."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field

from .errors import ValidationFailed

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_NAME_RE = re.compile(NAME_PATTERN)
_REF_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,63})@v(?P<version>[1-9][0-9]*)$")


def validate_name(value: str, *, kind: str) -> str:
    """Validate an adapter or alias name. ``@`` is reserved for version references."""
    if not _NAME_RE.match(value):
        raise ValidationFailed(
            f"invalid {kind} name {value!r}: use 1-64 chars of letters, digits, '.', '_' or '-',"
            " starting with a letter or digit"
        )
    return value


def served_name(adapter: str, version: int) -> str:
    """The immutable name a version is loaded under in vLLM, e.g. ``support-bot@v7``."""
    return f"{adapter}@v{version}"


def parse_ref(ref: str) -> tuple[str, int | None]:
    """Parse ``name`` or ``name@vN`` into ``(name, version-or-None)``."""
    match = _REF_RE.match(ref)
    if match:
        return match["name"], int(match["version"])
    return validate_name(ref, kind="adapter"), None


class VersionState(StrEnum):
    """Lifecycle of a version. ``loaded`` (in vLLM or not) is tracked separately."""

    REGISTERED = "registered"
    LOAD_FAILED = "load_failed"
    SMOKE_FAILED = "smoke_failed"
    READY = "ready"
    ROLLED_BACK = "rolled_back"

    @property
    def promotable(self) -> bool:
        return self in {VersionState.READY, VersionState.ROLLED_BACK}


class EventType(StrEnum):
    REGISTER = "register"
    LOAD = "load"
    LOAD_FAIL = "load_fail"
    SMOKE_PASS = "smoke_pass"
    SMOKE_FAIL = "smoke_fail"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    UNLOAD = "unload"
    UNLOAD_FAIL = "unload_fail"
    RELOAD = "reload"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Adapter(_Record):
    id: int
    name: str
    base_model: str
    description: str | None
    created_at: datetime


class Version(_Record):
    id: int
    adapter_id: int
    adapter: str
    version: int
    uri: str
    base_model: str
    sha256: str | None
    notes: str | None
    state: VersionState
    loaded: bool
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def served_name(self) -> str:
        return served_name(self.adapter, self.version)


class Alias(_Record):
    name: str
    adapter_id: int
    adapter: str
    version_id: int
    version: int
    revision: int
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target(self) -> str:
        return served_name(self.adapter, self.version)


class Event(_Record):
    id: int
    at: datetime
    type: EventType
    adapter: str
    version: int | None
    alias: str | None
    from_version: int | None
    duration_ms: float | None
    detail: dict[str, Any]
