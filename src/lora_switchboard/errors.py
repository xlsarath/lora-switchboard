"""Domain errors.

Every error carries an HTTP status so the API layer can translate it with a
single exception handler instead of per-route ``try``/``except`` blocks.
"""

from __future__ import annotations

from typing import Any


class SwitchboardError(Exception):
    """Base class for all expected, user-facing failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


class Unauthorized(SwitchboardError):
    status_code = 401
    code = "unauthorized"


class NotFoundError(SwitchboardError):
    status_code = 404
    code = "not_found"


class ConflictError(SwitchboardError):
    """The request is valid but clashes with current state (e.g. a concurrent change)."""

    status_code = 409
    code = "conflict"


class ValidationFailed(SwitchboardError):
    status_code = 422
    code = "invalid_request"


class SmokeTestFailed(SwitchboardError):
    status_code = 422
    code = "smoke_failed"


class UpstreamError(SwitchboardError):
    """vLLM answered, but with an error."""

    status_code = 502
    code = "upstream_error"


class UpstreamUnavailable(SwitchboardError):
    """vLLM could not be reached, or did not answer in time."""

    status_code = 503
    code = "upstream_unavailable"
