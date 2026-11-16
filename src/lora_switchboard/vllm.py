"""Async client for the parts of the vLLM OpenAI-compatible server we use.

Admin endpoints (runtime LoRA updates, needs ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``):

* ``POST /v1/load_lora_adapter``   ``{"lora_name": ..., "lora_path": ...}``
* ``POST /v1/unload_lora_adapter`` ``{"lora_name": ...}``
* ``GET  /v1/models``              base model plus every loaded adapter

Load and unload are made idempotent here: vLLM answers 400 "has already been
loaded" for a duplicate load and 404 "cannot be found" for an unknown unload.
Because served names are immutable (``name@vN`` always means the same
weights), both outcomes are treated as success.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .errors import UpstreamError, UpstreamUnavailable

log = logging.getLogger(__name__)

_ALREADY_LOADED = "already been loaded"
_NOT_FOUND = "cannot be found"


@dataclass(frozen=True, slots=True)
class Completion:
    """Result of a single non-streaming inference call."""

    text: str
    latency_ms: float
    finish_reason: str | None


def error_message(response: httpx.Response) -> str:
    """Extract a human-readable message from a vLLM / OpenAI-style error body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if body.get("message"):
            return str(body["message"])
        if body.get("detail"):
            return str(body["detail"])
    return response.text[:500]


class VLLMClient:
    """Thin, typed wrapper over one ``httpx.AsyncClient`` pointed at vLLM."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        load_timeout_s: float = 120.0,
        unload_timeout_s: float = 30.0,
        request_timeout_s: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.base_url = base_url
        self._load_timeout = load_timeout_s
        self._unload_timeout = unload_timeout_s
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(request_timeout_s, connect=5.0),
            transport=transport,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        )

    @property
    def http(self) -> httpx.AsyncClient:
        """The shared connection pool, used by the proxy to stream requests through."""
        return self._http

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, body: Mapping[str, Any], timeout_s: float) -> httpx.Response:
        try:
            # asyncio.timeout bounds the whole exchange, not just each socket read.
            async with asyncio.timeout(timeout_s):
                return await self._http.post(path, json=body, timeout=timeout_s)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise UpstreamUnavailable(
                f"vLLM did not answer {path} within {timeout_s:g}s", details={"path": path}
            ) from exc
        except httpx.TransportError as exc:
            raise UpstreamUnavailable(
                f"cannot reach vLLM at {self.base_url}: {exc.__class__.__name__}",
                details={"path": path},
            ) from exc

    # ------------------------------------------------------------------ admin

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._http.get("/v1/models", timeout=10.0)
        except httpx.TransportError as exc:
            raise UpstreamUnavailable(
                f"cannot reach vLLM at {self.base_url}: {exc.__class__.__name__}"
            ) from exc
        if response.status_code != 200:
            raise UpstreamError(f"GET /v1/models failed: {error_message(response)}")
        data = response.json().get("data", [])
        return [m for m in data if isinstance(m, dict)]

    async def loaded_model_ids(self) -> set[str]:
        return {str(m.get("id")) for m in await self.list_models()}

    async def load_adapter(self, name: str, path: str) -> None:
        response = await self._post(
            "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, self._load_timeout
        )
        if response.status_code == 200:
            return
        message = error_message(response)
        if response.status_code == 400 and _ALREADY_LOADED in message:
            log.info("adapter already loaded in vLLM", extra={"target": name})
            return
        raise UpstreamError(
            f"vLLM refused to load {name!r}: {message}",
            details={"status": response.status_code},
        )

    async def unload_adapter(self, name: str) -> None:
        response = await self._post(
            "/v1/unload_lora_adapter", {"lora_name": name}, self._unload_timeout
        )
        if response.status_code == 200:
            return
        message = error_message(response)
        if response.status_code == 404 or _NOT_FOUND in message:
            log.info("adapter was not loaded in vLLM", extra={"target": name})
            return
        raise UpstreamError(
            f"vLLM refused to unload {name!r}: {message}",
            details={"status": response.status_code},
        )

    # ------------------------------------------------------------------ inference

    async def complete(
        self,
        model: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 60.0,
    ) -> Completion:
        """One non-streaming chat (``messages``) or text (``prompt``) completion."""
        if (messages is None) == (prompt is None):
            raise ValueError("pass exactly one of messages or prompt")
        body: dict[str, Any] = {"model": model, **(params or {}), "stream": False}
        if messages is not None:
            path, body["messages"] = "/v1/chat/completions", messages
        else:
            path, body["prompt"] = "/v1/completions", prompt
        started = time.perf_counter()
        response = await self._post(path, body, timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise UpstreamError(
                f"{path} failed for {model!r}: {error_message(response)}",
                details={"status": response.status_code},
            )
        choice = (response.json().get("choices") or [{}])[0]
        if messages is not None:
            text = (choice.get("message") or {}).get("content") or ""
        else:
            text = choice.get("text") or ""
        return Completion(
            text=text, latency_ms=latency_ms, finish_reason=choice.get("finish_reason")
        )
