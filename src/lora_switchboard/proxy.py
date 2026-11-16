"""OpenAI-compatible routing proxy.

Clients send ``model: <alias>``; the proxy swaps in the concrete served name
(``support-bot@v7``) and streams vLLM's response back byte for byte, so SSE
streams, usage chunks and error bodies all pass through unchanged. Names that
are not aliases (the base model, a pinned ``name@vN``) are forwarded as-is.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from .metrics import Metrics
from .router import Lease, TrafficRouter
from .service import RolloutService
from .vllm import VLLMClient

log = logging.getLogger(__name__)

TARGET_HEADER = "x-switchboard-target"

# vLLM's own admin endpoints must never be reachable through the public proxy.
BLOCKED_PATHS = frozenset({"load_lora_adapter", "unload_lora_adapter"})

_FORWARD_REQUEST_HEADERS = ("authorization", "accept", "x-request-id", "traceparent", "tracestate")
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
    }
)


def openai_error(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "param": None, "code": code}},
    )


class ProxyService:
    def __init__(
        self,
        router: TrafficRouter,
        vllm: VLLMClient,
        rollout: RolloutService,
        metrics: Metrics,
    ) -> None:
        self._router = router
        self._vllm = vllm
        self._rollout = rollout
        self._metrics = metrics

    async def forward(self, path: str, raw_body: bytes, headers: Mapping[str, str]) -> Response:
        if path in BLOCKED_PATHS:
            return openai_error(404, f"/v1/{path} is not exposed by the proxy", "not_found")
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except ValueError:
            return openai_error(400, "request body must be JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return openai_error(400, "request body must be a JSON object", "invalid_request_error")

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            return openai_error(400, "'model' is required", "invalid_request_error", "model")

        lease = self._router.acquire(model)
        payload["model"] = lease.target
        started = time.perf_counter()
        request = self._vllm.http.build_request(
            "POST",
            f"/v1/{path}",
            content=json.dumps(payload).encode(),
            headers=self._upstream_headers(headers),
        )
        if "authorization" not in headers:
            # Never lend the switchboard's own vLLM key to an unauthenticated client.
            request.headers.pop("authorization", None)
        try:
            upstream = await self._vllm.http.send(request, stream=True)
        except httpx.TimeoutException:
            self._finish(lease, started, "timeout")
            return openai_error(504, "upstream model server timed out", "upstream_timeout")
        except httpx.TransportError as exc:
            self._finish(lease, started, "unreachable")
            log.warning(
                "upstream unreachable",
                extra={"alias": lease.alias, "target": lease.target, "error": type(exc).__name__},
            )
            return openai_error(502, "upstream model server is unreachable", "upstream_error")
        except BaseException:
            self._finish(lease, started, "error")
            raise

        return StreamingResponse(
            self._relay(upstream, lease, started),
            status_code=upstream.status_code,
            headers=self._downstream_headers(upstream, lease),
        )

    async def _relay(
        self, upstream: httpx.Response, lease: Lease, started: float
    ) -> AsyncIterator[bytes]:
        outcome = str(upstream.status_code)
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.TransportError as exc:
            outcome = "broken_stream"
            log.warning(
                "upstream stream broke",
                extra={"alias": lease.alias, "target": lease.target, "error": type(exc).__name__},
            )
        finally:
            await upstream.aclose()
            self._finish(lease, started, outcome)

    def _finish(self, lease: Lease, started: float, outcome: str) -> None:
        lease.release()
        alias_label = lease.alias or "-"
        self._metrics.requests.labels(alias=alias_label, target=lease.target, status=outcome).inc()
        self._metrics.request_seconds.labels(alias=alias_label).observe(
            time.perf_counter() - started
        )
        log.debug(
            "proxied request finished",
            extra={"alias": lease.alias, "target": lease.target, "status": outcome},
        )

    @staticmethod
    def _upstream_headers(headers: Mapping[str, str]) -> dict[str, str]:
        out = {"content-type": "application/json"}
        for name in _FORWARD_REQUEST_HEADERS:
            value = headers.get(name)
            if value:
                out[name] = value
        return out

    @staticmethod
    def _downstream_headers(upstream: httpx.Response, lease: Lease) -> dict[str, str]:
        out = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
        out[TARGET_HEADER] = lease.target
        return out

    async def list_models(self, headers: Mapping[str, str]) -> Response:
        """vLLM's model list plus one entry per alias (``root`` = current target)."""
        try:
            response = await self._vllm.http.get(
                "/v1/models", headers=self._upstream_headers(headers), timeout=10.0
            )
        except httpx.TransportError:
            return openai_error(502, "upstream model server is unreachable", "upstream_error")
        if response.status_code != 200:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        body: dict[str, Any] = response.json()
        data: list[dict[str, Any]] = list(body.get("data", []))
        base_models = {a.id: a.base_model for a in self._rollout.list_adapters()}
        created = int(time.time())
        for alias in self._rollout.list_aliases():
            data.append(
                {
                    "id": alias.name,
                    "object": "model",
                    "created": created,
                    "owned_by": "lora-switchboard",
                    "root": alias.target,
                    "parent": base_models.get(alias.adapter_id),
                }
            )
        return JSONResponse({"object": "list", "data": data})
