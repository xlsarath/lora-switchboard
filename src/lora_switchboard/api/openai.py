"""OpenAI-compatible routes. All work happens in :class:`ProxyService`."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..proxy import ProxyService

router = APIRouter(prefix="/v1", tags=["openai"])


def _proxy(request: Request) -> ProxyService:
    proxy: ProxyService = request.app.state.proxy
    return proxy


@router.get("/models")
async def list_models(request: Request) -> Response:
    return await _proxy(request).list_models(request.headers)


@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy(request).forward("chat/completions", await request.body(), request.headers)


@router.post("/completions")
async def completions(request: Request) -> Response:
    return await _proxy(request).forward("completions", await request.body(), request.headers)


@router.post("/{path:path}", include_in_schema=False)
async def passthrough(path: str, request: Request) -> Response:
    """Other model-addressed endpoints (embeddings, responses, tokenize, ...)."""
    return await _proxy(request).forward(path, await request.body(), request.headers)
