"""FastAPI application factory and wiring."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import __version__
from .api import admin, openai
from .config import Settings
from .errors import SwitchboardError
from .metrics import Metrics
from .proxy import ProxyService
from .registry import Registry
from .router import TrafficRouter
from .service import RolloutService
from .vllm import VLLMClient

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    vllm_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the app. ``vllm_transport`` lets tests run vLLM in-process."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        metrics = Metrics()
        registry = Registry(settings.db_path)
        vllm = VLLMClient(
            settings.vllm_url,
            api_key=settings.vllm_api_key.get_secret_value() if settings.vllm_api_key else None,
            load_timeout_s=settings.load_timeout_s,
            unload_timeout_s=settings.unload_timeout_s,
            request_timeout_s=settings.request_timeout_s,
            transport=vllm_transport,
        )
        router = TrafficRouter(metrics)
        rollout = RolloutService(settings, registry, vllm, router, metrics)
        rollout.load_routes()

        app.state.settings = settings
        app.state.metrics = metrics
        app.state.rollout = rollout
        app.state.vllm = vllm
        app.state.proxy = ProxyService(router, vllm, rollout, metrics)

        result = await rollout.reconcile()
        log.info(
            "switchboard started",
            extra={
                "version": __version__,
                "vllm_url": settings.vllm_url,
                "vllm_reachable": result.reachable,
                "aliases": len(router.routes()),
            },
        )
        reconciler: asyncio.Task[None] | None = None
        if settings.reconcile_interval_s > 0:
            reconciler = asyncio.create_task(
                rollout.reconcile_forever(settings.reconcile_interval_s), name="reconcile"
            )
        try:
            yield
        finally:
            if reconciler is not None:
                reconciler.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reconciler
            await rollout.shutdown()
            await vllm.aclose()
            registry.close()
            log.info("switchboard stopped")

    app = FastAPI(
        title="lora-switchboard",
        version=__version__,
        summary="Safe LoRA adapter rollouts for vLLM.",
        lifespan=lifespan,
    )

    @app.exception_handler(SwitchboardError)
    async def _domain_error(_: Request, exc: SwitchboardError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    @app.exception_handler(ValidationError)
    async def _validation_error(_: Request, exc: ValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors(include_url=False)]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": str(exc.title),
                    "details": {"errors": errors},
                }
            },
        )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> JSONResponse:
        vllm: VLLMClient = request.app.state.vllm
        try:
            await vllm.list_models()
        except SwitchboardError as exc:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "reason": exc.message}
            )
        return JSONResponse({"status": "ready"})

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: Request) -> Response:
        m: Metrics = request.app.state.metrics
        return Response(m.render(), media_type=m.content_type)

    app.include_router(admin.router)
    app.include_router(openai.router)
    return app
