"""Admin REST API. Handlers only translate HTTP <-> service calls."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, status

from ..domain import Adapter, Alias, Event, Version
from ..errors import Unauthorized
from ..service import (
    DeployResult,
    FlipResult,
    GcResult,
    ReconcileResult,
    RolloutService,
    StatusReport,
)
from .schemas import (
    DeployRequest,
    GcRequest,
    PromoteRequest,
    RegisterRequest,
    RollbackRequest,
    UnloadRequest,
)


def get_service(request: Request) -> RolloutService:
    service: RolloutService = request.app.state.rollout
    return service


def require_admin(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    token = request.app.state.settings.admin_token
    if token is None:
        return
    expected = f"Bearer {token.get_secret_value()}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise Unauthorized("missing or invalid admin token")


Service = Annotated[RolloutService, Depends(get_service)]

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/status")
async def get_status(service: Service) -> StatusReport:
    return await service.status()


@router.get("/adapters")
def list_adapters(service: Service) -> list[Adapter]:
    return service.list_adapters()


@router.get("/versions")
def list_versions(service: Service, adapter: str | None = None) -> list[Version]:
    return service.list_versions(adapter)


@router.post("/adapters/{adapter}/versions", status_code=status.HTTP_201_CREATED)
def register_version(adapter: str, body: RegisterRequest, service: Service) -> Version:
    return service.register(
        adapter,
        body.uri,
        base_model=body.base_model,
        version=body.version,
        sha256=body.sha256,
        notes=body.notes,
        description=body.description,
    )


@router.post("/deploy")
async def deploy(body: DeployRequest, service: Service) -> DeployResult:
    smoke = body.smoke_spec()
    if body.uri is not None:
        return await service.register_and_deploy(
            body.adapter,
            body.uri,
            alias=body.alias,
            smoke=smoke,
            base_model=body.base_model,
            sha256=body.sha256,
            notes=body.notes,
        )
    assert body.version is not None  # guaranteed by DeployRequest validation
    return await service.deploy(body.adapter, body.version, alias=body.alias, smoke=smoke)


@router.get("/aliases")
def list_aliases(service: Service) -> list[Alias]:
    return service.list_aliases()


@router.get("/aliases/{alias}")
def get_alias(alias: str, service: Service) -> Alias:
    return service.get_alias(alias)


@router.post("/aliases/{alias}/promote")
async def promote(alias: str, body: PromoteRequest, service: Service) -> FlipResult:
    return await service.promote(alias, body.version, adapter=body.adapter, force=body.force)


@router.post("/aliases/{alias}/rollback")
async def rollback(alias: str, service: Service, body: RollbackRequest | None = None) -> FlipResult:
    return await service.rollback(alias, to_version=body.to_version if body else None)


@router.post("/adapters/{adapter}/versions/{version}/unload")
async def unload(
    adapter: str, version: int, service: Service, body: UnloadRequest | None = None
) -> Version:
    return await service.unload(adapter, version, drain=body.drain if body else True)


@router.post("/adapters/{adapter}/gc")
async def gc(adapter: str, service: Service, body: GcRequest | None = None) -> GcResult:
    return await service.gc(adapter, keep=body.keep if body else None)


@router.post("/reconcile")
async def reconcile(service: Service) -> ReconcileResult:
    return await service.reconcile()


@router.get("/history")
def history(
    service: Service,
    adapter: str | None = None,
    alias: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[Event]:
    return service.history(adapter=adapter, alias=alias, limit=limit)
