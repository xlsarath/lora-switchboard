from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from lora_switchboard.domain import EventType, VersionState
from lora_switchboard.errors import (
    ConflictError,
    NotFoundError,
    SmokeTestFailed,
    UpstreamError,
    UpstreamUnavailable,
    ValidationFailed,
)
from lora_switchboard.metrics import Metrics
from lora_switchboard.registry import Registry
from lora_switchboard.router import TrafficRouter
from lora_switchboard.service import RolloutService
from lora_switchboard.smoke import SmokeSpec
from lora_switchboard.vllm import VLLMClient

from .conftest import BASE_MODEL, Stack, make_settings


async def ship(stack: Stack, spec: SmokeSpec, uri: str, alias: str = "support") -> int:
    result = await stack.service.register_and_deploy(
        "support-bot", uri, alias=alias, smoke=spec, base_model=BASE_MODEL
    )
    return result.version.version


def event_types(stack: Stack) -> list[EventType]:
    return [e.type for e in reversed(stack.service.history(adapter="support-bot"))]


# ---------------------------------------------------------------------- deploy


async def test_deploy_loads_smokes_and_promotes(stack: Stack, smoke_spec: SmokeSpec) -> None:
    result = await stack.service.register_and_deploy(
        "support-bot",
        "/adapters/support/v1",
        alias="support",
        smoke=smoke_spec,
        base_model=BASE_MODEL,
        notes="initial",
    )
    assert result.smoke is not None
    assert result.smoke.passed
    assert result.flip is not None
    assert result.flip.from_version is None
    assert set(result.timings_ms) == {"load_ms", "smoke_ms", "flip_ms", "total_ms"}
    assert result.version.state is VersionState.READY
    assert result.version.loaded

    assert stack.router.resolve("support") == "support-bot@v1"
    assert "support-bot@v1" in stack.fake.adapters
    assert event_types(stack) == [
        EventType.REGISTER,
        EventType.LOAD,
        EventType.SMOKE_PASS,
        EventType.PROMOTE,
    ]


async def test_smoke_failure_unloads_and_keeps_alias(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    with pytest.raises(SmokeTestFailed) as caught:
        await ship(stack, smoke_spec, "/adapters/support/v2-broken")

    report = caught.value.details["report"]
    assert not report["passed"]
    assert {c["name"] for c in report["cases"] if not c["passed"]} == {
        "helpful reply",
        "structured output",
    }
    v2 = stack.service.get_version("support-bot", 2)
    assert v2.state is VersionState.SMOKE_FAILED
    assert not v2.loaded
    assert "support-bot@v2" not in stack.fake.adapters
    assert stack.router.resolve("support") == "support-bot@v1"
    assert stack.service.get_alias("support").version == 1
    assert event_types(stack)[-1] is EventType.UNLOAD
    assert EventType.SMOKE_FAIL in event_types(stack)


async def test_load_failure_is_recorded_without_routing(
    stack: Stack, smoke_spec: SmokeSpec
) -> None:
    with pytest.raises(UpstreamError, match="refused to load"):
        await ship(stack, smoke_spec, "/adapters/missing/v1")
    version = stack.service.get_version("support-bot", 1)
    assert version.state is VersionState.LOAD_FAILED
    assert not version.loaded
    assert stack.service.list_aliases() == []
    assert stack.router.routes() == {}


async def test_load_timeout_cleans_up(stack: Stack, smoke_spec: SmokeSpec) -> None:
    with pytest.raises(UpstreamUnavailable, match="did not answer"):
        await ship(stack, smoke_spec, "/adapters/slow/v1")
    version = stack.service.get_version("support-bot", 1)
    assert version.state is VersionState.LOAD_FAILED
    assert not version.loaded
    load_fail = stack.service.history(adapter="support-bot")[0]
    assert load_fail.type is EventType.LOAD_FAIL


async def test_vllm_down_leaves_consistent_state(tmp_path: Path, smoke_spec: SmokeSpec) -> None:
    settings = make_settings(tmp_path, vllm_url="http://127.0.0.1:9")
    registry = Registry(settings.db_path)
    vllm = VLLMClient(settings.vllm_url)
    service = RolloutService(settings, registry, vllm, TrafficRouter(), Metrics())
    try:
        with pytest.raises(UpstreamUnavailable, match="cannot reach vLLM"):
            await service.register_and_deploy(
                "support-bot", "/a/1", alias="support", smoke=smoke_spec, base_model=BASE_MODEL
            )
        assert service.get_version("support-bot", 1).state is VersionState.LOAD_FAILED
        assert service.list_aliases() == []
        status = await service.status()
        assert not status.vllm_reachable
        assert status.vllm_error
        assert (await service.reconcile()).reachable is False
    finally:
        await vllm.aclose()
        registry.close()


async def test_deploy_is_idempotent_for_live_version(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    again = await stack.service.deploy("support-bot", 1, alias="support", smoke=smoke_spec)
    assert again.flip is None
    assert again.timings_ms == {}
    assert stack.fake.load_calls == ["support-bot@v1"]


async def test_concurrent_deploy_of_same_version_conflicts(
    stack: Stack, smoke_spec: SmokeSpec
) -> None:
    stack.service.register("support-bot", "/adapters/support/v1", base_model=BASE_MODEL)
    results = await asyncio.gather(
        stack.service.deploy("support-bot", 1, alias="support", smoke=smoke_spec),
        stack.service.deploy("support-bot", 1, alias="support", smoke=smoke_spec),
        return_exceptions=True,
    )
    assert sum(isinstance(r, ConflictError) for r in results) == 1
    assert stack.service.get_alias("support").version == 1


async def test_register_validations(stack: Stack) -> None:
    with pytest.raises(ValidationFailed, match="base_model is required"):
        stack.service.register("support-bot", "/a/1")
    with pytest.raises(ValidationFailed, match="invalid adapter name"):
        stack.service.register("bad@name", "/a/1", base_model=BASE_MODEL)
    with pytest.raises(ValidationFailed, match="sha256"):
        stack.service.register("support-bot", "/a/1", base_model=BASE_MODEL, sha256="abc")
    stack.service.register("support-bot", "/a/1", base_model=BASE_MODEL)
    with pytest.raises(ConflictError, match="bound to base model"):
        stack.service.register("support-bot", "/a/2", base_model="other/model")
    with pytest.raises(NotFoundError):
        stack.service.get_version("support-bot", 9)


# ---------------------------------------------------------------------- promote & rollback


async def test_rollback_flips_to_previous_good_version(stack: Stack, smoke_spec: SmokeSpec) -> None:
    for n in (1, 2, 3):
        await ship(stack, smoke_spec, f"/adapters/support/v{n}")
    assert stack.router.resolve("support") == "support-bot@v3"

    flip = await stack.service.rollback("support")
    assert (flip.from_version, flip.to_version, flip.reloaded) == (3, 2, False)
    assert flip.total_ms < 250  # no load needed: a DB write and a dict update
    assert stack.router.resolve("support") == "support-bot@v2"
    assert stack.service.get_version("support-bot", 3).state is VersionState.ROLLED_BACK

    # A second rollback skips the rolled-back v3 and goes to v1.
    flip = await stack.service.rollback("support")
    assert (flip.from_version, flip.to_version) == (2, 1)

    with pytest.raises(ConflictError, match="no earlier good version"):
        await stack.service.rollback("support")

    # Explicit target may re-promote a rolled-back version.
    flip = await stack.service.rollback("support", to_version=3)
    assert flip.to_version == 3


async def test_rollback_reloads_target_that_was_unloaded(
    stack: Stack, smoke_spec: SmokeSpec
) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    await ship(stack, smoke_spec, "/adapters/support/v2")
    await stack.service.unload("support-bot", 1)
    assert "support-bot@v1" not in stack.fake.adapters

    flip = await stack.service.rollback("support")
    assert flip.reloaded
    assert flip.to_version == 1
    assert "support-bot@v1" in stack.fake.adapters
    assert EventType.RELOAD in event_types(stack)


async def test_promote_rules(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    stack.service.register("support-bot", "/adapters/support/v2")
    with pytest.raises(ConflictError, match="registered"):
        await stack.service.promote("support", 2)
    flip = await stack.service.promote("support", 2, force=True)
    assert flip.reloaded
    assert flip.to_version == 2

    with pytest.raises(ValidationFailed, match="name the adapter"):
        await stack.service.promote("brand-new", 1)

    stack.service.register("sql-helper", "/adapters/sql/v1", base_model=BASE_MODEL)
    with pytest.raises(ConflictError, match="belongs to adapter"):
        await stack.service.promote("support", 1, adapter="sql-helper", force=True)
    with pytest.raises(NotFoundError):
        await stack.service.rollback("no-such-alias")


async def test_concurrent_promotes_are_serialised(stack: Stack, smoke_spec: SmokeSpec) -> None:
    for n in (1, 2, 3):
        await stack.service.register_and_deploy(
            "support-bot",
            f"/adapters/support/v{n}",
            alias=None,
            smoke=smoke_spec,
            base_model=BASE_MODEL,
        )
    await stack.service.promote("support", 1, adapter="support-bot")

    flips = await asyncio.gather(*(stack.service.promote("support", n) for n in (2, 3, 2, 3)))
    alias = stack.service.get_alias("support")
    promotes = [e for e in stack.service.history(alias="support") if e.type is EventType.PROMOTE]
    real_flips = [f for f in flips if f.from_version != f.to_version]
    # Every effective flip bumped the revision exactly once, and the log agrees.
    assert alias.revision == 1 + len(real_flips) == len(promotes)
    assert (
        stack.router.resolve("support")
        == alias.target
        == promotes[0].adapter + f"@v{promotes[0].version}"
    )


async def test_cross_process_lost_update_is_detected(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    await stack.service.register_and_deploy(
        "support-bot", "/adapters/support/v2", alias=None, smoke=smoke_spec
    )
    alias = stack.service.get_alias("support")
    # Simulate another process flipping the alias behind this one's back.
    v2 = stack.service.get_version("support-bot", 2)
    stack.registry.set_alias("support", v2.adapter_id, v2.id, expected_revision=alias.revision)
    with pytest.raises(ConflictError, match="changed concurrently"):
        stack.registry.set_alias("support", v2.adapter_id, v2.id, expected_revision=alias.revision)


# ---------------------------------------------------------------------- unload, drain, gc


async def test_unload_refuses_live_version(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    with pytest.raises(ConflictError, match="target of an alias"):
        await stack.service.unload("support-bot", 1)


async def test_drain_waits_for_inflight_requests(stack: Stack, smoke_spec: SmokeSpec) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    lease = stack.router.acquire("support")  # a request in flight on v1
    await ship(stack, smoke_spec, "/adapters/support/v2")
    assert lease.target == "support-bot@v1"
    assert stack.router.resolve("support") == "support-bot@v2"

    unload = asyncio.create_task(stack.service.unload("support-bot", 1))
    await asyncio.sleep(0.1)
    assert not unload.done()
    assert "support-bot@v1" in stack.fake.adapters  # still serving the in-flight request
    assert (await stack.service.status()).unloading == ["support-bot@v1"]

    lease.release()
    version = await asyncio.wait_for(unload, timeout=2)
    assert not version.loaded
    assert "support-bot@v1" not in stack.fake.adapters


async def test_rollback_during_drain_cancels_unload(stack: Stack, smoke_spec: SmokeSpec) -> None:
    stack.settings.drain_delay_s = 0.2
    await ship(stack, smoke_spec, "/adapters/support/v1")
    await ship(stack, smoke_spec, "/adapters/support/v2")

    unload = asyncio.create_task(stack.service.unload("support-bot", 1))
    await asyncio.sleep(0.05)  # unload is in its drain delay
    await stack.service.rollback("support")  # back to v1
    with pytest.raises(ConflictError, match="target of an alias"):
        await unload
    assert "support-bot@v1" in stack.fake.adapters
    assert stack.router.resolve("support") == "support-bot@v1"


async def test_gc_keeps_alias_target_and_newest_rollback_candidates(
    stack: Stack, smoke_spec: SmokeSpec
) -> None:
    for n in (1, 2, 3, 4):
        await ship(stack, smoke_spec, f"/adapters/support/v{n}")
    result = await stack.service.gc("support-bot", keep=2)
    assert sorted(result.kept) == ["support-bot@v3", "support-bot@v4"]
    assert sorted(result.unloaded) == ["support-bot@v1", "support-bot@v2"]
    assert sorted(stack.fake.adapters) == ["support-bot@v3", "support-bot@v4"]

    result = await stack.service.gc("support-bot", keep=1)
    assert result.kept == ["support-bot@v4"]
    assert result.unloaded == ["support-bot@v3"]


async def test_auto_gc_after_promote_respects_drain(tmp_path: Path, fake_app, smoke_spec) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(tmp_path, keep_loaded=1, auto_gc=True, drain_delay_s=0.05)
    registry = Registry(settings.db_path)
    vllm = VLLMClient("http://fake", transport=httpx.ASGITransport(app=fake_app))
    router = TrafficRouter()
    service = RolloutService(settings, registry, vllm, router, Metrics())
    try:
        for n in (1, 2):
            await service.register_and_deploy(
                "support-bot", f"/a/{n}", alias="support", smoke=smoke_spec, base_model=BASE_MODEL
            )
        lease = router.acquire("support-bot@v1")  # a pinned request still running on v1
        await asyncio.sleep(0.2)
        assert "support-bot@v1" in fake_app.state.fake.adapters
        lease.release()
        await service.wait_background()
        assert sorted(fake_app.state.fake.adapters) == ["support-bot@v2"]
    finally:
        await service.shutdown()
        await vllm.aclose()
        registry.close()


# ---------------------------------------------------------------------- reconcile


async def test_reconcile_reloads_alias_targets_after_vllm_restart(
    stack: Stack, smoke_spec: SmokeSpec
) -> None:
    await ship(stack, smoke_spec, "/adapters/support/v1")
    await ship(stack, smoke_spec, "/adapters/support/v2")
    stack.fake.adapters.clear()  # vLLM restarted: runtime-loaded adapters are gone
    stack.fake.adapters["support-bot@v9"] = "/somewhere"  # loaded by hand, unknown to us

    result = await stack.service.reconcile()
    assert result.reloaded == ["support-bot@v2"]
    assert sorted(result.marked_unloaded) == ["support-bot@v1", "support-bot@v2"]
    assert result.unknown_in_vllm == ["support-bot@v9"]
    assert stack.service.get_version("support-bot", 2).loaded
    assert not stack.service.get_version("support-bot", 1).loaded
    assert "support-bot@v2" in stack.fake.adapters
