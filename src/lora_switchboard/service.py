"""Rollout orchestration: register, deploy, promote, rollback, unload, gc, reconcile.

Concurrency model
-----------------
The service runs on one asyncio event loop. Invariants it protects:

1. **An alias never routes to a version that is not loaded.** Flips happen
   only after the target is confirmed loaded, and unloads refuse versions that
   an alias references or that a promote is preparing (``_reserved``).
2. **Alias flips are serialised.** Each adapter has an ``asyncio.Lock`` held
   only around the short flip critical section; the database update is also a
   compare-and-swap on the alias revision, so a concurrent writer in another
   process gets a 409 rather than a lost update.
3. **Rollback is never blocked by a slow deploy.** Loading and smoke testing
   happen outside the lock; only the final flip takes it.
4. **Unload drains first.** After an alias moves, the old version stays loaded
   for ``drain_delay_s`` and until in-flight requests finish (bounded by
   ``drain_timeout_s``); the "still unreferenced?" check is repeated under the
   lock after draining, because a rollback may have moved the alias back.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .domain import (
    Adapter,
    Alias,
    Event,
    EventType,
    Version,
    VersionState,
    served_name,
    validate_name,
)
from .errors import (
    ConflictError,
    NotFoundError,
    SmokeTestFailed,
    SwitchboardError,
    UpstreamUnavailable,
    ValidationFailed,
)
from .metrics import Metrics
from .registry import Registry
from .router import TrafficRouter
from .smoke import SmokeReport, SmokeRunner, SmokeSpec
from .vllm import VLLMClient

log = logging.getLogger(__name__)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


class FlipResult(BaseModel):
    alias: Alias
    from_version: int | None
    to_version: int
    reloaded: bool
    flip_ms: float
    total_ms: float


class DeployResult(BaseModel):
    version: Version
    smoke: SmokeReport | None
    flip: FlipResult | None
    timings_ms: dict[str, float]


class GcResult(BaseModel):
    unloaded: list[str] = Field(default_factory=list)
    kept: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)


class ReconcileResult(BaseModel):
    reachable: bool
    reloaded: list[str] = Field(default_factory=list)
    marked_unloaded: list[str] = Field(default_factory=list)
    marked_loaded: list[str] = Field(default_factory=list)
    unknown_in_vllm: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


class AliasStatus(BaseModel):
    alias: Alias
    inflight: int
    in_vllm: bool | None


class StatusReport(BaseModel):
    vllm_url: str
    vllm_reachable: bool
    vllm_error: str | None
    vllm_models: list[str]
    aliases: list[AliasStatus]
    loaded_versions: list[Version]
    unloading: list[str]
    deploying: list[str]


class RolloutService:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        vllm: VLLMClient,
        router: TrafficRouter,
        metrics: Metrics,
        *,
        clock: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._vllm = vllm
        self._router = router
        self._metrics = metrics
        self._smoke = SmokeRunner(vllm)
        self._sleep = clock
        self._adapter_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._reserved: Counter[int] = Counter()
        self._deploying: set[int] = set()
        self._unloading: dict[int, asyncio.Future[None]] = {}
        self._background: set[asyncio.Task[Any]] = set()

    # ================================================================== lifecycle

    def load_routes(self) -> None:
        """Populate the in-memory routing table from the registry."""
        aliases = self._registry.list_aliases()
        self._router.replace_routes({a.name: a.target for a in aliases})
        for alias in aliases:
            self._metrics.alias_version.labels(alias=alias.name, adapter=alias.adapter).set(
                alias.version
            )
        log.info("routes loaded", extra={"aliases": len(aliases)})

    async def shutdown(self) -> None:
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._background.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            self._background.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error("background task failed", extra={"task": name}, exc_info=t.exception())

        task.add_done_callback(_done)
        return task

    async def wait_background(self) -> None:
        """Wait for scheduled gc/drain work (used by tests and graceful shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    # ================================================================== queries

    def get_adapter(self, name: str) -> Adapter:
        adapter = self._registry.get_adapter(name)
        if adapter is None:
            raise NotFoundError(f"adapter {name!r} is not registered")
        return adapter

    def get_version(self, adapter: str, version: int) -> Version:
        found = self._registry.get_version(adapter, version)
        if found is None:
            self.get_adapter(adapter)  # raise the more specific error first
            raise NotFoundError(f"{served_name(adapter, version)} is not registered")
        return found

    def list_adapters(self) -> list[Adapter]:
        return self._registry.list_adapters()

    def list_versions(self, adapter: str | None = None) -> list[Version]:
        if adapter is not None:
            self.get_adapter(adapter)
        return self._registry.list_versions(adapter)

    def list_aliases(self) -> list[Alias]:
        return self._registry.list_aliases()

    def get_alias(self, name: str) -> Alias:
        alias = self._registry.get_alias(name)
        if alias is None:
            raise NotFoundError(f"alias {name!r} does not exist")
        return alias

    def history(
        self, *, adapter: str | None = None, alias: str | None = None, limit: int = 100
    ) -> list[Event]:
        return self._registry.list_events(adapter=adapter, alias=alias, limit=limit)

    def inflight(self, target: str) -> int:
        return self._router.inflight(target)

    # ================================================================== registration

    def register(
        self,
        adapter: str,
        uri: str,
        *,
        base_model: str | None = None,
        version: int | None = None,
        sha256: str | None = None,
        notes: str | None = None,
        description: str | None = None,
    ) -> Version:
        """Register a new immutable version (creating the adapter on first use)."""
        validate_name(adapter, kind="adapter")
        if not uri.strip():
            raise ValidationFailed("uri must not be empty")
        if sha256 is not None and (len(sha256) != 64 or not _is_hex(sha256)):
            raise ValidationFailed("sha256 must be 64 hex characters")
        with self._registry.transaction():
            record = self._registry.get_adapter(adapter)
            if record is None:
                if not base_model:
                    raise ValidationFailed(
                        f"adapter {adapter!r} is new: base_model is required on first register"
                    )
                record = self._registry.create_adapter(adapter, base_model, description)
            elif base_model and base_model != record.base_model:
                raise ConflictError(
                    f"adapter {adapter!r} is bound to base model {record.base_model!r}, "
                    f"not {base_model!r}; register a new adapter name instead"
                )
            number = version or self._registry.next_version_number(record.id)
            created = self._registry.create_version(record.id, number, uri, sha256, notes)
            self._record(EventType.REGISTER, created, detail={"uri": uri})
        log.info(
            "version registered",
            extra={"adapter": adapter, "version": created.version, "uri": uri},
        )
        return created

    # ================================================================== deploy

    async def deploy(
        self,
        adapter: str,
        version: int,
        *,
        alias: str | None,
        smoke: SmokeSpec | None,
    ) -> DeployResult:
        """Load -> smoke -> (optionally) promote. Leaves consistent state on every failure."""
        if alias is not None:
            validate_name(alias, kind="alias")
        target = self.get_version(adapter, version)
        if alias is not None:
            self._check_alias_binding(alias, target)
        routed_by = [a.name for a in self._registry.list_aliases() if a.version_id == target.id]
        if routed_by:
            if alias in routed_by:
                # Idempotent re-run (e.g. a retried CI job): already live, nothing to do.
                log.info(
                    "deploy is a no-op: version already live",
                    extra={"alias": alias, "target": target.served_name},
                )
                return DeployResult(version=target, smoke=None, flip=None, timings_ms={})
            raise ConflictError(
                f"{target.served_name} is live behind {', '.join(routed_by)}; "
                "deploy a new version instead of re-testing a live one"
            )
        if target.id in self._deploying:
            raise ConflictError(f"{target.served_name} is already being deployed")

        started = time.perf_counter()
        timings: dict[str, float] = {}
        self._deploying.add(target.id)
        self._reserved[target.id] += 1
        try:
            if target.id in self._unloading:
                await self._unloading[target.id]
            loaded = await self._load(target, event=EventType.LOAD)
            timings["load_ms"] = loaded[1]
            target = loaded[0]

            report: SmokeReport | None = None
            if smoke is not None:
                report = await self._run_smoke(target, smoke)
                timings["smoke_ms"] = round(report.duration_ms, 3)
                if not report.passed:
                    await self._reject(target, report)
                    live = self._registry.get_alias(alias) if alias is not None else None
                    raise SmokeTestFailed(
                        f"{target.served_name} failed {len(report.failed_cases)} of "
                        f"{len(report.cases)} smoke tests; it was unloaded and not promoted",
                        details={
                            "version": target.served_name,
                            "report": report.model_dump(),
                            "timings_ms": timings,
                            "alias": alias,
                            "alias_target": live.target if live else None,
                        },
                    )
            target = self._registry.update_version(target.id, state=VersionState.READY)

            flip: FlipResult | None = None
            if alias is not None:
                flip = await self._flip(
                    alias, target, kind=EventType.PROMOTE, started=time.perf_counter()
                )
                timings["flip_ms"] = flip.flip_ms
            timings["total_ms"] = _ms(started)
            log.info(
                "deploy finished",
                extra={
                    "adapter": adapter,
                    "version": version,
                    "alias": alias,
                    "outcome": "promoted" if flip else "ready",
                    **timings,
                },
            )
            return DeployResult(version=target, smoke=report, flip=flip, timings_ms=timings)
        finally:
            self._deploying.discard(target.id)
            self._release_reservation(target.id)

    async def register_and_deploy(
        self,
        adapter: str,
        uri: str,
        *,
        alias: str | None,
        smoke: SmokeSpec | None,
        base_model: str | None = None,
        sha256: str | None = None,
        notes: str | None = None,
    ) -> DeployResult:
        """Register the next version of ``adapter`` from ``uri`` and deploy it."""
        if alias is not None:
            validate_name(alias, kind="alias")
        created = self.register(adapter, uri, base_model=base_model, sha256=sha256, notes=notes)
        return await self.deploy(adapter, created.version, alias=alias, smoke=smoke)

    async def _run_smoke(self, version: Version, spec: SmokeSpec) -> SmokeReport:
        report = await self._smoke.run(spec, version.served_name)
        outcome = "pass" if report.passed else "fail"
        self._metrics.smoke_seconds.labels(adapter=version.adapter, outcome=outcome).observe(
            report.duration_ms / 1000
        )
        if report.passed:
            self._record(
                EventType.SMOKE_PASS,
                version,
                duration_ms=report.duration_ms,
                detail={"cases": len(report.cases)},
            )
        return report

    async def _reject(self, version: Version, report: SmokeReport) -> None:
        """Record a smoke failure and unload. No alias references it, so no drain."""
        with self._registry.transaction():
            self._registry.update_version(version.id, state=VersionState.SMOKE_FAILED)
            self._record(
                EventType.SMOKE_FAIL,
                version,
                duration_ms=report.duration_ms,
                detail={
                    "cases": len(report.cases),
                    "failed": [
                        {"name": c.name, "failures": c.failures} for c in report.failed_cases
                    ],
                },
            )
        await self._unload_now(version, reason="smoke_fail")

    # ================================================================== promote / rollback

    async def promote(
        self, alias: str, version: int, *, adapter: str | None = None, force: bool = False
    ) -> FlipResult:
        """Point ``alias`` at an existing version, loading it first if needed.

        ``adapter`` defaults to the adapter the alias is already bound to.
        """
        validate_name(alias, kind="alias")
        if adapter is None:
            existing = self._registry.get_alias(alias)
            if existing is None:
                raise ValidationFailed(f"alias {alias!r} is new: name the adapter to bind it to")
            adapter = existing.adapter
        target = self.get_version(adapter, version)
        self._check_alias_binding(alias, target)
        if not target.state.promotable and not force:
            raise ConflictError(
                f"{target.served_name} is {target.state.value}; deploy it (with smoke tests) "
                "first, or pass force=true",
                details={"state": target.state.value},
            )
        return await self._prepare_and_flip(alias, target, kind=EventType.PROMOTE)

    async def rollback(self, alias: str, *, to_version: int | None = None) -> FlipResult:
        """Point ``alias`` back at the previous good version (or ``to_version``)."""
        current = self.get_alias(alias)
        if to_version is not None:
            if to_version == current.version:
                raise ConflictError(f"{alias!r} already routes to {current.target}")
            target = self.get_version(current.adapter, to_version)
            if not target.state.promotable:
                raise ConflictError(
                    f"{target.served_name} is {target.state.value} and cannot receive traffic"
                )
        else:
            target = self._previous_good(current)
        return await self._prepare_and_flip(alias, target, kind=EventType.ROLLBACK)

    def _previous_good(self, current: Alias) -> Version:
        seen: set[int] = {current.version_id}
        for version_id in self._registry.alias_target_history(current.name):
            if version_id in seen:
                continue
            seen.add(version_id)
            candidate = self._registry.get_version_by_id(version_id)
            if candidate is not None and candidate.state is VersionState.READY:
                return candidate
        raise ConflictError(
            f"alias {current.name!r} has no earlier good version to roll back to",
            details={"current": current.target},
        )

    async def _prepare_and_flip(
        self, alias: str, target: Version, *, kind: EventType
    ) -> FlipResult:
        started = time.perf_counter()
        self._reserved[target.id] += 1
        try:
            reloaded = False
            if target.id in self._unloading:
                log.info(
                    "waiting for in-progress unload before reloading",
                    extra={"alias": alias, "target": target.served_name},
                )
                await self._unloading[target.id]
                target = self.get_version(target.adapter, target.version)
            if not target.loaded:
                target, _ = await self._load(target, event=EventType.RELOAD)
                reloaded = True
            result = await self._flip(alias, target, kind=kind, started=started)
            return FlipResult(
                alias=result.alias,
                from_version=result.from_version,
                to_version=result.to_version,
                reloaded=reloaded,
                flip_ms=result.flip_ms,
                total_ms=_ms(started),
            )
        finally:
            self._release_reservation(target.id)

    async def _flip(
        self, alias_name: str, target: Version, *, kind: EventType, started: float
    ) -> FlipResult:
        lock = self._adapter_locks[target.adapter_id]
        async with lock:
            flip_started = time.perf_counter()
            current = self._registry.get_alias(alias_name)
            if current is not None and current.version_id == target.id:
                log.info(
                    "alias already routes to target",
                    extra={"alias": alias_name, "target": target.served_name},
                )
                return FlipResult(
                    alias=current,
                    from_version=current.version,
                    to_version=target.version,
                    reloaded=False,
                    flip_ms=0.0,
                    total_ms=_ms(started),
                )
            with self._registry.transaction():
                updated = self._registry.set_alias(
                    alias_name,
                    target.adapter_id,
                    target.id,
                    expected_revision=current.revision if current else None,
                )
                if kind is EventType.ROLLBACK and current is not None:
                    self._registry.update_version(
                        current.version_id, state=VersionState.ROLLED_BACK
                    )
                # The event is written inside the flip transaction, so its duration
                # covers the whole operation (including any reload) up to the commit.
                self._record(
                    kind,
                    target,
                    alias=alias_name,
                    from_version_id=current.version_id if current else None,
                    duration_ms=_ms(started),
                    detail={"revision": updated.revision},
                )
            # Commit first, then route: if the commit fails, traffic never moved.
            self._router.set_route(alias_name, target.served_name)
            flip_ms = _ms(flip_started)
        self._metrics.flip_seconds.labels(alias=alias_name, kind=kind.value).observe(flip_ms / 1000)
        self._metrics.alias_version.labels(alias=alias_name, adapter=target.adapter).set(
            target.version
        )
        log.info(
            "alias flipped",
            extra={
                "alias": alias_name,
                "kind": kind.value,
                "from_version": current.version if current else None,
                "to_version": target.version,
                "flip_ms": flip_ms,
            },
        )
        if self._settings.auto_gc:
            self._spawn(self.gc(target.adapter), name=f"gc:{target.adapter}")
        return FlipResult(
            alias=updated,
            from_version=current.version if current else None,
            to_version=target.version,
            reloaded=False,
            flip_ms=flip_ms,
            total_ms=_ms(started),
        )

    def _check_alias_binding(self, alias: str, target: Version) -> None:
        existing = self._registry.get_alias(alias)
        if existing is not None and existing.adapter_id != target.adapter_id:
            raise ConflictError(
                f"alias {alias!r} belongs to adapter {existing.adapter!r}; "
                f"it cannot route to {target.served_name}"
            )

    # ================================================================== load / unload

    async def _load(self, version: Version, *, event: EventType) -> tuple[Version, float]:
        started = time.perf_counter()
        name = version.served_name
        try:
            await self._vllm.load_adapter(name, version.uri)
        except SwitchboardError as exc:
            elapsed = _ms(started)
            self._metrics.load_seconds.labels(adapter=version.adapter, outcome="fail").observe(
                elapsed / 1000
            )
            with self._registry.transaction():
                self._registry.update_version(
                    version.id, state=VersionState.LOAD_FAILED if event is EventType.LOAD else None
                )
                self._record(
                    EventType.LOAD_FAIL,
                    version,
                    duration_ms=elapsed,
                    detail={"error": exc.message, "during": event.value},
                )
            log.warning(
                "adapter load failed",
                extra={"target": name, "outcome": "fail", "load_ms": elapsed, "error": exc.message},
            )
            if isinstance(exc, UpstreamUnavailable):
                # A timeout may still complete server-side: clean up so vLLM matches the registry.
                await self._best_effort_unload(name)
            raise
        elapsed = _ms(started)
        self._metrics.load_seconds.labels(adapter=version.adapter, outcome="ok").observe(
            elapsed / 1000
        )
        with self._registry.transaction():
            updated = self._registry.update_version(version.id, loaded=True)
            self._record(event, updated, duration_ms=elapsed, detail={"uri": version.uri})
        log.info("adapter loaded", extra={"target": name, "outcome": "ok", "load_ms": elapsed})
        return updated, elapsed

    async def _best_effort_unload(self, name: str) -> None:
        try:
            await self._vllm.unload_adapter(name)
        except SwitchboardError as exc:
            log.warning(
                "cleanup unload failed; reconcile will retry",
                extra={"target": name, "error": exc.message},
            )

    async def _unload_now(self, version: Version, *, reason: str) -> None:
        """Unload a version that is known to carry no alias traffic."""
        started = time.perf_counter()
        try:
            await self._vllm.unload_adapter(version.served_name)
        except SwitchboardError as exc:
            self._record(
                EventType.UNLOAD_FAIL, version, detail={"error": exc.message, "reason": reason}
            )
            log.warning(
                "unload failed; version stays marked loaded until reconcile",
                extra={"target": version.served_name, "error": exc.message, "reason": reason},
            )
            return
        with self._registry.transaction():
            self._registry.update_version(version.id, loaded=False)
            self._record(
                EventType.UNLOAD, version, duration_ms=_ms(started), detail={"reason": reason}
            )
        log.info("adapter unloaded", extra={"target": version.served_name, "reason": reason})

    async def unload(
        self, adapter: str, version: int, *, drain: bool = True, reason: str = "manual"
    ) -> Version:
        """Drain and unload a version. Refuses versions an alias currently routes to."""
        target = self.get_version(adapter, version)
        self._ensure_unloadable(target)
        if target.id in self._unloading:
            await self._unloading[target.id]
            return self.get_version(adapter, version)
        if drain and target.loaded:
            await self._sleep(self._settings.drain_delay_s)

        # Re-check under the lock: a rollback may have moved an alias back during the drain.
        async with self._adapter_locks[target.adapter_id]:
            target = self.get_version(adapter, version)
            self._ensure_unloadable(target)
            in_progress = self._unloading.get(target.id)
            if in_progress is None:
                self._unloading[target.id] = asyncio.get_running_loop().create_future()
        if in_progress is not None:
            await in_progress
            return self.get_version(adapter, version)

        try:
            idle = await self._router.wait_idle(target.served_name, self._settings.drain_timeout_s)
            if not idle:
                log.warning(
                    "drain timeout; unloading with requests still in flight",
                    extra={
                        "target": target.served_name,
                        "inflight": self._router.inflight(target.served_name),
                    },
                )
            await self._unload_now(target, reason=reason)
        finally:
            future = self._unloading.pop(target.id)
            if not future.done():
                future.set_result(None)
        return self.get_version(adapter, version)

    def _ensure_unloadable(self, version: Version) -> None:
        if self._router.is_routed(version.served_name) or any(
            a.version_id == version.id for a in self._registry.list_aliases()
        ):
            raise ConflictError(
                f"{version.served_name} is the target of an alias; promote or roll back first"
            )
        if self._reserved[version.id]:
            raise ConflictError(f"{version.served_name} is being deployed or promoted")

    def _release_reservation(self, version_id: int) -> None:
        self._reserved[version_id] -= 1
        if self._reserved[version_id] <= 0:
            del self._reserved[version_id]

    # ================================================================== gc

    async def gc(self, adapter: str, *, keep: int | None = None) -> GcResult:
        """Unload loaded versions beyond the ``keep`` most useful ones per adapter.

        Alias targets are always kept and count toward ``keep``; the remaining
        slots go to the most recent rollback candidates (READY versions the
        aliases pointed at before), then to the newest READY versions.
        """
        keep = keep if keep is not None else self._settings.keep_loaded
        if keep < 1:
            raise ValidationFailed("keep must be at least 1")
        record = self.get_adapter(adapter)
        aliases = [a for a in self._registry.list_aliases() if a.adapter_id == record.id]
        targets = {a.version_id for a in aliases}
        loaded = {v.id: v for v in self._registry.list_versions(adapter) if v.loaded}

        ranked: list[int] = []
        for alias in aliases:
            ranked.extend(self._registry.alias_target_history(alias.name))
        ranked.extend(sorted(loaded, key=lambda vid: loaded[vid].version, reverse=True))
        rollback_slots = max(keep - len(targets), 0)

        kept: list[str] = [loaded[v].served_name for v in targets if v in loaded]
        candidates: list[Version] = []
        seen: set[int] = set(targets)
        for version_id in ranked:
            if version_id in seen or version_id not in loaded:
                continue
            seen.add(version_id)
            version = loaded[version_id]
            if version.state is VersionState.READY and rollback_slots > 0:
                rollback_slots -= 1
                kept.append(version.served_name)
            else:
                candidates.append(version)

        result = GcResult(kept=kept)
        for version in candidates:
            try:
                await self.unload(version.adapter, version.version, reason="gc")
                result.unloaded.append(version.served_name)
            except ConflictError as exc:
                result.skipped[version.served_name] = exc.message
        if result.unloaded or result.skipped:
            log.info(
                "gc finished",
                extra={
                    "adapter": adapter,
                    "keep": keep,
                    "unloaded": result.unloaded,
                    "skipped": len(result.skipped),
                },
            )
        return result

    # ================================================================== reconcile

    async def reconcile(self) -> ReconcileResult:
        """Make the registry and vLLM agree; re-load alias targets vLLM lost (e.g. restart)."""
        try:
            present = await self._vllm.loaded_model_ids()
        except SwitchboardError as exc:
            log.warning("reconcile skipped: vLLM unreachable", extra={"error": exc.message})
            return ReconcileResult(reachable=False)

        result = ReconcileResult(reachable=True)
        busy = self._deploying | set(self._unloading)
        known: set[str] = set()
        for version in self._registry.list_versions():
            known.add(version.served_name)
            if version.id in busy:
                continue
            in_vllm = version.served_name in present
            if version.loaded and not in_vllm:
                self._registry.update_version(version.id, loaded=False)
                result.marked_unloaded.append(version.served_name)
            elif not version.loaded and in_vllm:
                self._registry.update_version(version.id, loaded=True)
                result.marked_loaded.append(version.served_name)

        for alias in self._registry.list_aliases():
            if alias.target in present or alias.version_id in busy:
                continue
            version = self._registry.get_version_by_id(alias.version_id)
            assert version is not None
            try:
                await self._load(version, event=EventType.RELOAD)
                result.reloaded.append(alias.target)
            except SwitchboardError as exc:
                result.errors[alias.target] = exc.message
                log.error(
                    "alias target missing from vLLM and reload failed",
                    extra={"alias": alias.name, "target": alias.target, "error": exc.message},
                )

        result.unknown_in_vllm.extend(sorted(n for n in present if "@v" in n and n not in known))
        if result.marked_unloaded or result.marked_loaded or result.reloaded or result.errors:
            log.warning(
                "registry drift corrected",
                extra={
                    "reloaded": result.reloaded,
                    "marked_unloaded": result.marked_unloaded,
                    "marked_loaded": result.marked_loaded,
                    "errors": len(result.errors),
                },
            )
        return result

    async def reconcile_forever(self, interval_s: float) -> None:
        while True:
            await self._sleep(interval_s)
            try:
                await self.reconcile()
            except Exception:
                log.exception("reconcile loop iteration failed")

    # ================================================================== status

    async def status(self) -> StatusReport:
        """Aliases with live traffic, loaded versions, and drift against vLLM."""
        vllm_ids: set[str] | None
        try:
            vllm_ids = await self._vllm.loaded_model_ids()
            vllm_error = None
        except SwitchboardError as exc:
            vllm_ids, vllm_error = None, exc.message
        aliases = [
            AliasStatus(
                alias=a,
                inflight=self._router.inflight(a.target),
                in_vllm=None if vllm_ids is None else a.target in vllm_ids,
            )
            for a in self._registry.list_aliases()
        ]
        loaded = [v for v in self._registry.list_versions() if v.loaded]
        return StatusReport(
            vllm_url=self._vllm.base_url,
            vllm_reachable=vllm_ids is not None,
            vllm_error=vllm_error,
            vllm_models=sorted(vllm_ids or []),
            aliases=aliases,
            loaded_versions=loaded,
            unloading=[v.served_name for v in loaded if v.id in self._unloading],
            deploying=sorted(
                v.served_name for v in self._registry.list_versions() if v.id in self._deploying
            ),
        )

    # ================================================================== events

    def _record(
        self,
        type_: EventType,
        version: Version,
        *,
        alias: str | None = None,
        from_version_id: int | None = None,
        duration_ms: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._registry.append_event(
            type_,
            adapter_id=version.adapter_id,
            version_id=version.id,
            alias=alias,
            from_version_id=from_version_id,
            duration_ms=duration_ms,
            detail=detail,
        )
        self._metrics.rollout_events.labels(adapter=version.adapter, event=type_.value).inc()


def _is_hex(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
