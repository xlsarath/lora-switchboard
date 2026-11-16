"""In-memory routing table and in-flight request tracking.

The router is what makes flips safe:

* ``acquire`` resolves an alias **and** takes a lease on the concrete target in
  one synchronous step (no ``await`` in between), so on a single event loop a
  request can never observe a half-applied flip.
* A request keeps the target it resolved for its whole lifetime, even if the
  alias moves while it is streaming.
* Unloading waits on :meth:`wait_idle`, so a version is removed from vLLM
  only once every lease on it has been released.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field

from .metrics import Metrics

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Lease:
    """A claim on one concrete target, released exactly once."""

    model: str
    alias: str | None
    target: str
    _router: TrafficRouter = field(repr=False)
    _released: bool = field(default=False, repr=False)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._router._release(self.target)


class TrafficRouter:
    def __init__(self, metrics: Metrics | None = None) -> None:
        self._routes: dict[str, str] = {}
        self._inflight: Counter[str] = Counter()
        self._idle: dict[str, asyncio.Event] = {}
        self._metrics = metrics

    # ------------------------------------------------------------------ routes

    def replace_routes(self, routes: dict[str, str]) -> None:
        self._routes = dict(routes)

    def set_route(self, alias: str, target: str) -> None:
        previous = self._routes.get(alias)
        self._routes[alias] = target
        log.info("route updated", extra={"alias": alias, "target": target, "previous": previous})

    def routes(self) -> dict[str, str]:
        return dict(self._routes)

    def is_routed(self, target: str) -> bool:
        return target in self._routes.values()

    def resolve(self, model: str) -> str:
        return self._routes.get(model, model)

    # ------------------------------------------------------------------ leases

    def acquire(self, model: str) -> Lease:
        """Resolve ``model`` (alias or pass-through name) and lease the target."""
        alias = model if model in self._routes else None
        target = self._routes.get(model, model)
        self._inflight[target] += 1
        if self._metrics is not None:
            self._metrics.inflight.labels(target=target).inc()
        return Lease(model=model, alias=alias, target=target, _router=self)

    def _release(self, target: str) -> None:
        self._inflight[target] -= 1
        if self._metrics is not None:
            self._metrics.inflight.labels(target=target).dec()
        if self._inflight[target] <= 0:
            del self._inflight[target]
            waiter = self._idle.pop(target, None)
            if waiter is not None:
                waiter.set()

    def inflight(self, target: str) -> int:
        return self._inflight.get(target, 0)

    async def wait_idle(self, target: str, timeout_s: float) -> bool:
        """Wait until ``target`` has no leases. Returns ``False`` on timeout."""
        if self.inflight(target) == 0:
            return True
        waiter = self._idle.setdefault(target, asyncio.Event())
        try:
            await asyncio.wait_for(waiter.wait(), timeout=timeout_s)
        except TimeoutError:
            return self.inflight(target) == 0
        return True
