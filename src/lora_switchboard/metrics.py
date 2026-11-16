"""Prometheus metrics. Each app instance owns its own registry (keeps tests isolated)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_FLIP_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5)


class Metrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "lsb_proxy_requests_total",
            "Proxied inference requests.",
            ["alias", "target", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "lsb_proxy_request_seconds",
            "Time until the upstream response completed (streams included).",
            ["alias"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inflight = Gauge(
            "lsb_proxy_inflight_requests",
            "Requests currently in flight per served adapter.",
            ["target"],
            registry=self.registry,
        )
        self.rollout_events = Counter(
            "lsb_rollout_events_total",
            "Rollout events recorded in the registry.",
            ["adapter", "event"],
            registry=self.registry,
        )
        self.smoke_seconds = Histogram(
            "lsb_smoke_duration_seconds",
            "Wall time of a smoke-test suite.",
            ["adapter", "outcome"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.load_seconds = Histogram(
            "lsb_adapter_load_seconds",
            "Time vLLM took to load an adapter.",
            ["adapter", "outcome"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.flip_seconds = Histogram(
            "lsb_alias_flip_seconds",
            "Time to atomically repoint an alias (promote or rollback).",
            ["alias", "kind"],
            buckets=_FLIP_BUCKETS,
            registry=self.registry,
        )
        self.alias_version = Gauge(
            "lsb_alias_version",
            "Version number an alias currently routes to.",
            ["alias", "adapter"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
