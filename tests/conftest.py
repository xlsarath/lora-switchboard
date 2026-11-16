from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from lora_switchboard.app import create_app
from lora_switchboard.config import Settings
from lora_switchboard.fake_vllm import FakeState, create_fake_vllm
from lora_switchboard.metrics import Metrics
from lora_switchboard.registry import Registry
from lora_switchboard.router import TrafficRouter
from lora_switchboard.service import RolloutService
from lora_switchboard.smoke import SmokeSpec
from lora_switchboard.vllm import VLLMClient

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SMOKE_YAML = """
defaults:
  params: {temperature: 0, max_tokens: 32}
  max_latency_ms: 5000
tests:
  - name: helpful reply
    messages: [{role: user, content: "Where is my order?"}]
    expect:
      contains: ["help"]
      not_contains: ["lorem"]
  - name: structured output
    prompt: "Return JSON for: I want a refund"
    expect:
      json_schema:
        type: object
        required: [intent, confidence]
        properties:
          intent: {type: string}
          confidence: {type: number, minimum: 0, maximum: 1}
"""


@pytest.fixture
def smoke_spec() -> SmokeSpec:
    return SmokeSpec.from_yaml(SMOKE_YAML)


# ---------------------------------------------------------------------- in-process stack


@dataclass
class Stack:
    service: RolloutService
    registry: Registry
    router: TrafficRouter
    vllm: VLLMClient
    fake: FakeState
    settings: Settings


@pytest.fixture
def fake_app() -> FastAPI:
    return create_fake_vllm(base_model=BASE_MODEL, load_delay_s=0.0, slow_load_s=5.0)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "db_path": tmp_path / "registry.db",
        "drain_delay_s": 0.0,
        "drain_timeout_s": 5.0,
        "reconcile_interval_s": 0,
        "load_timeout_s": 2.0,
        **overrides,
    }
    return Settings(**values)


@pytest.fixture
async def stack(tmp_path: Path, fake_app: FastAPI) -> AsyncIterator[Stack]:
    settings = make_settings(tmp_path, auto_gc=False)
    registry = Registry(settings.db_path)
    vllm = VLLMClient(
        "http://fake-vllm",
        load_timeout_s=settings.load_timeout_s,
        transport=httpx.ASGITransport(app=fake_app),
    )
    metrics = Metrics()
    router = TrafficRouter(metrics)
    service = RolloutService(settings, registry, vllm, router, metrics)
    yield Stack(service, registry, router, vllm, fake_app.state.fake, settings)
    await service.shutdown()
    await vllm.aclose()
    registry.close()


# ---------------------------------------------------------------------- real servers


class ServerThread:
    """Run an ASGI app under uvicorn on a free port in a background thread."""

    def __init__(self, app: FastAPI) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_config=None,
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *_: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@dataclass
class LiveStack:
    url: str
    vllm_url: str
    fake: FakeState
    app: FastAPI


@pytest.fixture
def live(tmp_path: Path) -> Iterator[LiveStack]:
    """Fake vLLM + switchboard, both real HTTP servers (needed for true streaming)."""
    fake = create_fake_vllm(base_model=BASE_MODEL, load_delay_s=0.0, token_delay_s=0.005)
    with ServerThread(fake) as vllm_server:
        settings = make_settings(
            tmp_path, vllm_url=vllm_server.url, drain_delay_s=0.2, keep_loaded=1
        )
        app = create_app(settings)
        with ServerThread(app) as sb_server:
            yield LiveStack(sb_server.url, vllm_server.url, fake.state.fake, app)
