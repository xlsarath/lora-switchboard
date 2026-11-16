"""End-to-end tests over real HTTP: fake vLLM <- switchboard <- client."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from .conftest import BASE_MODEL, SMOKE_YAML, LiveStack


def deploy(client: httpx.Client, uri: str, alias: str = "support") -> dict[str, Any]:
    response = client.post(
        "/admin/deploy",
        json={
            "adapter": "support-bot",
            "uri": uri,
            "alias": alias,
            "base_model": BASE_MODEL,
            "smoke_yaml": SMOKE_YAML,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def chat(model: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "Where is my order?"}],
    }


def sse_events(raw: str) -> list[str]:
    return [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]


def test_alias_is_rewritten_for_plain_requests(live: LiveStack) -> None:
    with httpx.Client(base_url=live.url) as client:
        deploy(client, "/adapters/support/v1")
        response = client.post("/v1/chat/completions", json=chat("support"))
        assert response.status_code == 200
        assert response.headers["x-switchboard-target"] == "support-bot@v1"
        body = response.json()
        assert body["model"] == "support-bot@v1"
        assert "served by support-bot@v1" in body["choices"][0]["message"]["content"]

        completion = client.post("/v1/completions", json={"model": "support", "prompt": "hi"})
        assert completion.json()["choices"][0]["text"].startswith("Happy to help")


def test_streaming_passthrough(live: LiveStack) -> None:
    with httpx.Client(base_url=live.url) as client:
        deploy(client, "/adapters/support/v1")
        with client.stream("POST", "/v1/chat/completions", json=chat("support", stream=True)) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            raw = "".join(r.iter_text())
    events = sse_events(raw)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Happy to help with that! (served by support-bot@v1)"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_models_lists_aliases_and_passthrough_names(live: LiveStack) -> None:
    with httpx.Client(base_url=live.url) as client:
        deploy(client, "/adapters/support/v1")
        ids = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        assert ids["support"]["root"] == "support-bot@v1"
        assert ids["support"]["parent"] == BASE_MODEL
        assert {BASE_MODEL, "support-bot@v1"} <= set(ids)

        # Base model and pinned versions are forwarded untouched.
        pinned = client.post("/v1/chat/completions", json=chat("support-bot@v1"))
        assert pinned.headers["x-switchboard-target"] == "support-bot@v1"
        assert client.post("/v1/chat/completions", json=chat(BASE_MODEL)).status_code == 200
        unknown = client.post("/v1/chat/completions", json=chat("nope"))
        assert unknown.status_code == 404
        assert "does not exist" in unknown.json()["error"]["message"]


def test_proxy_guards(live: LiveStack) -> None:
    with httpx.Client(base_url=live.url) as client:
        blocked = client.post("/v1/load_lora_adapter", json={"lora_name": "x", "lora_path": "/x"})
        assert blocked.status_code == 404
        assert client.post("/v1/chat/completions", content=b"{oops").status_code == 400
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
        assert client.post("/v1/chat/completions", json=[1, 2]).status_code == 400
    assert live.fake.load_calls == []


def test_admin_api_errors_are_structured(live: LiveStack) -> None:
    with httpx.Client(base_url=live.url) as client:
        deploy(client, "/adapters/support/v1")
        failed = client.post(
            "/admin/deploy",
            json={
                "adapter": "support-bot",
                "uri": "/adapters/v2-broken",
                "alias": "support",
                "smoke_yaml": SMOKE_YAML,
            },
        )
        assert failed.status_code == 422
        error = failed.json()["error"]
        assert error["code"] == "smoke_failed"
        assert not error["details"]["report"]["passed"]

        bad_yaml = client.post(
            "/admin/deploy",
            json={"adapter": "support-bot", "version": 1, "smoke_yaml": "tests: [{name: x}]"},
        )
        assert bad_yaml.status_code == 422

        no_smoke = client.post("/admin/deploy", json={"adapter": "support-bot", "version": 1})
        assert no_smoke.status_code == 422
        assert client.post("/admin/aliases/nope/rollback").status_code == 404
        assert client.post("/admin/aliases/support/rollback").status_code == 409

        history = client.get("/admin/history", params={"alias": "support"}).json()
        assert [e["type"] for e in history] == ["promote"]
        metrics = client.get("/metrics").text
        assert 'lsb_rollout_events_total{adapter="support-bot",event="smoke_fail"} 1.0' in metrics
        assert "lsb_alias_flip_seconds_bucket" in metrics


@pytest.mark.parametrize("stream", [True, False])
async def test_zero_failed_requests_during_flips(live: LiveStack, stream: bool) -> None:
    """Hammer the alias while flipping it back and forth; nothing may fail."""
    # keep_loaded=1 would unload the idle version after its drain delay; keep both loaded
    # so this test measures pure flips (reloads are covered elsewhere).
    live.app.state.settings.auto_gc = False
    async with httpx.AsyncClient(base_url=live.url, timeout=30) as client:
        for n in (1, 2):
            r = await client.post(
                "/admin/deploy",
                json={
                    "adapter": "support-bot",
                    "uri": f"/adapters/support/v{n}",
                    "alias": "support",
                    "base_model": BASE_MODEL,
                    "smoke_yaml": SMOKE_YAML,
                },
            )
            assert r.status_code == 200, r.text
        await client.post("/admin/aliases/support/promote", json={"version": 1})
        await client.post("/admin/aliases/support/promote", json={"version": 2})

        seen: dict[str, int] = {}
        failures: list[str] = []

        async def one_request() -> None:
            try:
                if stream:
                    async with client.stream(
                        "POST", "/v1/chat/completions", json=chat("support", stream=True)
                    ) as r:
                        raw = "".join([c async for c in r.aiter_text()])
                        ok = r.status_code == 200 and sse_events(raw)[-1] == "[DONE]"
                else:
                    r = await client.post("/v1/chat/completions", json=chat("support"))
                    ok = r.status_code == 200
                target = r.headers.get("x-switchboard-target", "?")
                seen[target] = seen.get(target, 0) + 1
                if not ok:
                    failures.append(f"{r.status_code} {target}")
            except httpx.HTTPError as exc:
                failures.append(repr(exc))

        async def flipper() -> None:
            for i in range(20):
                path = "rollback" if i % 2 == 0 else "promote"
                body = {} if path == "rollback" else {"version": 2}
                r = await client.post(f"/admin/aliases/support/{path}", json=body)
                assert r.status_code == 200, r.text
                await asyncio.sleep(0.01)

        await asyncio.gather(flipper(), *(one_request() for _ in range(200)))

    assert failures == []
    assert sum(seen.values()) == 200
    assert set(seen) <= {"support-bot@v1", "support-bot@v2"}


async def test_old_version_unloads_only_after_streams_drain(live: LiveStack) -> None:
    async with httpx.AsyncClient(base_url=live.url, timeout=30) as client:
        r = await client.post(
            "/admin/deploy",
            json={
                "adapter": "support-bot",
                "uri": "/adapters/support/v1",
                "alias": "support",
                "base_model": BASE_MODEL,
                "smoke_yaml": SMOKE_YAML,
            },
        )
        assert r.status_code == 200
        live.fake.token_delay_s = 0.1  # ~1s streams

        async def long_stream() -> tuple[int, str]:
            async with client.stream(
                "POST", "/v1/chat/completions", json=chat("support", stream=True)
            ) as resp:
                raw = "".join([c async for c in resp.aiter_text()])
                return resp.status_code, raw

        streams = [asyncio.create_task(long_stream()) for _ in range(5)]
        await asyncio.sleep(0.15)  # streams have started on v1
        r = await client.post(
            "/admin/deploy",
            json={
                "adapter": "support-bot",
                "uri": "/adapters/support/v2",
                "alias": "support",
                "smoke_yaml": SMOKE_YAML,
            },
        )
        assert r.status_code == 200
        await asyncio.sleep(0.4)  # past the 0.2s drain delay, streams still running
        assert "support-bot@v1" in live.fake.adapters
        status = (await client.get("/admin/status")).json()
        assert status["unloading"] == ["support-bot@v1"]

        results = await asyncio.gather(*streams)
        assert all(
            code == 200
            and "served by support-bot@v1"
            in "".join(
                json.loads(e)["choices"][0]["delta"].get("content", "")
                for e in sse_events(raw)[:-1]
            )
            for code, raw in results
        )

        for _ in range(100):
            if "support-bot@v1" not in live.fake.adapters:
                break
            await asyncio.sleep(0.02)
        assert sorted(live.fake.adapters) == ["support-bot@v2"]
        history = (await client.get("/admin/history", params={"adapter": "support-bot"})).json()
        assert history[0]["type"] == "unload"
        assert history[0]["detail"] == {"reason": "gc"}
