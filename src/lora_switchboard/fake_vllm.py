"""A tiny stand-in for vLLM's OpenAI-compatible server, for tests and demos.

It implements just enough of the real API to exercise a rollout end to end:

* ``GET  /v1/models``                 base model + loaded adapters (``root``/``parent`` like vLLM)
* ``POST /v1/load_lora_adapter``      400 "has already been loaded" on duplicates, like vLLM
* ``POST /v1/unload_lora_adapter``    404 "cannot be found" for unknown names, like vLLM
* ``POST /v1/chat/completions``       canned replies, optional SSE streaming
* ``POST /v1/completions``

Behaviour is steered by the adapter *path*, so demos need no extra setup:

* contains ``broken``  -> the adapter "loads" but answers gibberish (fails smoke tests)
* contains ``missing`` -> load fails with 404, as when the path does not exist
* contains ``slow``    -> load takes ``slow_load_s`` seconds (exercise load timeouts)

Run it with ``lora-switchboard fake-vllm --port 8001``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class FakeState:
    base_model: str = DEFAULT_BASE_MODEL
    load_delay_s: float = 0.05
    slow_load_s: float = 30.0
    token_delay_s: float = 0.0
    adapters: dict[str, str] = field(default_factory=dict)
    load_calls: list[str] = field(default_factory=list)
    unload_calls: list[str] = field(default_factory=list)
    served: dict[str, int] = field(default_factory=dict)


def _error(status: int, message: str, type_: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "param": None, "code": status}},
    )


def _reply(state: FakeState, model: str, user_text: str) -> str:
    path = state.adapters.get(model, "")
    if "broken" in path:
        return "lorem ipsum dolor sit amet"
    if "json" in user_text.lower():
        return json.dumps({"intent": "refund", "confidence": 0.93})
    return f"Happy to help with that! (served by {model})"


def _last_user_text(body: dict[str, Any]) -> str:
    if "messages" in body:
        for message in reversed(body.get("messages") or []):
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else json.dumps(content)
        return ""
    prompt = body.get("prompt", "")
    return prompt if isinstance(prompt, str) else json.dumps(prompt)


def create_fake_vllm(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    load_delay_s: float = 0.05,
    slow_load_s: float = 30.0,
    token_delay_s: float = 0.0,
) -> FastAPI:
    app = FastAPI(title="fake-vllm")
    state = FakeState(
        base_model=base_model,
        load_delay_s=load_delay_s,
        slow_load_s=slow_load_s,
        token_delay_s=token_delay_s,
    )
    app.state.fake = state

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        now = int(time.time())
        data = [
            {
                "id": state.base_model,
                "object": "model",
                "created": now,
                "owned_by": "vllm",
                "root": state.base_model,
                "parent": None,
            }
        ]
        data += [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "vllm",
                "root": path,
                "parent": state.base_model,
            }
            for name, path in state.adapters.items()
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/load_lora_adapter", response_model=None)
    async def load(request: Request) -> PlainTextResponse | JSONResponse:
        body = await request.json()
        name, path = body.get("lora_name"), body.get("lora_path")
        if not name or not path:
            return _error(
                400, "Both 'lora_name' and 'lora_path' must be provided.", "BadRequestError"
            )
        if name in state.adapters and not body.get("load_inplace"):
            return _error(
                400,
                f"The lora adapter '{name}' has already been loaded. If you want to load the "
                "adapter in place, set 'load_inplace' to True.",
                "BadRequestError",
            )
        state.load_calls.append(name)
        await asyncio.sleep(state.slow_load_s if "slow" in path else state.load_delay_s)
        if "missing" in path:
            return _error(
                404, f"Loading lora {name} failed: No adapter found for {path}", "NotFoundError"
            )
        state.adapters[name] = path
        return PlainTextResponse(f"Success: LoRA adapter '{name}' added successfully.")

    @app.post("/v1/unload_lora_adapter", response_model=None)
    async def unload(request: Request) -> PlainTextResponse | JSONResponse:
        body = await request.json()
        name = body.get("lora_name")
        if not name:
            return _error(
                400, "'lora_name' needs to be provided to unload a LoRA adapter.", "BadRequestError"
            )
        if name not in state.adapters:
            return _error(404, f"The lora adapter '{name}' cannot be found.", "NotFoundError")
        del state.adapters[name]
        state.unload_calls.append(name)
        return PlainTextResponse(f"Success: LoRA adapter '{name}' removed successfully.")

    async def generate(request: Request, kind: str) -> Any:
        body = await request.json()
        model = body.get("model")
        if model != state.base_model and model not in state.adapters:
            return _error(404, f"The model `{model}` does not exist.", "NotFoundError")
        state.served[model] = state.served.get(model, 0) + 1
        text = _reply(state, model, _last_user_text(body))
        rid = f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if body.get("stream"):
            return StreamingResponse(
                _stream(state, kind, rid, created, model, text), media_type="text/event-stream"
            )
        usage = {
            "prompt_tokens": 8,
            "completion_tokens": len(text.split()),
            "total_tokens": 8 + len(text.split()),
        }
        if kind == "chat":
            choice: dict[str, Any] = {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
            obj = "chat.completion"
        else:
            choice = {"index": 0, "finish_reason": "stop", "text": text}
            obj = "text_completion"
        return {
            "id": rid,
            "object": obj,
            "created": created,
            "model": model,
            "choices": [choice],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        return await generate(request, "chat")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        return await generate(request, "text")

    return app


async def _stream(
    state: FakeState, kind: str, rid: str, created: int, model: str, text: str
) -> AsyncIterator[bytes]:
    obj = "chat.completion.chunk" if kind == "chat" else "text_completion"

    def chunk(delta: str | None, finish: str | None) -> bytes:
        if kind == "chat":
            choice: dict[str, Any] = {
                "index": 0,
                "finish_reason": finish,
                "delta": {} if delta is None else {"content": delta},
            }
        else:
            choice = {"index": 0, "finish_reason": finish, "text": delta or ""}
        payload = {
            "id": rid,
            "object": obj,
            "created": created,
            "model": model,
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    words = text.split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(state.token_delay_s)
        yield chunk(word if i == 0 else f" {word}", None)
    yield chunk(None, "stop")
    yield b"data: [DONE]\n\n"
