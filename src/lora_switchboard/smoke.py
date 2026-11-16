"""Smoke tests: a small YAML suite every version must pass before it gets traffic.

Example::

    defaults:
      params: {temperature: 0, max_tokens: 64}
      max_latency_ms: 5000
    tests:
      - name: greets politely
        messages:
          - {role: user, content: "Hi, I need help with my order"}
        expect:
          contains: ["help"]
          not_contains: ["as an AI"]
      - name: emits valid JSON
        prompt: "Classify as JSON: 'I want a refund'"
        expect:
          json_schema:
            type: object
            required: [intent]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import SwitchboardError, ValidationFailed
from .vllm import VLLMClient

log = logging.getLogger(__name__)

_PREVIEW_CHARS = 200


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


class Expectation(BaseModel):
    """Checks applied to one response. All configured checks must pass."""

    model_config = ConfigDict(extra="forbid")

    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: str | None = None
    json_schema: dict[str, Any] | None = None
    max_latency_ms: float | None = Field(default=None, gt=0)
    case_sensitive: bool = False

    _as_list_contains = field_validator("contains", "not_contains", mode="before")(_as_list)

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regex {value!r}: {exc}") from exc
        return value

    @field_validator("json_schema")
    @classmethod
    def _valid_schema(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            try:
                jsonschema.Draft202012Validator.check_schema(value)
            except jsonschema.SchemaError as exc:
                raise ValueError(f"invalid JSON schema: {exc.message}") from exc
        return value


class SmokeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    expect: Expectation = Field(default_factory=Expectation)

    @model_validator(mode="after")
    def _one_input(self) -> SmokeCase:
        if (self.messages is None) == (self.prompt is None):
            raise ValueError(f"test {self.name!r}: set exactly one of 'messages' or 'prompt'")
        return self


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0, "max_tokens": 128})
    timeout_s: float = Field(default=60.0, gt=0)
    max_latency_ms: float | None = Field(default=None, gt=0)


class SmokeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: Defaults = Field(default_factory=Defaults)
    concurrency: int = Field(default=4, ge=1, le=64)
    tests: list[SmokeCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> SmokeSpec:
        names = [t.name for t in self.tests]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate test names: {', '.join(duplicates)}")
        return self

    @classmethod
    def from_yaml(cls, text: str) -> SmokeSpec:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValidationFailed(f"smoke file is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationFailed("smoke file must be a mapping with a 'tests' list")
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Path | str) -> SmokeSpec:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


class CaseResult(BaseModel):
    name: str
    passed: bool
    latency_ms: float | None
    failures: list[str]
    output_preview: str | None = None


class SmokeReport(BaseModel):
    target: str
    passed: bool
    duration_ms: float
    cases: list[CaseResult]

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]


def evaluate(
    expect: Expectation, text: str, latency_ms: float, default_max_ms: float | None
) -> list[str]:
    """Return the list of failed checks for one response (empty means pass)."""
    failures: list[str] = []
    haystack = text if expect.case_sensitive else text.lower()

    def norm(needle: str) -> str:
        return needle if expect.case_sensitive else needle.lower()

    failures.extend(f"missing {n!r}" for n in expect.contains if norm(n) not in haystack)
    failures.extend(f"unexpected {n!r}" for n in expect.not_contains if norm(n) in haystack)
    if expect.regex is not None:
        flags = 0 if expect.case_sensitive else re.IGNORECASE
        if not re.search(expect.regex, text, flags | re.DOTALL):
            failures.append(f"no match for /{expect.regex}/")
    if expect.json_schema is not None:
        failures.extend(_check_json(text, expect.json_schema))
    limit = expect.max_latency_ms if expect.max_latency_ms is not None else default_max_ms
    if limit is not None and latency_ms > limit:
        failures.append(f"latency {latency_ms:.0f}ms > {limit:.0f}ms")
    return failures


def _check_json(text: str, schema: dict[str, Any]) -> list[str]:
    candidate = text.strip()
    # Tolerate a fenced code block, which instruction-tuned models emit often.
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        document = json.loads(candidate)
    except ValueError as exc:
        return [f"output is not JSON ({exc.msg})"]
    error = jsonschema.exceptions.best_match(
        jsonschema.Draft202012Validator(schema).iter_errors(document)
    )
    return [] if error is None else [f"schema violation: {error.message}"]


class SmokeRunner:
    """Runs a :class:`SmokeSpec` against one served adapter name."""

    def __init__(self, client: VLLMClient) -> None:
        self._client = client

    async def run(self, spec: SmokeSpec, target: str) -> SmokeReport:
        started = time.perf_counter()
        gate = asyncio.Semaphore(spec.concurrency)

        async def run_case(case: SmokeCase) -> CaseResult:
            async with gate:
                return await self._run_case(spec.defaults, case, target)

        cases = await asyncio.gather(*(run_case(c) for c in spec.tests))
        report = SmokeReport(
            target=target,
            passed=all(c.passed for c in cases),
            duration_ms=(time.perf_counter() - started) * 1000,
            cases=list(cases),
        )
        log.log(
            logging.INFO if report.passed else logging.WARNING,
            "smoke suite finished",
            extra={
                "target": target,
                "outcome": "pass" if report.passed else "fail",
                "cases": len(cases),
                "failed": len(report.failed_cases),
                "duration_ms": round(report.duration_ms, 1),
            },
        )
        return report

    async def _run_case(self, defaults: Defaults, case: SmokeCase, target: str) -> CaseResult:
        try:
            result = await self._client.complete(
                target,
                messages=case.messages,
                prompt=case.prompt,
                params={**defaults.params, **case.params},
                timeout_s=defaults.timeout_s,
            )
        except SwitchboardError as exc:
            log.warning(
                "smoke case errored",
                extra={"target": target, "case": case.name, "error": exc.message},
            )
            return CaseResult(name=case.name, passed=False, latency_ms=None, failures=[exc.message])
        failures = evaluate(case.expect, result.text, result.latency_ms, defaults.max_latency_ms)
        log.debug(
            "smoke case evaluated",
            extra={"target": target, "case": case.name, "failures": len(failures)},
        )
        return CaseResult(
            name=case.name,
            passed=not failures,
            latency_ms=round(result.latency_ms, 1),
            failures=failures,
            output_preview=result.text[:_PREVIEW_CHARS] if failures else None,
        )
