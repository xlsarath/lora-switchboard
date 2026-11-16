from __future__ import annotations

import pydantic
import pytest

from lora_switchboard.errors import ValidationFailed
from lora_switchboard.smoke import Expectation, SmokeSpec, evaluate


def test_contains_is_case_insensitive_by_default() -> None:
    assert evaluate(Expectation(contains=["HELP"]), "happy to help", 10, None) == []
    strict = Expectation(contains=["HELP"], case_sensitive=True)
    assert evaluate(strict, "happy to help", 10, None) == ["missing 'HELP'"]


def test_not_contains_and_regex() -> None:
    expect = Expectation(not_contains="as an AI", regex=r"order #\d+")
    assert evaluate(expect, "Your order #42 shipped", 1, None) == []
    failures = evaluate(expect, "As an AI I cannot", 1, None)
    assert failures == ["unexpected 'as an AI'", r"no match for /order #\d+/"]


def test_json_schema_accepts_fenced_json() -> None:
    expect = Expectation(json_schema={"type": "object", "required": ["intent"]})
    assert evaluate(expect, '```json\n{"intent": "refund"}\n```', 1, None) == []
    assert evaluate(expect, "{}", 1, None) == ["schema violation: 'intent' is a required property"]
    assert evaluate(expect, "not json", 1, None)[0].startswith("output is not JSON")


def test_latency_budget_case_overrides_default() -> None:
    assert evaluate(Expectation(), "x", 900, 1000) == []
    assert evaluate(Expectation(), "x", 1200, 1000) == ["latency 1200ms > 1000ms"]
    assert evaluate(Expectation(max_latency_ms=2000), "x", 1200, 1000) == []


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("tests: []", "at least 1 item"),
        ("tests: [{name: a}]", "exactly one of 'messages' or 'prompt'"),
        ("tests: [{name: a, prompt: x, expect: {regex: '('}}]", "invalid regex"),
        ("tests: [{name: a, prompt: x, expect: {json_schema: {type: 12}}}]", "invalid JSON schema"),
        ("tests: [{name: a, prompt: x}, {name: a, prompt: y}]", "duplicate test names: a"),
        ("tests: [{name: a, prompt: x, expect: {containz: [y]}}]", "Extra inputs"),
    ],
)
def test_invalid_specs_are_rejected(yaml_text: str, message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        SmokeSpec.from_yaml(yaml_text)


def test_non_mapping_yaml_is_rejected() -> None:
    with pytest.raises(ValidationFailed):
        SmokeSpec.from_yaml("- just a list")
    with pytest.raises(ValidationFailed):
        SmokeSpec.from_yaml("tests: [unclosed")


def test_example_suites_parse() -> None:
    from pathlib import Path

    for path in sorted(Path(__file__).parent.parent.joinpath("examples").glob("*.yaml")):
        if path.name.startswith("smoke"):
            assert SmokeSpec.from_file(path).tests
