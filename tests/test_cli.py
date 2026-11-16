from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from lora_switchboard.app import create_app
from lora_switchboard.cli import EXIT_SMOKE_FAILED, app
from lora_switchboard.fake_vllm import create_fake_vllm

from .conftest import BASE_MODEL, SMOKE_YAML, LiveStack, ServerThread, make_settings

runner = CliRunner()


@pytest.fixture
def smoke_file(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.yaml"
    path.write_text(SMOKE_YAML)
    return path


def cli(live: LiveStack, *args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["--url", live.url, *args])
    return result.exit_code, result.output


def test_full_rollout_through_the_cli(live: LiveStack, smoke_file: Path) -> None:
    code, out = cli(
        live,
        "deploy",
        "support-bot",
        "--path",
        "/adapters/support/v1",
        "--base-model",
        BASE_MODEL,
        "--alias",
        "support",
        "--smoke",
        str(smoke_file),
    )
    assert code == 0, out
    assert "smoke tests (2) ... pass" in out
    assert "support now routes to support-bot@v1" in out

    code, out = cli(
        live,
        "deploy",
        "support-bot",
        "--path",
        "/adapters/support/v2-broken",
        "--alias",
        "support",
        "--smoke",
        str(smoke_file),
    )
    assert code == EXIT_SMOKE_FAILED
    assert "FAIL" in out
    assert "missing 'help'" in out

    code, out = cli(
        live,
        "deploy",
        "support-bot",
        "--path",
        "/adapters/support/v3",
        "--alias",
        "support",
        "--smoke",
        str(smoke_file),
    )
    assert code == 0, out

    code, out = cli(live, "rollback", "support")
    assert code == 0, out
    assert "rollback support: v3 → v1" in out

    code, out = cli(live, "promote", "support", "support-bot@v3")
    assert code == 0, out

    code, out = cli(live, "status")
    assert code == 0
    assert "support-bot@v3" in out
    assert "reachable" in out

    code, out = cli(live, "history", "--alias", "support")
    assert code == 0
    assert "rollback" in out
    assert "(from v3)" in out

    code, out = cli(live, "versions", "support-bot")
    assert code == 0
    assert "smoke_failed" in out

    code, out = cli(live, "--json", "status")
    assert code == 0
    assert '"vllm_reachable": true' in out


def test_cli_argument_errors(live: LiveStack, smoke_file: Path) -> None:
    code, out = cli(live, "deploy", "support-bot", "--path", "/a")
    assert code == 1
    assert "--skip-smoke" in out
    code, out = cli(live, "deploy", "support-bot", "--smoke", str(smoke_file))
    assert code == 1
    assert "exactly one of --uri or --version" in out
    code, out = cli(live, "rollback", "nope")
    assert code == 1
    assert "does not exist" in out
    code, out = runner.invoke(app, ["--url", "http://127.0.0.1:9", "status"]).exit_code, ""
    assert code == 1


def test_admin_token_is_enforced(tmp_path: Path) -> None:
    with ServerThread(create_fake_vllm()) as vllm:
        settings = make_settings(tmp_path, vllm_url=vllm.url, admin_token="test-admin-token")
        with ServerThread(create_app(settings)) as sb:
            assert httpx.get(f"{sb.url}/admin/status").status_code == 401
            bad = httpx.get(f"{sb.url}/admin/status", headers={"Authorization": "Bearer nope"})
            assert bad.status_code == 401
            good = httpx.get(
                f"{sb.url}/admin/status", headers={"Authorization": "Bearer test-admin-token"}
            )
            assert good.status_code == 200
            # The inference proxy and probes are not behind the admin token.
            assert httpx.get(f"{sb.url}/healthz").status_code == 200
            assert httpx.get(f"{sb.url}/readyz").status_code == 200
            result = runner.invoke(app, ["--url", sb.url, "--token", "test-admin-token", "status"])
            assert result.exit_code == 0


def test_readyz_reports_vllm_down(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, vllm_url="http://127.0.0.1:9")
    with ServerThread(create_app(settings)) as sb:
        assert httpx.get(f"{sb.url}/healthz").status_code == 200
        assert httpx.get(f"{sb.url}/readyz").status_code == 503
        proxied = httpx.post(f"{sb.url}/v1/chat/completions", json={"model": "x", "messages": []})
        assert proxied.status_code == 502


@pytest.fixture(autouse=True)
def _no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
