"""Command-line interface.

``serve`` and ``fake-vllm`` run servers; every other command is a thin client
of the admin API, so the running service stays the single source of truth for
routing state.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any, NoReturn

import httpx
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .domain import parse_ref

app = typer.Typer(
    name="lora-switchboard",
    help="Safe LoRA adapter rollouts for vLLM: deploy, smoke test, promote, roll back.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
out = Console(highlight=False)
err = Console(stderr=True, highlight=False)

EXIT_ERROR = 1
EXIT_SMOKE_FAILED = 2


class _Ctx:
    def __init__(self, url: str, token: str | None, as_json: bool) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.as_json = as_json

    def client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return httpx.Client(
            base_url=self.url, headers=headers, timeout=httpx.Timeout(900.0, connect=5.0)
        )

    def call(
        self, method: str, path: str, body: dict[str, Any] | None = None, **params: Any
    ) -> Any:
        query = {k: v for k, v in params.items() if v is not None}
        try:
            with self.client() as client:
                response = client.request(method, path, json=body, params=query)
        except httpx.TransportError as exc:
            _fail(f"cannot reach switchboard at {self.url} ({exc.__class__.__name__})")
        if response.status_code >= 400:
            _api_error(response, self.as_json)
        return response.json()


def _fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    err.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


def _api_error(response: httpx.Response, as_json: bool) -> NoReturn:
    try:
        payload = response.json()
    except ValueError:
        _fail(f"HTTP {response.status_code}: {response.text[:300]}")
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if as_json:
        print(json.dumps(payload, indent=2))
    if error.get("code") == "smoke_failed":
        details = error.get("details", {})
        if not as_json:
            load_ms = details.get("timings_ms", {}).get("load_ms")
            out.print(f"→ loading {details.get('version')} ... [green]ok[/] ({_ms(load_ms)})")
            _print_smoke(details.get("report", {}))
            out.print(f"→ unloaded {details.get('version')}; not promoted")
            if details.get("alias_target"):
                out.print(
                    f"[yellow]✘[/] [bold]{details['alias']}[/] still routes to "
                    f"[bold]{details['alias_target']}[/]"
                )
        _fail(error.get("message", "smoke tests failed"), EXIT_SMOKE_FAILED)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    _fail(error.get("message") or json.dumps(detail or payload))


def _ctx(ctx: typer.Context) -> _Ctx:
    state: _Ctx = ctx.obj
    return state


def _emit_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 10:
        return f"{value:.2f} ms"
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def _print_smoke(report: dict[str, Any]) -> None:
    cases = report.get("cases", [])
    passed = report.get("passed")
    verdict = "[green]pass[/]" if passed else "[red]FAIL[/]"
    out.print(f"→ smoke tests ({len(cases)}) ... {verdict} ({_ms(report.get('duration_ms'))})")
    for case in cases:
        mark = "[green]✓[/]" if case["passed"] else "[red]✗[/]"
        out.print(f"    {mark} {case['name']:<28} {_ms(case.get('latency_ms'))}")
        for failure in case.get("failures", []):
            out.print(f"        [red]- {failure}[/]")
        if case.get("output_preview"):
            out.print(f"        [dim]got: {case['output_preview']!r}[/]")


def _print_flip(flip: dict[str, Any], verb: str) -> None:
    alias = flip["alias"]
    before = f"v{flip['from_version']}" if flip.get("from_version") else "(new)"
    reload_note = " [yellow](reloaded)[/]" if flip.get("reloaded") else ""
    out.print(
        f"→ {verb} {alias['name']}: {before} → v{flip['to_version']} ... [green]ok[/] "
        f"(flip {_ms(flip['flip_ms'])}){reload_note}"
    )


def _version_ref(ref: str, alias: str | None = None) -> tuple[str | None, int]:
    """Accept ``name@vN``, ``vN`` or ``N``; the adapter may come from the alias."""
    bare = ref[1:] if ref.startswith("v") and ref[1:].isdigit() else ref
    if bare.isdigit():
        return None, int(bare)
    name, version = parse_ref(ref)
    if version is None:
        _fail(f"expected a version like 'v7' or '{name}@v7', got {ref!r}")
    return name, version


# ---------------------------------------------------------------------- root


@app.callback()
def root(
    ctx: typer.Context,
    url: Annotated[
        str, typer.Option(envvar="LSB_URL", help="Switchboard base URL.")
    ] = "http://127.0.0.1:8000",
    token: Annotated[
        str | None,
        typer.Option(envvar="LSB_ADMIN_TOKEN", help="Admin bearer token.", show_default=False),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print raw JSON responses.")] = False,
) -> None:
    ctx.obj = _Ctx(url, token, as_json)


@app.command()
def version() -> None:
    """Print the client version."""
    out.print(__version__)


# ---------------------------------------------------------------------- servers


@app.command()
def serve(
    host: Annotated[str, typer.Option(envvar="LSB_HOST")] = "127.0.0.1",
    port: Annotated[int, typer.Option(envvar="LSB_PORT")] = 8000,
) -> None:
    """Run the switchboard (admin API + OpenAI-compatible proxy). Configure with LSB_* env vars."""
    import uvicorn

    from .app import create_app
    from .config import Settings
    from .logs import configure_logging

    settings = Settings()
    configure_logging(settings.log_level, json_lines=settings.log_json)
    uvicorn.run(create_app(settings), host=host, port=port, log_config=None, access_log=False)


@app.command("fake-vllm")
def fake_vllm(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8001,
    base_model: Annotated[str, typer.Option(help="Model id to report as the base model.")] = (
        "Qwen/Qwen2.5-7B-Instruct"
    ),
    load_delay: Annotated[float, typer.Option(help="Seconds a load takes.")] = 0.4,
    token_delay: Annotated[float, typer.Option(help="Seconds per streamed word.")] = 0.02,
) -> None:
    """Run a fake vLLM server (no GPU) for demos and tests."""
    import uvicorn

    from .fake_vllm import create_fake_vllm
    from .logs import configure_logging

    configure_logging("INFO")
    uvicorn.run(
        create_fake_vllm(base_model=base_model, load_delay_s=load_delay, token_delay_s=token_delay),
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )


# ---------------------------------------------------------------------- rollout


@app.command()
def register(
    ctx: typer.Context,
    adapter: Annotated[str, typer.Argument(help="Adapter name, e.g. support-bot.")],
    uri: Annotated[
        str, typer.Option("--uri", "--path", help="Path/URI vLLM loads the adapter from.")
    ],
    base_model: Annotated[str | None, typer.Option(help="Required the first time.")] = None,
    sha256: Annotated[str | None, typer.Option()] = None,
    notes: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Register a new version without loading it."""
    c = _ctx(ctx)
    body = {"uri": uri, "base_model": base_model, "sha256": sha256, "notes": notes}
    version = c.call("POST", f"/admin/adapters/{adapter}/versions", body)
    if c.as_json:
        return _emit_json(version)
    out.print(f"registered [bold]{version['served_name']}[/] ({version['uri']})")
    return None


@app.command()
def deploy(
    ctx: typer.Context,
    adapter: Annotated[str, typer.Argument(help="Adapter name, e.g. support-bot.")],
    uri: Annotated[
        str | None,
        typer.Option("--uri", "--path", help="Register the next version from this path."),
    ] = None,
    version_: Annotated[
        int | None, typer.Option("--version", "-v", help="Deploy an already registered version.")
    ] = None,
    alias: Annotated[str | None, typer.Option(help="Alias to promote on success.")] = None,
    smoke: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="Smoke-test YAML."),
    ] = None,
    skip_smoke: Annotated[bool, typer.Option(help="Deploy without smoke tests.")] = False,
    base_model: Annotated[str | None, typer.Option()] = None,
    notes: Annotated[str | None, typer.Option()] = None,
    sha256: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Load a version, smoke test it and (with --alias) promote it."""
    c = _ctx(ctx)
    if (uri is None) == (version_ is None):
        _fail("pass exactly one of --uri or --version")
    if smoke is None and not skip_smoke:
        _fail("pass --smoke FILE, or --skip-smoke to deploy without tests")
    body: dict[str, Any] = {
        "adapter": adapter,
        "uri": uri,
        "version": version_,
        "alias": alias,
        "base_model": base_model,
        "notes": notes,
        "sha256": sha256,
        "skip_smoke": skip_smoke,
    }
    if smoke is not None:
        body["smoke_yaml"] = smoke.read_text(encoding="utf-8")
    result = c.call("POST", "/admin/deploy", {k: v for k, v in body.items() if v is not None})
    if c.as_json:
        return _emit_json(result)
    target = result["version"]["served_name"]
    timings = result["timings_ms"]
    if not timings:
        out.print(f"[green]✔[/] {target} is already live behind {alias}; nothing to do")
        return None
    out.print(f"→ loading {target} ... [green]ok[/] ({_ms(timings.get('load_ms'))})")
    if result.get("smoke"):
        _print_smoke(result["smoke"])
    else:
        out.print("→ smoke tests ... [yellow]skipped[/]")
    if result.get("flip"):
        _print_flip(result["flip"], "promote")
        out.print(
            f"[green]✔[/] [bold]{alias}[/] now routes to [bold]{target}[/] "
            f"(total {_ms(timings.get('total_ms'))})"
        )
    else:
        out.print(f"[green]✔[/] {target} is loaded and ready (no alias given)")
    return None


@app.command()
def promote(
    ctx: typer.Context,
    alias: Annotated[str, typer.Argument()],
    ref: Annotated[str, typer.Argument(help="Version: 'support-bot@v7', 'v7' or '7'.")],
    force: Annotated[
        bool, typer.Option(help="Allow versions that never passed smoke tests.")
    ] = False,
) -> None:
    """Point an alias at a version (loads it if needed)."""
    c = _ctx(ctx)
    adapter, number = _version_ref(ref)
    flip = c.call(
        "POST",
        f"/admin/aliases/{alias}/promote",
        {"version": number, "adapter": adapter, "force": force},
    )
    if c.as_json:
        return _emit_json(flip)
    _print_flip(flip, "promote")
    return None


@app.command()
def rollback(
    ctx: typer.Context,
    alias: Annotated[str, typer.Argument()],
    to: Annotated[
        str | None, typer.Option(help="Specific version, e.g. v5 (default: previous good).")
    ] = None,
) -> None:
    """Point an alias back at its previous good version."""
    c = _ctx(ctx)
    body = {"to_version": _version_ref(to)[1]} if to else {}
    started = time.perf_counter()
    flip = c.call("POST", f"/admin/aliases/{alias}/rollback", body)
    round_trip_ms = (time.perf_counter() - started) * 1000
    if c.as_json:
        return _emit_json(flip)
    _print_flip(flip, "rollback")
    out.print(
        f"[green]✔[/] [bold]{alias}[/] now routes to [bold]{flip['alias']['target']}[/] "
        f"(server {_ms(flip['total_ms'])}, round trip {_ms(round_trip_ms)})"
    )
    return None


@app.command()
def unload(
    ctx: typer.Context,
    ref: Annotated[str, typer.Argument(help="Version to unload, e.g. support-bot@v5.")],
    drain: Annotated[
        bool, typer.Option(help="Wait for the drain delay and in-flight requests.")
    ] = True,
) -> None:
    """Unload a version that no alias routes to."""
    c = _ctx(ctx)
    adapter, number = _version_ref(ref)
    if adapter is None:
        _fail("give the full name, e.g. support-bot@v5")
    version = c.call(
        "POST", f"/admin/adapters/{adapter}/versions/{number}/unload", {"drain": drain}
    )
    if c.as_json:
        return _emit_json(version)
    out.print(f"unloaded {version['served_name']}")
    return None


@app.command()
def gc(
    ctx: typer.Context,
    adapter: Annotated[str, typer.Argument()],
    keep: Annotated[
        int | None, typer.Option(min=1, help="Versions to keep loaded (default: server's).")
    ] = None,
) -> None:
    """Unload versions beyond the newest rollback candidates."""
    c = _ctx(ctx)
    result = c.call("POST", f"/admin/adapters/{adapter}/gc", {"keep": keep} if keep else {})
    if c.as_json:
        return _emit_json(result)
    out.print(f"kept: {', '.join(result['kept']) or '-'}")
    out.print(f"unloaded: {', '.join(result['unloaded']) or '-'}")
    for name, reason in result["skipped"].items():
        out.print(f"[yellow]skipped {name}: {reason}[/]")
    return None


@app.command()
def reconcile(ctx: typer.Context) -> None:
    """Re-sync the registry with what vLLM actually has loaded."""
    c = _ctx(ctx)
    _emit_json(c.call("POST", "/admin/reconcile"))


# ---------------------------------------------------------------------- inspection


@app.command()
def status(ctx: typer.Context) -> None:
    """Show aliases, their live targets, in-flight requests and loaded versions."""
    c = _ctx(ctx)
    report = c.call("GET", "/admin/status")
    if c.as_json:
        return _emit_json(report)
    reach = (
        "[green]reachable[/]"
        if report["vllm_reachable"]
        else f"[red]down[/] ({report['vllm_error']})"
    )
    out.print(f"vLLM {report['vllm_url']}: {reach}")
    table = Table("alias", "target", "in-flight", "in vLLM", "updated", box=None, pad_edge=False)
    for item in report["aliases"]:
        alias = item["alias"]
        in_vllm = {True: "[green]yes[/]", False: "[red]NO[/]", None: "?"}[item["in_vllm"]]
        table.add_row(
            alias["name"], alias["target"], str(item["inflight"]), in_vllm, alias["updated_at"][:19]
        )
    out.print(table if report["aliases"] else "no aliases yet")
    loaded = ", ".join(v["served_name"] for v in report["loaded_versions"]) or "-"
    out.print(f"loaded versions: {loaded}")
    if report["unloading"]:
        out.print(f"draining/unloading: {', '.join(report['unloading'])}")
    return None


@app.command()
def versions(ctx: typer.Context, adapter: Annotated[str | None, typer.Argument()] = None) -> None:
    """List registered versions."""
    c = _ctx(ctx)
    rows = c.call("GET", "/admin/versions", adapter=adapter)
    if c.as_json:
        return _emit_json(rows)
    table = Table("version", "state", "loaded", "uri", "notes", box=None, pad_edge=False)
    for v in rows:
        table.add_row(
            v["served_name"], v["state"], "yes" if v["loaded"] else "no", v["uri"], v["notes"] or ""
        )
    out.print(table)
    return None


@app.command()
def history(
    ctx: typer.Context,
    adapter: Annotated[str | None, typer.Option()] = None,
    alias: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 30,
) -> None:
    """Show the append-only rollout event log, newest first."""
    c = _ctx(ctx)
    events = c.call("GET", "/admin/history", adapter=adapter, alias=alias, limit=limit)
    if c.as_json:
        return _emit_json(events)
    table = Table(box=None, pad_edge=False)
    for column in ("time (UTC)", "event", "version", "alias", "duration"):
        table.add_column(column, no_wrap=True)
    table.add_column("detail", overflow="ellipsis", no_wrap=True)
    for e in events:
        version = f"{e['adapter']}@v{e['version']}" if e["version"] else e["adapter"]
        alias_col = e["alias"] or ""
        if e["from_version"]:
            alias_col += f" (from v{e['from_version']})"
        table.add_row(
            e["at"][5:19].replace("T", " "),
            e["type"],
            version,
            alias_col,
            _ms(e["duration_ms"]),
            _short_detail(e["detail"]),
        )
    out.print(table)
    return None


def _short_detail(detail: dict[str, Any]) -> str:
    if "error" in detail:
        return str(detail["error"])[:60]
    if "failed" in detail:
        return f"{len(detail['failed'])}/{detail.get('cases', '?')} cases failed"
    if "reason" in detail:
        return f"reason={detail['reason']}"
    if "cases" in detail:
        return f"{detail['cases']} cases"
    return ""


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
