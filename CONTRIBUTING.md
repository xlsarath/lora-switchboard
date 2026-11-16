# Contributing

Thanks for helping make LoRA rollouts boring. Small, focused pull requests are
the easiest to review; for anything larger than a bug fix, please open an issue
first so we can agree on the approach.

## Development setup

```bash
git clone https://github.com/xlsarath/lora-switchboard
cd lora-switchboard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                       # ~10 s, no GPU needed
ruff check . && ruff format .
```

Run the whole stack locally without Docker:

```bash
lora-switchboard fake-vllm --port 8001 &
LSB_VLLM_URL=http://127.0.0.1:8001 LSB_DRAIN_DELAY_S=2 lora-switchboard serve &
LSB_CLI=lora-switchboard SMOKE=examples/smoke-support-bot.yaml ./examples/demo.sh
```

## Where things live

| Module | Responsibility |
| --- | --- |
| `registry/` | SQLite schema, migrations and the repository. The only place with SQL. No business rules. |
| `service.py` | Rollout orchestration: deploy, promote, rollback, unload, gc, reconcile. All invariants live here. |
| `router.py` | In-memory alias table and in-flight leases (drain). |
| `proxy.py` | OpenAI-compatible forwarding and SSE pass-through. |
| `smoke.py` | Smoke-suite schema, evaluation and runner. |
| `vllm.py` | Typed client for vLLM's HTTP API. |
| `api/` | FastAPI routers. Handlers translate HTTP to service calls and nothing else. |
| `cli.py` | Typer CLI; a client of the admin API. |
| `fake_vllm.py` | GPU-free vLLM stand-in for tests and demos. |

## Ground rules

- **Keep layers honest.** HTTP handlers stay thin, the repository only talks to
  the database, and rules belong in `RolloutService`.
- **Every failure path leaves consistent state.** If you add a step to a
  rollout, add a test for it failing (vLLM down, timeout, concurrent change).
- **Never log prompts or completions at INFO or above.** Log ids, names,
  counts, outcomes and timings using `extra={...}`.
- **Type hints everywhere**, and `ruff check` / `ruff format` must pass.
- **Tests use the fake vLLM**, never a network service. Use the `stack`
  fixture for service-level tests and `live` for real HTTP/streaming tests.
- Update `CHANGELOG.md` under *Unreleased* for user-facing changes.

## Commit messages

Use the imperative mood ("Add gc dry-run flag"), keep the subject under ~70
characters, and explain the *why* in the body when it is not obvious.

## Releasing (maintainers)

1. Bump `__version__` in `src/lora_switchboard/__init__.py` and move the
   *Unreleased* changelog entries under the new version.
2. Tag `vX.Y.Z` and push the tag. The release workflow runs the tests, pushes
   `ghcr.io/xlsarath/lora-switchboard:X.Y.Z` and creates the GitHub release.

## Code of conduct

Be kind, assume good intent, and keep discussions about the work.
