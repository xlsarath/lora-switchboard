# Good first issues

Five well-scoped starter issues, ready to be filed. Each lists the files to
touch and what "done" looks like. Suggested labels: `good first issue` plus the
one shown.

---

## 1. Add `--dry-run` to `gc`

**Label:** `enhancement`

`lora-switchboard gc support-bot --keep 2` unloads versions straight away.
Operators want to see what *would* be unloaded first.

**Scope**

- Add `dry_run: bool = False` to `GcRequest` (`src/lora_switchboard/api/schemas.py`)
  and to `RolloutService.gc` (`src/lora_switchboard/service.py`). In dry-run
  mode, compute `kept` / `unloaded` exactly as today but skip the `unload` calls
  and return the would-be list.
- Add `--dry-run` to the `gc` CLI command and print `would unload: ...`.

**Done when**

- A test in `tests/test_rollout.py` shows that dry-run returns the same lists
  as a real run and leaves `fake.adapters` untouched.
- README CLI reference mentions the flag.

---

## 2. `contains_any` expectation for smoke tests

**Label:** `smoke-tests`

`contains` requires *all* strings. A common need is "at least one of these"
(e.g. `["refund", "money back"]`).

**Scope**

- Add `contains_any: list[str]` to `Expectation` in `src/lora_switchboard/smoke.py`
  (accept a single string like `contains` does), and evaluate it in
  `evaluate()` honouring `case_sensitive`.
- Failure message: `none of ['refund', 'money back'] found`.

**Done when**

- Unit tests in `tests/test_smoke.py` cover pass, fail and case sensitivity.
- The smoke-test YAML reference table in the README lists it.

---

## 3. Show the registered URI and notes in `history`

**Label:** `cli`

`register` events store `{"uri": ...}` in `detail`, and versions carry
`notes`, but `lora-switchboard history` shows neither, which makes the lineage
less useful.

**Scope**

- In `src/lora_switchboard/cli.py`, extend `_short_detail()` so `register`
  events show the URI (shortened to the last 40 characters with a leading `…`).
- Store `notes` in the `register` event detail in `RolloutService.register`
  and show it when present.

**Done when**

- `tests/test_cli.py` asserts that the URI appears in `history` output.

---

## 4. `LSB_DB_PATH` parent directory is created automatically

**Label:** `bug`

Starting with `LSB_DB_PATH=/tmp/new-dir/switchboard.db` fails with
`sqlite3.OperationalError: unable to open database file` when `new-dir` does
not exist.

**Scope**

- In `Registry.__init__` (`src/lora_switchboard/registry/repository.py`),
  create the parent directory (`mkdir(parents=True, exist_ok=True)`) unless the
  path is `:memory:`.
- Log the resolved database path once at startup in `app.py`.

**Done when**

- A test in `tests/test_registry.py` opens a registry under a nested,
  non-existent directory.

---

## 5. Per-alias request counter in `status`

**Label:** `observability`

`status` shows in-flight requests, but not how much traffic an alias has
served since startup, which is handy right after a promote.

**Scope**

- Keep a `Counter[str]` of completed requests per alias in `TrafficRouter`
  (`src/lora_switchboard/router.py`), incremented in `Lease.release()` when the
  lease has an alias.
- Expose it on `AliasStatus` (`served_since_start: int`) and add a column to the
  `status` table in the CLI.

**Done when**

- `tests/test_proxy.py` sends N requests through an alias and asserts the
  count in `GET /admin/status`.
- Prometheus metrics are unchanged (they already count requests; this is a
  convenience view).
