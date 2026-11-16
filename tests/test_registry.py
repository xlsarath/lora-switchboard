from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lora_switchboard.domain import EventType, VersionState
from lora_switchboard.errors import ConflictError
from lora_switchboard.registry import Registry

BASE = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "r.db")


def test_versions_are_numbered_and_unique(registry: Registry) -> None:
    adapter = registry.create_adapter("sql-helper", BASE, None)
    assert registry.next_version_number(adapter.id) == 1
    v1 = registry.create_version(adapter.id, 1, "/adapters/sql/v1", None, "first cut")
    assert v1.served_name == "sql-helper@v1"
    assert v1.state is VersionState.REGISTERED
    assert not v1.loaded
    assert registry.next_version_number(adapter.id) == 2
    with pytest.raises(ConflictError):
        registry.create_version(adapter.id, 1, "/elsewhere", None, None)


def test_duplicate_adapter_is_a_conflict(registry: Registry) -> None:
    registry.create_adapter("sql-helper", BASE, None)
    with pytest.raises(ConflictError):
        registry.create_adapter("sql-helper", BASE, None)


def test_alias_compare_and_swap(registry: Registry) -> None:
    adapter = registry.create_adapter("support-bot", BASE, None)
    v1 = registry.create_version(adapter.id, 1, "/a/1", None, None)
    v2 = registry.create_version(adapter.id, 2, "/a/2", None, None)

    alias = registry.set_alias("support", adapter.id, v1.id, expected_revision=None)
    assert (alias.target, alias.revision) == ("support-bot@v1", 1)

    with pytest.raises(ConflictError):  # someone else created it first
        registry.set_alias("support", adapter.id, v2.id, expected_revision=None)

    alias = registry.set_alias("support", adapter.id, v2.id, expected_revision=1)
    assert (alias.version, alias.revision) == (2, 2)

    with pytest.raises(ConflictError):  # stale revision: lost update prevented
        registry.set_alias("support", adapter.id, v1.id, expected_revision=1)
    assert registry.get_alias("support").version == 2  # type: ignore[union-attr]


def test_failed_transaction_rolls_back(registry: Registry) -> None:
    adapter = registry.create_adapter("support-bot", BASE, None)

    def failing_change() -> None:
        with registry.transaction():
            registry.create_version(adapter.id, 1, "/a/1", None, None)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        failing_change()
    assert registry.list_versions("support-bot") == []


def test_events_are_append_only(registry: Registry, tmp_path: Path) -> None:
    adapter = registry.create_adapter("support-bot", BASE, None)
    v1 = registry.create_version(adapter.id, 1, "/a/1", None, None)
    registry.append_event(
        EventType.LOAD, adapter_id=adapter.id, version_id=v1.id, duration_ms=12.3456
    )
    events = registry.list_events(adapter="support-bot")
    assert [e.type for e in events] == [EventType.LOAD]
    assert events[0].duration_ms == 12.346

    raw = sqlite3.connect(tmp_path / "r.db")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        raw.execute("UPDATE events SET type = 'promote'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        raw.execute("DELETE FROM events")
    raw.close()


def test_alias_target_history_is_newest_first(registry: Registry) -> None:
    adapter = registry.create_adapter("support-bot", BASE, None)
    ids = [registry.create_version(adapter.id, n, f"/a/{n}", None, None).id for n in (1, 2, 3)]
    for vid in ids:
        registry.append_event(EventType.PROMOTE, adapter_id=adapter.id, version_id=vid, alias="s")
    registry.append_event(EventType.LOAD, adapter_id=adapter.id, version_id=ids[0], alias="s")
    assert registry.alias_target_history("s") == list(reversed(ids))


def test_schema_survives_reopen(tmp_path: Path) -> None:
    first = Registry(tmp_path / "r.db")
    first.create_adapter("support-bot", BASE, "customer support tone")
    first.close()
    again = Registry(tmp_path / "r.db")
    assert again.get_adapter("support-bot").description == "customer support tone"  # type: ignore[union-attr]
