"""Persistence for adapters, versions, aliases and the event log.

This is the only module that speaks SQL. It holds no business rules: it reads
and writes rows, enforces integrity through constraints, and exposes a
compare-and-swap primitive for alias updates so concurrent promotes (even from
separate processes sharing the file) cannot silently overwrite each other.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain import Adapter, Alias, Event, EventType, Version, VersionState
from ..errors import ConflictError
from .schema import MIGRATIONS

log = logging.getLogger(__name__)

_VERSION_COLUMNS = """
    v.id, v.adapter_id, a.name AS adapter, v.version, v.uri, a.base_model, v.sha256,
    v.notes, v.state, v.loaded, v.created_at
"""
_ALIAS_COLUMNS = """
    al.name, al.adapter_id, a.name AS adapter, al.version_id, v.version, al.revision,
    al.updated_at
"""
_EVENT_COLUMNS = """
    e.id, e.at, e.type, a.name AS adapter, v.version AS version, e.alias,
    fv.version AS from_version, e.duration_ms, e.detail
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Registry:
    """Thread-safe repository over a single SQLite connection.

    Reads and writes are short, so they run inline; a re-entrant lock
    serialises access to the connection. Multi-statement changes go through
    :meth:`transaction`, which takes SQLite's write lock up front
    (``BEGIN IMMEDIATE``) so the read-check-write sequence is atomic.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._depth = 0
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False, timeout=10.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    # ------------------------------------------------------------------ plumbing

    def _migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for index, script in enumerate(MIGRATIONS[current:], start=current + 1):
                log.info("applying registry migration", extra={"migration": index})
                self._conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {index}; COMMIT;")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the enclosed calls atomically. Nested use joins the outer transaction."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._depth = 0

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # ------------------------------------------------------------------ adapters

    def create_adapter(self, name: str, base_model: str, description: str | None) -> Adapter:
        try:
            cur = self._execute(
                "INSERT INTO adapters (name, base_model, description, created_at)"
                " VALUES (?, ?, ?, ?)",
                (name, base_model, description, _now()),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"adapter {name!r} already exists") from exc
        adapter = self.get_adapter_by_id(int(cur.lastrowid or 0))
        assert adapter is not None
        return adapter

    def get_adapter(self, name: str) -> Adapter | None:
        rows = self._query("SELECT * FROM adapters WHERE name = ?", (name,))
        return _adapter(rows[0]) if rows else None

    def get_adapter_by_id(self, adapter_id: int) -> Adapter | None:
        rows = self._query("SELECT * FROM adapters WHERE id = ?", (adapter_id,))
        return _adapter(rows[0]) if rows else None

    def list_adapters(self) -> list[Adapter]:
        return [_adapter(r) for r in self._query("SELECT * FROM adapters ORDER BY name")]

    # ------------------------------------------------------------------ versions

    def next_version_number(self, adapter_id: int) -> int:
        rows = self._query(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM versions WHERE adapter_id = ?",
            (adapter_id,),
        )
        return int(rows[0][0])

    def create_version(
        self,
        adapter_id: int,
        version: int,
        uri: str,
        sha256: str | None,
        notes: str | None,
    ) -> Version:
        try:
            cur = self._execute(
                "INSERT INTO versions (adapter_id, version, uri, sha256, notes, state, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (adapter_id, version, uri, sha256, notes, VersionState.REGISTERED.value, _now()),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"version v{version} already exists for this adapter") from exc
        created = self.get_version_by_id(int(cur.lastrowid or 0))
        assert created is not None
        return created

    def get_version(self, adapter: str, version: int) -> Version | None:
        rows = self._query(
            f"SELECT {_VERSION_COLUMNS} FROM versions v JOIN adapters a ON a.id = v.adapter_id"
            " WHERE a.name = ? AND v.version = ?",
            (adapter, version),
        )
        return _version(rows[0]) if rows else None

    def get_version_by_id(self, version_id: int) -> Version | None:
        rows = self._query(
            f"SELECT {_VERSION_COLUMNS} FROM versions v JOIN adapters a ON a.id = v.adapter_id"
            " WHERE v.id = ?",
            (version_id,),
        )
        return _version(rows[0]) if rows else None

    def list_versions(self, adapter: str | None = None) -> list[Version]:
        sql = f"SELECT {_VERSION_COLUMNS} FROM versions v JOIN adapters a ON a.id = v.adapter_id"
        params: tuple[Any, ...] = ()
        if adapter is not None:
            sql += " WHERE a.name = ?"
            params = (adapter,)
        sql += " ORDER BY a.name, v.version"
        return [_version(r) for r in self._query(sql, params)]

    def update_version(
        self, version_id: int, *, state: VersionState | None = None, loaded: bool | None = None
    ) -> Version:
        with self.transaction():
            if state is not None:
                self._execute(
                    "UPDATE versions SET state = ? WHERE id = ?", (state.value, version_id)
                )
            if loaded is not None:
                self._execute(
                    "UPDATE versions SET loaded = ? WHERE id = ?", (int(loaded), version_id)
                )
            updated = self.get_version_by_id(version_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------ aliases

    def get_alias(self, name: str) -> Alias | None:
        rows = self._query(
            f"SELECT {_ALIAS_COLUMNS} FROM aliases al JOIN adapters a ON a.id = al.adapter_id"
            " JOIN versions v ON v.id = al.version_id WHERE al.name = ?",
            (name,),
        )
        return _alias(rows[0]) if rows else None

    def list_aliases(self) -> list[Alias]:
        rows = self._query(
            f"SELECT {_ALIAS_COLUMNS} FROM aliases al JOIN adapters a ON a.id = al.adapter_id"
            " JOIN versions v ON v.id = al.version_id ORDER BY al.name"
        )
        return [_alias(r) for r in rows]

    def set_alias(
        self, name: str, adapter_id: int, version_id: int, *, expected_revision: int | None
    ) -> Alias:
        """Point ``name`` at ``version_id`` if the alias is still at ``expected_revision``.

        ``expected_revision=None`` means the caller believes the alias does not
        exist yet. Raises :class:`ConflictError` if another writer got there first.
        """
        now = _now()
        with self.transaction():
            if expected_revision is None:
                try:
                    self._execute(
                        "INSERT INTO aliases (name, adapter_id, version_id, revision, updated_at)"
                        " VALUES (?, ?, ?, 1, ?)",
                        (name, adapter_id, version_id, now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ConflictError(f"alias {name!r} was created concurrently") from exc
            else:
                cur = self._execute(
                    "UPDATE aliases SET version_id = ?, revision = revision + 1, updated_at = ?"
                    " WHERE name = ? AND adapter_id = ? AND revision = ?",
                    (version_id, now, name, adapter_id, expected_revision),
                )
                if cur.rowcount != 1:
                    raise ConflictError(
                        f"alias {name!r} changed concurrently; re-read and retry",
                        details={"expected_revision": expected_revision},
                    )
            alias = self.get_alias(name)
        assert alias is not None
        return alias

    # ------------------------------------------------------------------ events

    def append_event(
        self,
        type_: EventType,
        *,
        adapter_id: int,
        version_id: int | None = None,
        alias: str | None = None,
        from_version_id: int | None = None,
        duration_ms: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            "INSERT INTO events (at, type, adapter_id, version_id, alias, from_version_id,"
            " duration_ms, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                type_.value,
                adapter_id,
                version_id,
                alias,
                from_version_id,
                None if duration_ms is None else round(duration_ms, 3),
                json.dumps(detail or {}, sort_keys=True, default=str),
            ),
        )

    def list_events(
        self, *, adapter: str | None = None, alias: str | None = None, limit: int = 100
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[Any] = []
        if adapter is not None:
            clauses.append("a.name = ?")
            params.append(adapter)
        if alias is not None:
            clauses.append("e.alias = ?")
            params.append(alias)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT {_EVENT_COLUMNS} FROM events e JOIN adapters a ON a.id = e.adapter_id"
            " LEFT JOIN versions v ON v.id = e.version_id"
            " LEFT JOIN versions fv ON fv.id = e.from_version_id"
            f" {where} ORDER BY e.id DESC LIMIT ?",
            (*params, limit),
        )
        return [_event(r) for r in rows]

    def alias_target_history(self, alias: str) -> list[int]:
        """Version ids the alias has pointed at, most recent first (duplicates kept)."""
        rows = self._query(
            "SELECT version_id FROM events WHERE alias = ? AND type IN (?, ?) ORDER BY id DESC",
            (alias, EventType.PROMOTE.value, EventType.ROLLBACK.value),
        )
        return [int(r[0]) for r in rows]


# ---------------------------------------------------------------------- row mappers


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _adapter(row: sqlite3.Row) -> Adapter:
    return Adapter(
        id=row["id"],
        name=row["name"],
        base_model=row["base_model"],
        description=row["description"],
        created_at=_ts(row["created_at"]),
    )


def _version(row: sqlite3.Row) -> Version:
    return Version(
        id=row["id"],
        adapter_id=row["adapter_id"],
        adapter=row["adapter"],
        version=row["version"],
        uri=row["uri"],
        base_model=row["base_model"],
        sha256=row["sha256"],
        notes=row["notes"],
        state=VersionState(row["state"]),
        loaded=bool(row["loaded"]),
        created_at=_ts(row["created_at"]),
    )


def _alias(row: sqlite3.Row) -> Alias:
    return Alias(
        name=row["name"],
        adapter_id=row["adapter_id"],
        adapter=row["adapter"],
        version_id=row["version_id"],
        version=row["version"],
        revision=row["revision"],
        updated_at=_ts(row["updated_at"]),
    )


def _event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        at=_ts(row["at"]),
        type=EventType(row["type"]),
        adapter=row["adapter"],
        version=row["version"],
        alias=row["alias"],
        from_version=row["from_version"],
        duration_ms=row["duration_ms"],
        detail=json.loads(row["detail"]),
    )
