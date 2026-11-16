"""SQLite schema and forward-only migrations, tracked with ``PRAGMA user_version``."""

from __future__ import annotations

MIGRATIONS: tuple[str, ...] = (
    # 1: initial schema
    """
    CREATE TABLE adapters (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL UNIQUE,
        base_model  TEXT    NOT NULL,
        description TEXT,
        created_at  TEXT    NOT NULL
    );

    CREATE TABLE versions (
        id          INTEGER PRIMARY KEY,
        adapter_id  INTEGER NOT NULL REFERENCES adapters(id),
        version     INTEGER NOT NULL CHECK (version > 0),
        uri         TEXT    NOT NULL,
        sha256      TEXT,
        notes       TEXT,
        state       TEXT    NOT NULL,
        loaded      INTEGER NOT NULL DEFAULT 0 CHECK (loaded IN (0, 1)),
        created_at  TEXT    NOT NULL,
        UNIQUE (adapter_id, version)
    );

    CREATE TABLE aliases (
        name        TEXT    PRIMARY KEY,
        adapter_id  INTEGER NOT NULL REFERENCES adapters(id),
        version_id  INTEGER NOT NULL REFERENCES versions(id),
        revision    INTEGER NOT NULL DEFAULT 1,
        updated_at  TEXT    NOT NULL
    );

    CREATE TABLE events (
        id              INTEGER PRIMARY KEY,
        at              TEXT    NOT NULL,
        type            TEXT    NOT NULL,
        adapter_id      INTEGER NOT NULL REFERENCES adapters(id),
        version_id      INTEGER REFERENCES versions(id),
        alias           TEXT,
        from_version_id INTEGER REFERENCES versions(id),
        duration_ms     REAL,
        detail          TEXT    NOT NULL DEFAULT '{}'
    );

    CREATE INDEX events_by_adapter ON events (adapter_id, id);
    CREATE INDEX events_by_alias   ON events (alias, id);

    -- The event log is the lineage record: make it append-only.
    CREATE TRIGGER events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
    CREATE TRIGGER events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
    """,
)
