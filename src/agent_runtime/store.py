"""SQLite-backed append-only event store plus the idempotency ledger.

The events table has no UPDATE or DELETE path in this codebase. Appends
are serialized per run by a UNIQUE(run_id, seq) constraint, so two
writers racing on the same run cannot both claim a sequence number.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .events import Event, ALL_TYPES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    run_id  TEXT    NOT NULL,
    seq     INTEGER NOT NULL,
    type    TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    payload TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (run_id, type);
CREATE TABLE IF NOT EXISTS executions (
    idem_key TEXT PRIMARY KEY,
    run_id   TEXT NOT NULL,
    result   TEXT NOT NULL,
    ts       REAL NOT NULL
);
"""


class EventStore:
    """Append-only store. One instance per database file."""

    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self.clock = clock
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- events ----------------------------------------------------------

    def new_run_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def append(
        self,
        run_id: str,
        type: str,
        payload: dict[str, Any],
        ts: float | None = None,
    ) -> Event:
        if type not in ALL_TYPES:
            raise ValueError(f"unknown event type: {type}")
        if ts is None:
            ts = self.clock()
        for _ in range(5):  # retry on a lost seq race
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE run_id = ?",
                (run_id,),
            )
            seq = cur.fetchone()[0]
            try:
                self._conn.execute(
                    "INSERT INTO events (run_id, seq, type, ts, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, seq, type, ts, json.dumps(payload, sort_keys=True)),
                )
                self._conn.commit()
                return Event(run_id=run_id, seq=seq, type=type, ts=ts, payload=payload)
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError(f"could not append event for run {run_id}")

    def events(self, run_id: str, until: int | None = None) -> list[Event]:
        sql = "SELECT run_id, seq, type, ts, payload FROM events WHERE run_id = ?"
        args: list[Any] = [run_id]
        if until is not None:
            sql += " AND seq <= ?"
            args.append(until)
        sql += " ORDER BY seq"
        rows = self._conn.execute(sql, args).fetchall()
        return [
            Event(run_id=r[0], seq=r[1], type=r[2], ts=r[3], payload=json.loads(r[4]))
            for r in rows
        ]

    def run_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT run_id, MIN(ts) t FROM events GROUP BY run_id ORDER BY t"
        ).fetchall()
        return [r[0] for r in rows]

    # -- idempotency ledger -----------------------------------------------

    def claim_execution(self, idem_key: str, run_id: str, result: dict[str, Any]) -> None:
        """Record a completed tool execution. First write wins."""
        self._conn.execute(
            "INSERT OR IGNORE INTO executions (idem_key, run_id, result, ts) "
            "VALUES (?, ?, ?, ?)",
            (idem_key, run_id, json.dumps(result, sort_keys=True), self.clock()),
        )
        self._conn.commit()

    def get_execution(self, idem_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT result FROM executions WHERE idem_key = ?", (idem_key,)
        ).fetchone()
        return json.loads(row[0]) if row else None
