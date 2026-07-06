"""Built-in example tools.

All file paths are relative to the per-run workspace sandbox. Tools that
change anything are flagged side_effect=True and therefore stop at the
approval gate before executing.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from email.message import EmailMessage
from typing import Any
from urllib.parse import urlparse

from .registry import ToolContext, ToolError, ToolRegistry, ToolSpec

# -- filesystem ------------------------------------------------------------


def _read_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    path = ctx.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"no such file: {args['path']}")
    text = path.read_text(encoding="utf-8")
    return {"path": args["path"], "content": text[:20000], "bytes": len(text)}


def _write_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    path = ctx.resolve(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return {"path": args["path"], "bytes": len(args["content"])}


def _list_files(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    root = ctx.resolve(args.get("path", "."))
    if not root.is_dir():
        raise ToolError(f"not a directory: {args.get('path', '.')}")
    names = sorted(
        str(p.relative_to(ctx.workspace.resolve()))
        for p in root.rglob("*") if p.is_file()
    )
    return {"files": names[:500]}


# -- HTTP GET (allowlisted) --------------------------------------------------


def _http_get(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    url = args["url"]
    host = urlparse(url).hostname or ""
    if host not in ctx.http_allowlist:
        raise ToolError(f"host not in allowlist: {host!r}")
    if ctx.http_client is None:
        import httpx

        client = httpx.Client(timeout=10, follow_redirects=True)
        try:
            resp = client.get(url)
        finally:
            client.close()
    else:
        resp = ctx.http_client.get(url)
    return {"url": url, "status": resp.status_code, "body": resp.text[:20000]}


# -- CSV / sheet updater ------------------------------------------------------


def _load_csv(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [list(r) for r in csv.reader(fh)]


def _csv_update(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    path = ctx.resolve(args["path"])
    rows = _load_csv(path) if path.is_file() else []
    op = args["op"]
    if op == "append_row":
        rows.append([str(v) for v in args["row"]])
    elif op == "set_cell":
        r, c = args["row_index"], args["col_index"]
        if r >= len(rows) or c >= len(rows[r]):
            raise ToolError(f"cell out of range: ({r},{c})")
        rows[r][c] = str(args["value"])
    else:  # pragma: no cover - schema enum blocks this
        raise ToolError(f"unknown op: {op}")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return {"path": args["path"], "rows": len(rows), "op": op}


# -- email drafter (writes .eml, never sends) ---------------------------------


def _draft_email(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    msg = EmailMessage()
    msg["To"] = args["to"]
    msg["From"] = args.get("sender", "agent@localhost")
    msg["Subject"] = args["subject"]
    msg.set_content(args["body"])
    out_dir = ctx.resolve("outbox")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in args["subject"])[:60]
    path = out_dir / f"{safe or 'draft'}.eml"
    path.write_bytes(bytes(msg))
    rel = str(path.relative_to(ctx.workspace.resolve()))
    return {"draft_path": rel, "to": args["to"], "subject": args["subject"]}


# -- calendar store (SQLite) ---------------------------------------------------


def _calendar_conn(ctx: ToolContext) -> sqlite3.Connection:
    conn = sqlite3.connect(ctx.resolve("calendar.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY, title TEXT, starts_at TEXT, notes TEXT)"
    )
    return conn


def _calendar_add(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    conn = _calendar_conn(ctx)
    try:
        cur = conn.execute(
            "INSERT INTO events (title, starts_at, notes) VALUES (?, ?, ?)",
            (args["title"], args["starts_at"], args.get("notes", "")),
        )
        conn.commit()
        return {"event_id": cur.lastrowid, "title": args["title"],
                "starts_at": args["starts_at"]}
    finally:
        conn.close()


def _calendar_list(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    conn = _calendar_conn(ctx)
    try:
        rows = conn.execute(
            "SELECT id, title, starts_at, notes FROM events ORDER BY starts_at"
        ).fetchall()
        return {"events": [
            {"id": r[0], "title": r[1], "starts_at": r[2], "notes": r[3]}
            for r in rows
        ]}
    finally:
        conn.close()


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the run workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False},
        handler=_read_file,
    ))
    reg.register(ToolSpec(
        name="write_file",
        description="Write a UTF-8 text file inside the run workspace.",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"], "additionalProperties": False},
        handler=_write_file,
        side_effect=True,
    ))
    reg.register(ToolSpec(
        name="list_files",
        description="List files under a workspace directory, recursively.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "additionalProperties": False},
        handler=_list_files,
    ))
    reg.register(ToolSpec(
        name="http_get",
        description="HTTP GET a URL. Only allowlisted hosts are reachable.",
        parameters={"type": "object", "properties": {"url": {"type": "string"}},
                    "required": ["url"], "additionalProperties": False},
        handler=_http_get,
        timeout=15.0,
    ))
    reg.register(ToolSpec(
        name="csv_update",
        description="Append a row to a CSV, or set one cell by index.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "op": {"type": "string", "enum": ["append_row", "set_cell"]},
                "row": {"type": "array", "items": {"type": ["string", "number"]}},
                "row_index": {"type": "integer", "minimum": 0},
                "col_index": {"type": "integer", "minimum": 0},
                "value": {"type": ["string", "number"]},
            },
            "required": ["path", "op"],
            "additionalProperties": False,
        },
        handler=_csv_update,
        side_effect=True,
    ))
    reg.register(ToolSpec(
        name="draft_email",
        description="Write an RFC 5322 .eml draft to the workspace outbox. "
                    "Never sends anything.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "sender": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        handler=_draft_email,
        side_effect=True,
    ))
    reg.register(ToolSpec(
        name="calendar_add",
        description="Add an event to the workspace SQLite calendar.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "starts_at": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["title", "starts_at"],
            "additionalProperties": False,
        },
        handler=_calendar_add,
        side_effect=True,
    ))
    reg.register(ToolSpec(
        name="calendar_list",
        description="List events from the workspace SQLite calendar.",
        parameters={"type": "object", "properties": {},
                    "additionalProperties": False},
        handler=_calendar_list,
    ))
    return reg
