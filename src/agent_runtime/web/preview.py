"""Diff-style previews of proposed side-effecting tool calls."""

from __future__ import annotations

import csv
import difflib
import io
import json
from pathlib import Path
from typing import Any


def _unified(old: str, new: str, name: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}",
    )
    return "".join(diff) or "(no textual change)"


def preview(tool: str, args: dict[str, Any], workspace: Path) -> str:
    """Human-readable preview of what a proposed call would change."""
    try:
        if tool == "write_file":
            target = workspace / args.get("path", "")
            old = target.read_text(encoding="utf-8") if target.is_file() else ""
            return _unified(old, args.get("content", ""), args.get("path", "?"))
        if tool == "csv_update":
            target = workspace / args.get("path", "")
            old_rows: list[list[str]] = []
            if target.is_file():
                with open(target, newline="", encoding="utf-8") as fh:
                    old_rows = [list(r) for r in csv.reader(fh)]
            new_rows = [list(r) for r in old_rows]
            if args.get("op") == "append_row":
                new_rows.append([str(v) for v in args.get("row", [])])
            elif args.get("op") == "set_cell":
                r, c = args.get("row_index", 0), args.get("col_index", 0)
                if r < len(new_rows) and c < len(new_rows[r]):
                    new_rows[r][c] = str(args.get("value", ""))

            def render(rows: list[list[str]]) -> str:
                buf = io.StringIO()
                csv.writer(buf, lineterminator="\n").writerows(rows)
                return buf.getvalue()

            return _unified(render(old_rows), render(new_rows),
                            args.get("path", "?"))
        if tool == "draft_email":
            body = args.get("body", "")
            lines = [f"+ To: {args.get('to', '?')}",
                     f"+ Subject: {args.get('subject', '?')}", "+"]
            lines += [f"+ {ln}" for ln in body.splitlines()]
            return "\n".join(lines)
        if tool == "calendar_add":
            return (f"+ calendar event: {args.get('title', '?')} "
                    f"at {args.get('starts_at', '?')}"
                    + (f"\n+ notes: {args['notes']}" if args.get("notes") else ""))
    except OSError as exc:
        return f"(preview unavailable: {exc})"
    return json.dumps(args, indent=2, sort_keys=True)
