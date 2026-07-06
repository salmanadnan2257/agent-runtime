import email
from pathlib import Path

import httpx
import pytest

from agent_runtime.tools.builtin import build_default_registry
from agent_runtime.tools.registry import (
    ToolContext, ToolError, ToolRegistry, ToolSpec,
)


@pytest.fixture
def reg():
    return build_default_registry()


@pytest.fixture
def ctx(workspace):
    return ToolContext(workspace=workspace)


def run_tool(reg, name, args, ctx):
    spec = reg.get(name)
    spec.validate_args(args)
    return spec.handler(args, ctx)


# -- registry / schema ---------------------------------------------------


def test_duplicate_registration_rejected(reg):
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(reg.get("read_file"))


def test_invalid_schema_rejected():
    r = ToolRegistry()
    with pytest.raises(Exception):
        r.register(ToolSpec(name="bad", description="x",
                            parameters={"type": "not-a-type"},
                            handler=lambda a, c: {}))


def test_args_validated_against_schema(reg):
    with pytest.raises(Exception):
        reg.get("csv_update").validate_args({"path": "x.csv"})  # missing op
    with pytest.raises(Exception):
        reg.get("read_file").validate_args({"path": 42})
    reg.get("read_file").validate_args({"path": "ok.txt"})


def test_subset_and_public_shape(reg):
    sub = reg.subset(["read_file", "write_file"])
    assert set(sub.tools) == {"read_file", "write_file"}
    with pytest.raises(KeyError):
        reg.subset(["nope"])
    pub = reg.get("draft_email").public()
    assert set(pub) == {"name", "description", "parameters"}


def test_side_effect_flags(reg):
    assert not reg.get("read_file").side_effect
    assert not reg.get("http_get").side_effect
    assert not reg.get("calendar_list").side_effect
    for name in ("write_file", "csv_update", "draft_email", "calendar_add"):
        assert reg.get(name).side_effect, name


# -- filesystem sandbox ----------------------------------------------------


def test_write_then_read_and_list(reg, ctx):
    run_tool(reg, "write_file", {"path": "sub/a.txt", "content": "hello"}, ctx)
    got = run_tool(reg, "read_file", {"path": "sub/a.txt"}, ctx)
    assert got["content"] == "hello"
    listed = run_tool(reg, "list_files", {}, ctx)
    assert "sub/a.txt" in listed["files"]


def test_sandbox_escape_blocked(reg, ctx):
    for path in ("../outside.txt", "a/../../etc/passwd", "/etc/passwd"):
        with pytest.raises(ToolError, match="sandbox"):
            run_tool(reg, "read_file", {"path": path}, ctx)
    with pytest.raises(ToolError, match="sandbox"):
        run_tool(reg, "write_file", {"path": "../evil.txt", "content": "x"}, ctx)


def test_read_missing_file(reg, ctx):
    with pytest.raises(ToolError, match="no such file"):
        run_tool(reg, "read_file", {"path": "ghost.txt"}, ctx)


# -- csv --------------------------------------------------------------------


def test_csv_append_and_set_cell(reg, ctx, workspace):
    (workspace / "t.csv").write_text("a,b\n1,2\n")
    run_tool(reg, "csv_update",
             {"path": "t.csv", "op": "append_row", "row": ["3", 4]}, ctx)
    run_tool(reg, "csv_update",
             {"path": "t.csv", "op": "set_cell", "row_index": 1,
              "col_index": 1, "value": "9"}, ctx)
    assert (workspace / "t.csv").read_text() == "a,b\n1,9\n3,4\n"


def test_csv_set_cell_out_of_range(reg, ctx, workspace):
    (workspace / "t.csv").write_text("a,b\n")
    with pytest.raises(ToolError, match="out of range"):
        run_tool(reg, "csv_update",
                 {"path": "t.csv", "op": "set_cell", "row_index": 7,
                  "col_index": 0, "value": "x"}, ctx)


# -- email drafter ------------------------------------------------------------


def test_draft_email_writes_parseable_eml(reg, ctx, workspace):
    out = run_tool(reg, "draft_email", {
        "to": "ap@example.com", "subject": "Invoice 42 overdue",
        "body": "Please pay.\n"}, ctx)
    eml = workspace / out["draft_path"]
    assert eml.suffix == ".eml"
    msg = email.message_from_bytes(eml.read_bytes())
    assert msg["To"] == "ap@example.com"
    assert msg["Subject"] == "Invoice 42 overdue"
    assert "Please pay." in msg.get_payload()


# -- calendar -------------------------------------------------------------------


def test_calendar_add_and_list(reg, ctx):
    added = run_tool(reg, "calendar_add",
                     {"title": "Call", "starts_at": "2026-07-10T09:00:00"}, ctx)
    assert added["event_id"] == 1
    listed = run_tool(reg, "calendar_list", {}, ctx)
    assert listed["events"][0]["title"] == "Call"


# -- http allowlist ----------------------------------------------------------------


def test_http_get_blocked_host(reg, workspace):
    ctx = ToolContext(workspace=workspace, http_allowlist=("good.example",))
    with pytest.raises(ToolError, match="allowlist"):
        run_tool(reg, "http_get", {"url": "https://bad.example/x"}, ctx)


def test_http_get_allowed_host_uses_injected_client(reg, workspace):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text="payload for " + str(req.url)))
    client = httpx.Client(transport=transport)
    ctx = ToolContext(workspace=workspace, http_allowlist=("good.example",),
                      http_client=client)
    got = run_tool(reg, "http_get", {"url": "https://good.example/doc"}, ctx)
    assert got["status"] == 200
    assert "payload for" in got["body"]
