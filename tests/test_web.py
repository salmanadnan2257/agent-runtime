import re

import pytest
from fastapi.testclient import TestClient

from agent_runtime.bootstrap import start_scenario_run
from agent_runtime.projection import project
from agent_runtime.scenarios.loader import Scenario
from agent_runtime.store import EventStore
from agent_runtime.web.app import create_app
from agent_runtime.web.preview import preview

from conftest import SCENARIO_DIR


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "web.db"
    store = EventStore(db)
    scenario = Scenario.from_file(
        SCENARIO_DIR / "ops_assistant" / "chase_overdue_acme.yaml")
    state = start_scenario_run(store, scenario, "v1", tmp_path / "ws", None)
    store.close()
    client = TestClient(create_app(str(db)))
    return client, state.run_id, tmp_path


def test_index_lists_runs(env):
    client, rid, _ = env
    r = client.get("/")
    assert r.status_code == 200
    assert rid in r.text
    assert "waiting_approval" in r.text


def test_timeline_and_pending_approval_render(env):
    client, rid, _ = env
    r = client.get(f"/runs/{rid}")
    assert r.status_code == 200
    assert "timeline" in r.text
    assert "run_created" in r.text and "tool_requested" in r.text
    assert "pending approvals" in r.text
    assert "draft_email" in r.text
    assert "+ Subject: Overdue invoice INV-1001" in r.text  # diff preview
    assert "simulated usage" in r.text


def test_unknown_run_is_404(env):
    client, _, _ = env
    assert client.get("/runs/doesnotexist").status_code == 404


def test_approve_via_web_completes_run(env):
    client, rid, tmp_path = env
    page = client.get(f"/runs/{rid}").text
    call_id = re.search(r"calls/(call-[\w-]+)/approve", page).group(1)
    r = client.post(f"/runs/{rid}/calls/{call_id}/approve",
                    follow_redirects=False)
    assert r.status_code == 303
    after = client.get(f"/runs/{rid}").text
    assert "finished" in after
    assert "final answer" in after
    assert list((tmp_path / "ws" / "outbox").glob("*.eml"))
    # Approving again is a 409, not a double execution.
    again = client.post(f"/runs/{rid}/calls/{call_id}/approve")
    assert again.status_code == 409


def test_deny_via_web_with_reason(env):
    client, rid, tmp_path = env
    page = client.get(f"/runs/{rid}").text
    call_id = re.search(r"calls/(call-[\w-]+)/deny", page).group(1)
    r = client.post(f"/runs/{rid}/calls/{call_id}/deny",
                    data={"reason": "hold this invoice"},
                    follow_redirects=False)
    assert r.status_code == 303
    store = EventStore(tmp_path / "web.db")
    state = project(store.events(rid))
    store.close()
    assert state.status == "finished"
    denied = [c for c in state.calls.values() if c.status == "denied"]
    assert denied and denied[0].deny_reason == "hold this invoice"
    assert not list((tmp_path / "ws" / "outbox").glob("*.eml"))


def test_preview_write_file_unified_diff(tmp_path):
    (tmp_path / "a.txt").write_text("old line\n")
    diff = preview("write_file", {"path": "a.txt", "content": "new line\n"},
                   tmp_path)
    assert "-old line" in diff and "+new line" in diff


def test_preview_csv_and_calendar(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n")
    diff = preview("csv_update", {"path": "t.csv", "op": "append_row",
                                  "row": ["3", "4"]}, tmp_path)
    assert "+3,4" in diff
    cal = preview("calendar_add", {"title": "Sync",
                                   "starts_at": "2026-07-08T09:00"}, tmp_path)
    assert "Sync" in cal and cal.startswith("+")
