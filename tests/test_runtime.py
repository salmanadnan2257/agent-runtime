from agent_runtime import events as ev
from agent_runtime.projection import build_messages, project
from agent_runtime.runtime import approve_call, deny_call

from conftest import AUTO, make_runtime


def test_happy_path_reads_then_finishes(store, workspace):
    (workspace / "members.csv").write_text("name\njane\n")
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "read_file", "args": {"path": "members.csv"}}]},
        {"final": "The sheet has one member: jane."},
    ])
    state = rt.start("data_entry", "who is in the sheet?")
    assert state.status == "finished"
    assert state.final_answer == "The sheet has one member: jane."
    assert state.executed_tools() == ["read_file"]
    types = [e.type for e in store.events(state.run_id)]
    assert types[0] == ev.RUN_CREATED
    assert types[-1] == ev.RUN_FINISHED
    assert ev.CHECKPOINT in types


def test_side_effect_pauses_without_policy(store, workspace):
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "a.txt", "content": "x"}}]},
        {"final": "wrote it"},
    ])
    state = rt.start("data_entry", "write a file")
    assert state.status == "waiting_approval"
    assert not (workspace / "a.txt").exists(), "side effect ran before approval"
    pending = state.pending_approvals()
    assert len(pending) == 1 and pending[0].tool == "write_file"


def test_approval_continues_run(store, workspace):
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "a.txt", "content": "x"}}]},
        {"final": "wrote it"},
    ])
    state = rt.start("data_entry", "write a file")
    call = state.pending_approvals()[0]
    approve_call(store, state.run_id, call.call_id, approver="cli")
    state = rt.drive(state.run_id)
    assert state.status == "finished"
    assert (workspace / "a.txt").read_text() == "x"
    types = [e.type for e in store.events(state.run_id)]
    i_appr = types.index(ev.TOOL_APPROVED)
    i_exec = types.index(ev.TOOL_EXECUTED)
    assert i_appr < i_exec


def test_denial_feeds_observation_to_model(store, workspace):
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "a.txt", "content": "x"}}]},
        {"final": "understood, not writing the file"},
    ])
    state = rt.start("data_entry", "write a file")
    call = state.pending_approvals()[0]
    deny_call(store, state.run_id, call.call_id, reason="not allowed today")
    state = rt.drive(state.run_id)
    assert state.status == "finished"
    assert not (workspace / "a.txt").exists()
    # The denial is visible to the model as a tool observation.
    msgs = build_messages(project(store.events(state.run_id)), "sys")
    denials = [m for m in msgs if m["role"] == "tool"
               and m["content"].get("denied")]
    assert denials and denials[0]["content"]["reason"] == "not allowed today"


def test_approve_or_deny_requires_pending_call(store, workspace):
    rt = make_runtime(store, workspace, [{"final": "done"}])
    state = rt.start("data_entry", "nothing")
    try:
        approve_call(store, state.run_id, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_unknown_tool_reported_not_crash(store, workspace):
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "launch_rockets", "args": {}}]},
        {"final": "that tool does not exist"},
    ], policy=AUTO)
    state = rt.start("data_entry", "do something odd")
    assert state.status == "finished"
    failed = [e for e in store.events(state.run_id) if e.type == ev.TOOL_FAILED]
    assert failed and "unknown tool" in failed[0].payload["error"]


def test_model_call_limit_fails_run(store, workspace):
    script = [{"tool_calls": [{"tool": "list_files", "args": {}}]}] * 40
    rt = make_runtime(store, workspace, script, max_model_calls=5)
    state = rt.start("data_entry", "loop forever")
    assert state.status == "failed"
    assert state.failure_cause == "model_call_limit"


def test_read_only_tools_skip_approval_gate(store, workspace):
    (workspace / "f.txt").write_text("data")
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "read_file", "args": {"path": "f.txt"}}]},
        {"final": "read it"},
    ])  # no approval policy at all
    state = rt.start("data_entry", "read")
    assert state.status == "finished"
    types = [e.type for e in store.events(state.run_id)]
    assert ev.TOOL_APPROVED not in types


def test_two_tool_calls_in_one_turn(store, workspace):
    rt = make_runtime(store, workspace, [
        {"tool_calls": [
            {"tool": "write_file", "args": {"path": "a.txt", "content": "1"}},
            {"tool": "write_file", "args": {"path": "b.txt", "content": "2"}},
        ]},
        {"final": "wrote both"},
    ], policy=AUTO)
    state = rt.start("data_entry", "write two files")
    assert state.status == "finished"
    assert (workspace / "a.txt").exists() and (workspace / "b.txt").exists()
    assert state.executed_tools() == ["write_file", "write_file"]
