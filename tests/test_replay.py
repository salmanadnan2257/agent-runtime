from agent_runtime.agents import get_agent
from agent_runtime.faults import FaultPlan
from agent_runtime.replay import replay_run, replay_state
from agent_runtime.runtime import approve_call
from agent_runtime.tools.builtin import build_default_registry

from conftest import AUTO, make_runtime


def registry_for(agent: str):
    return build_default_registry().subset(list(get_agent(agent).tools))


def prompt_for(agent: str) -> str:
    return get_agent(agent).system_prompt


def test_replay_is_byte_identical(store, workspace):
    (workspace / "n.txt").write_text("note")
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "read_file", "args": {"path": "n.txt"}}]},
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "out.txt", "content": "summary"}}]},
        {"final": "done, summary written"},
    ], policy=AUTO)
    state = rt.start("data_entry", "summarize n.txt")
    assert state.status == "finished"

    result = replay_run(store, registry_for("data_entry"), state.run_id,
                        prompt_for("data_entry"))
    assert result.identical, (
        f"divergence at {result.first_divergence}:\n"
        f"{result.original_canonical}\nvs\n{result.replay_canonical}")
    assert result.replayed_events == result.original_events


def test_replay_does_not_touch_tools_or_adapter(store, workspace):
    (workspace / "n.txt").write_text("v1")
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "n.txt", "content": "v2"}}]},
        {"final": "updated"},
    ], policy=AUTO)
    state = rt.start("data_entry", "update the file")
    assert (workspace / "n.txt").read_text() == "v2"
    (workspace / "n.txt").write_text("changed after the run")

    result = replay_run(store, registry_for("data_entry"), state.run_id,
                        prompt_for("data_entry"))
    assert result.identical
    # Replay never re-executed the tool: the file is untouched.
    assert (workspace / "n.txt").read_text() == "changed after the run"


def test_replay_covers_denials_and_faults(store, workspace):
    faults = FaultPlan.from_dict({
        "model": [{"at_call": 1, "kind": "adapter_500"}],
        "tools": {"list_files": [{"at_attempt": 1, "kind": "exception"}]},
    })
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "list_files", "args": {}}]},
        {"tool_calls": [{"tool": "write_file",
                         "args": {"path": "x.txt", "content": "x"}}]},
        {"final": "handled everything"},
    ], faults=faults)
    state = rt.start("data_entry", "messy run")
    call = state.pending_approvals()[0]
    from agent_runtime.runtime import deny_call
    deny_call(store, state.run_id, call.call_id, reason="no writes")
    # continue: model turn 3 needed; script index already at 2
    state = rt.drive(state.run_id)
    assert state.status == "finished"

    result = replay_run(store, registry_for("data_entry"), state.run_id,
                        prompt_for("data_entry"))
    assert result.identical, (
        f"divergence at event {result.first_divergence}")


def test_replay_until_gives_stepwise_state(store, workspace):
    (workspace / "n.txt").write_text("note")
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "read_file", "args": {"path": "n.txt"}}]},
        {"final": "done"},
    ])
    state = rt.start("data_entry", "read the note")
    events = store.events(state.run_id)
    # Find the seq right after the tool executed but before the final answer.
    exec_seq = next(e.seq for e in events if e.type == "tool_executed")
    mid = replay_state(store, state.run_id, until=exec_seq)
    assert mid.status == "running"
    assert mid.executed_tools() == ["read_file"]
    assert mid.final_answer is None

    result = replay_run(store, registry_for("data_entry"), state.run_id,
                        prompt_for("data_entry"), until=exec_seq)
    assert result.identical
    assert result.replayed_events == exec_seq + 1


def test_replay_of_failed_run(store, workspace):
    faults = FaultPlan.from_dict({"model": [
        {"at_call": i, "kind": "adapter_500"} for i in (1, 2, 3)]})
    rt = make_runtime(store, workspace, [{"final": "never"}], faults=faults)
    state = rt.start("data_entry", "doomed")
    assert state.status == "failed"
    result = replay_run(store, registry_for("data_entry"), state.run_id,
                        prompt_for("data_entry"))
    assert result.identical
