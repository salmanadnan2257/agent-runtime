"""Crash mid-run, resume from the log, prove no duplicated side effects."""

import pytest

from agent_runtime import events as ev
from agent_runtime.adapters.mock import MockModelAdapter
from agent_runtime.bootstrap import resume_run, start_scenario_run
from agent_runtime.projection import project
from agent_runtime.runtime import Runtime, approve_call
from agent_runtime.scenarios.loader import Scenario
from agent_runtime.tools.builtin import build_default_registry
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.registry import ToolContext, ToolRegistry, ToolSpec

from conftest import AUTO, SCENARIO_DIR


class SimulatedCrash(BaseException):
    pass


class CrashAfterExecute:
    """Executes the tool for real, then dies before the event is appended:
    the worst possible crash point for duplicate side effects."""

    def __init__(self, inner: ToolExecutor):
        self.inner = inner

    def execute(self, spec, args, run_id, requested_seq):
        outcome = self.inner.execute(spec, args, run_id, requested_seq)
        raise SimulatedCrash


def make_counting_registry(counter: dict) -> ToolRegistry:
    def bump(args, ctx):
        counter["n"] = counter.get("n", 0) + 1
        return {"count": counter["n"]}

    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="bump", description="increment a counter",
        parameters={"type": "object", "properties": {},
                    "additionalProperties": False},
        handler=bump, side_effect=True))
    return reg


SCRIPT = [
    {"tool_calls": [{"tool": "bump", "args": {}}]},
    {"final": "bumped exactly once"},
]


def build(store, workspace, registry, executor, script_pos=0):
    adapter = MockModelAdapter(SCRIPT)
    adapter.calls = adapter.script_i = script_pos
    return Runtime(store, registry, adapter, executor, "sys",
                   approval_policy=AUTO, sleep=lambda _s: None)


def test_crash_after_side_effect_resumes_without_duplicate(store, workspace):
    counter: dict = {}
    registry = make_counting_registry(counter)
    real = ToolExecutor(store, ToolContext(workspace=workspace),
                        sleep=lambda _s: None)

    rt = build(store, workspace, registry, CrashAfterExecute(real))
    with pytest.raises(SimulatedCrash):
        rt.start("data_entry", "bump the counter")
    run_id = store.run_ids()[0]

    # The crash left the log with an approved call but no terminal event.
    state = project(store.events(run_id))
    assert state.status == "running"
    assert counter["n"] == 1, "side effect ran once before the crash"
    types = [e.type for e in store.events(run_id)]
    assert ev.TOOL_APPROVED in types and ev.TOOL_EXECUTED not in types

    # Fresh process: new runtime, real executor, adapter repositioned.
    rt2 = build(store, workspace, registry, real, script_pos=1)
    state = rt2.drive(run_id)
    assert state.status == "finished"
    assert state.final_answer == "bumped exactly once"
    assert counter["n"] == 1, "resume must not repeat the side effect"

    executed = [e for e in store.events(run_id) if e.type == ev.TOOL_EXECUTED]
    assert len(executed) == 1
    assert executed[0].payload["replayed"] is True
    assert executed[0].payload["result"] == {"count": 1}


def test_crash_before_execution_executes_on_resume(store, workspace):
    counter: dict = {}
    registry = make_counting_registry(counter)
    real = ToolExecutor(store, ToolContext(workspace=workspace),
                        sleep=lambda _s: None)

    class CrashBeforeExecute:
        def execute(self, spec, args, run_id, requested_seq):
            raise SimulatedCrash

    rt = build(store, workspace, registry, CrashBeforeExecute())
    with pytest.raises(SimulatedCrash):
        rt.start("data_entry", "bump the counter")
    run_id = store.run_ids()[0]
    assert counter.get("n", 0) == 0

    rt2 = build(store, workspace, registry, real, script_pos=1)
    state = rt2.drive(run_id)
    assert state.status == "finished"
    assert counter["n"] == 1
    executed = [e for e in store.events(run_id) if e.type == ev.TOOL_EXECUTED]
    assert executed[0].payload["replayed"] is False


def test_bootstrap_resume_after_approval(store, tmp_path):
    scenario = Scenario.from_file(
        SCENARIO_DIR / "ops_assistant" / "chase_overdue_acme.yaml")
    ws = tmp_path / "ws2"
    state = start_scenario_run(store, scenario, "v1", ws, approval_policy=None)
    assert state.status == "waiting_approval"
    call = state.pending_approvals()[0]
    assert call.tool == "draft_email"
    assert not list(ws.glob("outbox/*.eml"))

    approve_call(store, state.run_id, call.call_id)
    state = resume_run(store, state.run_id)
    assert state.status == "finished"
    assert "INV-1001" in (state.final_answer or "")
    assert list(ws.glob("outbox/*.eml"))
    # Resuming a finished run is a no-op.
    again = resume_run(store, state.run_id)
    assert again.status == "finished"
