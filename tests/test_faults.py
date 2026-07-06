import pytest

from agent_runtime import events as ev
from agent_runtime.faults import FaultPlan

from conftest import AUTO, make_runtime


def test_fault_plan_validation():
    plan = FaultPlan.from_dict({
        "model": [{"at_call": 1, "kind": "adapter_500"}],
        "tools": {"read_file": [{"at_attempt": 2, "kind": "timeout"}]},
    })
    assert plan.model_fault(1) == "adapter_500"
    assert plan.model_fault(2) is None
    assert plan.tool_fault("read_file", 2) is not None
    assert plan.tool_fault("read_file", 1) is None
    with pytest.raises(ValueError, match="unknown model fault"):
        FaultPlan.from_dict({"model": [{"at_call": 1, "kind": "meteor"}]})
    with pytest.raises(ValueError, match="unknown tool fault"):
        FaultPlan.from_dict({"tools": {"x": [{"at_attempt": 1, "kind": "meteor"}]}})


def test_adapter_500_retried_then_recovers(store, workspace):
    faults = FaultPlan.from_dict({"model": [{"at_call": 1, "kind": "adapter_500"}]})
    rt = make_runtime(store, workspace, [{"final": "recovered"}], faults=faults)
    state = rt.start("data_entry", "hi")
    assert state.status == "finished"
    errors = [e for e in store.events(state.run_id) if e.type == ev.MODEL_ERROR]
    assert len(errors) == 1
    assert "500" in errors[0].payload["error"]


def test_adapter_500_exhaustion_fails_run_cleanly(store, workspace):
    faults = FaultPlan.from_dict({"model": [
        {"at_call": i, "kind": "adapter_500"} for i in (1, 2, 3)]})
    rt = make_runtime(store, workspace, [{"final": "never"}], faults=faults)
    state = rt.start("data_entry", "hi")  # must not raise
    assert state.status == "failed"
    assert "adapter_error" in state.failure_cause
    assert len([e for e in store.events(state.run_id)
                if e.type == ev.MODEL_ERROR]) == 3


def test_malformed_output_gets_corrective_observation(store, workspace):
    faults = FaultPlan.from_dict({"model": [{"at_call": 1, "kind": "malformed"}]})
    rt = make_runtime(store, workspace, [{"final": "second try worked"}],
                      faults=faults)
    state = rt.start("data_entry", "hi")
    assert state.status == "finished"
    malformed = [e for e in store.events(state.run_id)
                 if e.type == ev.MODEL_RESPONDED and e.payload.get("malformed")]
    assert len(malformed) == 1


def test_repeated_malformed_output_fails_run(store, workspace):
    faults = FaultPlan.from_dict({"model": [
        {"at_call": i, "kind": "malformed"} for i in (1, 2, 3)]})
    rt = make_runtime(store, workspace, [{"final": "never reached"}],
                      faults=faults)
    state = rt.start("data_entry", "hi")
    assert state.status == "failed"
    assert "malformed_model_output" in state.failure_cause


def test_tool_timeout_injected_then_retry_succeeds(store, workspace):
    (workspace / "f.txt").write_text("data")
    faults = FaultPlan.from_dict({
        "tools": {"read_file": [{"at_attempt": 1, "kind": "timeout"}]}})
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "read_file", "args": {"path": "f.txt"}}]},
        {"final": "read after retry"},
    ], faults=faults)
    state = rt.start("data_entry", "read f.txt")
    assert state.status == "finished"
    executed = [e for e in store.events(state.run_id)
                if e.type == ev.TOOL_EXECUTED]
    assert executed[0].payload["attempts"] == 2


def test_tool_exception_exhausts_and_model_observes_failure(store, workspace):
    faults = FaultPlan.from_dict({"tools": {"list_files": [
        {"at_attempt": i, "kind": "exception", "message": "io down"}
        for i in (1, 2, 3)]}})
    rt = make_runtime(store, workspace, [
        {"tool_calls": [{"tool": "list_files", "args": {}}]},
        {"final": "listing failed, giving up gracefully"},
    ], faults=faults, policy=AUTO)
    state = rt.start("data_entry", "list files")
    assert state.status == "finished"  # loop degraded cleanly, no crash
    failed = [e for e in store.events(state.run_id) if e.type == ev.TOOL_FAILED]
    assert failed and "io down" in failed[0].payload["error"]
    assert failed[0].payload["attempts"] == 3
