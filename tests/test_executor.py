import pytest

from agent_runtime.tools.executor import ToolExecutor, ToolTimeout, idempotency_key
from agent_runtime.tools.registry import ToolContext, ToolError, ToolSpec


def spec_with(handler, **kw):
    defaults = dict(name="t", description="test tool",
                    parameters={"type": "object", "properties": {},
                                "additionalProperties": True},
                    handler=handler, retries=2, backoff=0.1, timeout=2.0)
    defaults.update(kw)
    return ToolSpec(**defaults)


@pytest.fixture
def executor(store, workspace):
    return ToolExecutor(store, ToolContext(workspace=workspace),
                        sleep=lambda _s: None)


def test_success_records_idempotent_result(executor, store):
    calls = []
    spec = spec_with(lambda a, c: (calls.append(1), {"ok": 1})[1])
    out = executor.execute(spec, {}, "run1", 5)
    assert out.ok and out.result == {"ok": 1} and not out.replayed
    # Second execution with the same (run, seq): handler not called again.
    out2 = executor.execute(spec, {}, "run1", 5)
    assert out2.ok and out2.replayed and out2.result == {"ok": 1}
    assert len(calls) == 1
    assert store.get_execution(idempotency_key("run1", 5)) == {"ok": 1}


def test_retries_then_success_with_backoff(store, workspace):
    sleeps = []
    ex = ToolExecutor(store, ToolContext(workspace=workspace),
                      sleep=sleeps.append)
    attempts = []

    def flaky(a, c):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return {"done": True}

    out = ex.execute(spec_with(flaky), {}, "r", 1)
    assert out.ok and out.attempts == 3
    assert sleeps == [0.1, 0.2]  # exponential backoff


def test_retries_exhausted(executor):
    def always(a, c):
        raise RuntimeError("boom")

    out = executor.execute(spec_with(always), {}, "r", 2)
    assert not out.ok and out.attempts == 3
    assert "boom" in out.error


def test_tool_error_not_retried(executor):
    calls = []

    def denied(a, c):
        calls.append(1)
        raise ToolError("host not allowed")

    out = executor.execute(spec_with(denied), {}, "r", 3)
    assert not out.ok and len(calls) == 1
    assert "host not allowed" in out.error


def test_timeout_enforced_and_retried(store, workspace):
    import time

    ex = ToolExecutor(store, ToolContext(workspace=workspace),
                      sleep=lambda _s: None)

    def slow(a, c):
        time.sleep(5)
        return {}

    out = ex.execute(spec_with(slow, timeout=0.05, retries=1), {}, "r", 4)
    assert not out.ok and out.attempts == 2
    assert "timeout" in out.error


def test_invalid_args_fail_before_execution(executor):
    calls = []
    spec = spec_with(lambda a, c: calls.append(1) or {},
                     parameters={"type": "object",
                                 "properties": {"n": {"type": "integer"}},
                                 "required": ["n"],
                                 "additionalProperties": False})
    out = executor.execute(spec, {"n": "not-int"}, "r", 6)
    assert not out.ok and "invalid arguments" in out.error
    assert not calls


def test_fault_hook_injection(store, workspace):
    hooked = []

    def hook(tool, attempt):
        hooked.append((tool, attempt))
        return ToolTimeout("injected") if attempt == 1 else None

    ex = ToolExecutor(store, ToolContext(workspace=workspace),
                      sleep=lambda _s: None, fault_hook=hook)
    out = ex.execute(spec_with(lambda a, c: {"fine": 1}), {}, "r", 7)
    assert out.ok and out.attempts == 2
    assert hooked[0] == ("t", 1)
