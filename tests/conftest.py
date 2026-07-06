from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.adapters.mock import MockModelAdapter
from agent_runtime.agents import get_agent
from agent_runtime.faults import FaultPlan
from agent_runtime.runtime import Runtime
from agent_runtime.store import EventStore
from agent_runtime.tools.builtin import build_default_registry
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.registry import ToolContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "scenarios"


@pytest.fixture
def store(tmp_path) -> EventStore:
    st = EventStore(tmp_path / "events.db")
    yield st
    st.close()


@pytest.fixture
def workspace(tmp_path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def make_runtime(
    store: EventStore,
    workspace: Path,
    script: list[dict],
    agent: str = "data_entry",
    policy=None,
    faults: FaultPlan | None = None,
    registry=None,
    **kwargs,
) -> Runtime:
    a = get_agent(agent)
    reg = registry or build_default_registry().subset(list(a.tools))
    faults = faults or FaultPlan()
    adapter = MockModelAdapter(script, faults=faults)
    executor = ToolExecutor(
        store, ToolContext(workspace=workspace),
        sleep=lambda _s: None, fault_hook=faults.tool_fault,
    )
    return Runtime(store, reg, adapter, executor, a.system_prompt,
                   approval_policy=policy, sleep=lambda _s: None, **kwargs)


AUTO = lambda call: ("approve", "policy:auto")  # noqa: E731
