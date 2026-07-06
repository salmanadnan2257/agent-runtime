"""Construct runtimes for new and existing runs (shared by CLI and web).

Run metadata (workspace path, adapter kind, scenario file, behavior
version) lives in the run_created event, so any process with the
database can rebuild the exact runtime needed to continue a run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import events as ev
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.base import ModelAdapter
from .adapters.mock import MockModelAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .agents import get_agent
from .faults import FaultPlan
from .projection import project
from .runtime import ApprovalPolicy, Runtime
from .scenarios.loader import Scenario
from .store import EventStore
from .tools.builtin import build_default_registry
from .tools.executor import ToolExecutor
from .tools.registry import ToolContext

DEFAULT_DB = os.environ.get("AGENTRUN_DB", "agentrun.db")


def default_allowlist() -> tuple[str, ...]:
    raw = os.environ.get("AGENTRUN_HTTP_ALLOWLIST", "")
    return tuple(h.strip() for h in raw.split(",") if h.strip())


def _make_adapter(meta: dict[str, Any], model_calls_done: int = 0) -> ModelAdapter:
    kind = meta.get("adapter", "mock")
    if kind == "mock":
        scenario = Scenario.from_file(meta["scenario_path"])
        script = scenario.behaviors[meta["behavior"]]
        faults = FaultPlan.from_dict(scenario.faults)
        adapter = MockModelAdapter(script, faults=faults)
        adapter.calls = model_calls_done
        adapter.script_i = model_calls_done
        return adapter
    if kind == "anthropic":
        return (AnthropicAdapter(model=meta["model"]) if meta.get("model")
                else AnthropicAdapter())
    if kind == "openai":
        return (OpenAIAdapter(model=meta["model"]) if meta.get("model")
                else OpenAIAdapter())
    raise ValueError(f"unknown adapter kind: {kind}")


def start_scenario_run(
    store: EventStore,
    scenario: Scenario,
    behavior: str,
    workspace: Path,
    approval_policy: ApprovalPolicy | None,
):
    """Start a mock-adapter run defined by a scenario file."""
    if behavior not in scenario.behaviors:
        raise ValueError(f"behavior {behavior!r} not in scenario "
                         f"(have: {', '.join(sorted(scenario.behaviors))})")
    workspace.mkdir(parents=True, exist_ok=True)
    for rel, content in scenario.workspace_files.items():
        p = workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    agent = get_agent(scenario.agent)
    meta = {
        "adapter": "mock",
        "scenario_path": str(Path(scenario.path).resolve()),
        "behavior": behavior,
        "workspace": str(workspace.resolve()),
        "allowlist": list(default_allowlist()),
    }
    rt = runtime_from_meta(store, meta, agent.name, approval_policy)
    return rt.start(agent.name, scenario.request, meta=meta)


def runtime_from_meta(
    store: EventStore,
    meta: dict[str, Any],
    agent_name: str,
    approval_policy: ApprovalPolicy | None,
    model_calls_done: int = 0,
) -> Runtime:
    agent = get_agent(agent_name)
    registry = build_default_registry().subset(list(agent.tools))
    ctx = ToolContext(
        workspace=Path(meta["workspace"]),
        http_allowlist=tuple(meta.get("allowlist", [])),
    )
    faults = None
    if meta.get("adapter") == "mock" and meta.get("scenario_path"):
        faults = FaultPlan.from_dict(
            Scenario.from_file(meta["scenario_path"]).faults)
    executor = ToolExecutor(
        store, ctx, fault_hook=faults.tool_fault if faults else None)
    adapter = _make_adapter(meta, model_calls_done)
    return Runtime(store, registry, adapter, executor, agent.system_prompt,
                   approval_policy=approval_policy)


def resume_run(
    store: EventStore, run_id: str, approval_policy: ApprovalPolicy | None = None
):
    """Rebuild the runtime for an existing run and drive it forward."""
    evts = store.events(run_id)
    if not evts:
        raise ValueError(f"unknown run: {run_id}")
    state = project(evts)
    if state.status in ("finished", "failed"):
        return state
    # Adapter position: one script turn was consumed per parsed model turn.
    consumed = sum(1 for e in evts
                   if e.type == ev.MODEL_RESPONDED and not e.payload.get("malformed"))
    rt = runtime_from_meta(store, state.meta, state.agent, approval_policy,
                           model_calls_done=consumed)
    if state.meta.get("adapter") == "mock" and hasattr(rt.adapter, "calls"):
        # Fault matching counts every adapter invocation, including errors.
        rt.adapter.calls = sum(
            1 for e in evts if e.type in (ev.MODEL_RESPONDED, ev.MODEL_ERROR))
    return rt.drive(run_id)
