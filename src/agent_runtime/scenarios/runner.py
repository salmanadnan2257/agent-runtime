"""Run scenario packs against a behavior version and evaluate assertions."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import events as ev
from ..accounting import account
from ..adapters.mock import MockModelAdapter
from ..agents import get_agent
from ..faults import FaultPlan
from ..projection import RunState, ToolCallState, project
from ..runtime import Runtime
from ..store import EventStore
from ..tools.builtin import build_default_registry
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolContext
from .loader import Scenario


@dataclass
class ScenarioResult:
    name: str
    version: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    event_sig: list[str] = field(default_factory=list)
    event_count: int = 0
    status: str = ""
    final_answer: str = ""
    cost_usd: float = 0.0
    error: str = ""  # infrastructure error, distinct from assertion failure


def _approval_policy(approvals: Any):
    if approvals == "auto":
        return lambda call: ("approve", "policy:auto")
    deny = set(approvals.get("deny", []))
    reason = approvals.get("reason", "denied by scenario policy")

    def policy(call: ToolCallState):
        if call.tool in deny:
            return ("deny", "policy:scenario", reason)
        return ("approve", "policy:auto")

    return policy


def event_signature(events) -> list[str]:
    """Compact per-event fingerprint used to locate divergence points."""
    sig = []
    for e in events:
        tool = e.payload.get("tool") or ""
        if not tool and "call_id" in e.payload:
            tool = e.payload["call_id"]
        sig.append(f"{e.type}:{tool}" if tool else e.type)
    return sig


def run_scenario(
    scenario: Scenario,
    version: str,
    workdir: str | Path | None = None,
) -> ScenarioResult:
    if version not in scenario.behaviors:
        return ScenarioResult(
            name=scenario.name, version=version, passed=False,
            error=f"behavior version {version!r} not defined "
                  f"(have: {', '.join(sorted(scenario.behaviors))})")
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="agentrun-"))
    ws = base / f"{scenario.name}-{version}" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    for rel, content in scenario.workspace_files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    store = EventStore(base / f"{scenario.name}-{version}" / "events.db")
    try:
        agent = get_agent(scenario.agent)
        registry = build_default_registry().subset(list(agent.tools))
        faults = FaultPlan.from_dict(scenario.faults)
        adapter = MockModelAdapter(scenario.behaviors[version], faults=faults)
        ctx = ToolContext(workspace=ws)
        executor = ToolExecutor(store, ctx, sleep=lambda _s: None,
                                fault_hook=faults.tool_fault)
        rt = Runtime(store, registry, adapter, executor, agent.system_prompt,
                     approval_policy=_approval_policy(scenario.approvals),
                     sleep=lambda _s: None)
        state = rt.start(agent.name, scenario.request)
        events = store.events(state.run_id)
        failures = check_expectations(scenario.expect, state, events, ws)
        return ScenarioResult(
            name=scenario.name, version=version,
            passed=not failures, failures=failures,
            event_sig=event_signature(events), event_count=len(events),
            status=state.status, final_answer=state.final_answer or "",
            cost_usd=account(events).cost_usd,
        )
    except Exception as exc:
        return ScenarioResult(name=scenario.name, version=version, passed=False,
                              error=f"{type(exc).__name__}: {exc}")
    finally:
        store.close()


def check_expectations(
    expect: dict[str, Any], state: RunState, events, workspace: Path
) -> list[str]:
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    if "status" in expect and state.status != expect["status"]:
        fail(f"status: expected {expect['status']}, got {state.status}")

    executed = state.executed_tools()
    if "tools_executed" in expect and executed != expect["tools_executed"]:
        fail(f"tools_executed: expected {expect['tools_executed']}, got {executed}")
    for tool in expect.get("tools_executed_contains", []):
        if tool not in executed:
            fail(f"tools_executed_contains: {tool} never executed (got {executed})")
    for tool in expect.get("tools_not_executed", []):
        if tool in executed:
            fail(f"tools_not_executed: {tool} was executed")

    denied = [c.tool for c in state.calls.values() if c.status == "denied"]
    for tool in expect.get("denied_tools", []):
        if tool not in denied:
            fail(f"denied_tools: {tool} was not denied (denied: {denied})")

    final = state.final_answer or ""
    for needle in expect.get("final_contains", []):
        if needle.lower() not in final.lower():
            fail(f"final_contains: {needle!r} not in final answer {final[:120]!r}")
    for needle in expect.get("final_not_contains", []):
        if needle.lower() in final.lower():
            fail(f"final_not_contains: {needle!r} found in final answer")

    if "failure_cause_contains" in expect:
        cause = state.failure_cause or ""
        if expect["failure_cause_contains"] not in cause:
            fail(f"failure_cause_contains: {expect['failure_cause_contains']!r} "
                 f"not in {cause!r}")

    for rel in expect.get("files_exist", []):
        if not (workspace / rel).is_file():
            fail(f"files_exist: workspace file missing: {rel}")

    if "min_events" in expect and len(events) < expect["min_events"]:
        fail(f"min_events: expected >= {expect['min_events']}, got {len(events)}")

    # Structural invariant, checked whenever asserted (and it always holds
    # by construction; the assertion documents it per scenario).
    if expect.get("no_side_effect_before_approval"):
        approved_at: dict[str, int] = {}
        requested_side_effect: dict[str, bool] = {}
        for e in events:
            p = e.payload
            if e.type == ev.TOOL_REQUESTED:
                requested_side_effect[p["call_id"]] = bool(p.get("side_effect"))
            elif e.type == ev.TOOL_APPROVED:
                approved_at[p["call_id"]] = e.seq
            elif e.type == ev.TOOL_EXECUTED:
                cid = p["call_id"]
                if requested_side_effect.get(cid) and (
                    cid not in approved_at or approved_at[cid] > e.seq
                ):
                    fail(f"side effect executed without prior approval: {cid}")
    return failures


def run_pack(
    scenarios: list[Scenario], version: str, workdir: str | Path | None = None
) -> list[ScenarioResult]:
    return [run_scenario(s, version, workdir=workdir) for s in scenarios]
