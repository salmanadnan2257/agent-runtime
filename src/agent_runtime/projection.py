"""Derive the full state of a run from its event log.

This is the only place that interprets event ordering. The runtime,
CLI, web UI and scenario assertions all go through RunState so they
cannot disagree about what a log means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import events as ev
from .events import Event

# Run statuses.
RUNNING = "running"
WAITING_APPROVAL = "waiting_approval"
FINISHED = "finished"
FAILED = "failed"


@dataclass
class ToolCallState:
    """Lifecycle of a single requested tool call."""

    call_id: str
    tool: str
    args: dict[str, Any]
    side_effect: bool
    requested_seq: int
    status: str = "requested"  # requested|approved|denied|executed|failed
    result: dict[str, Any] | None = None
    error: str | None = None
    deny_reason: str | None = None


@dataclass
class RunState:
    run_id: str
    agent: str = ""
    request: str = ""
    status: str = RUNNING
    final_answer: str | None = None
    failure_cause: str | None = None
    calls: dict[str, ToolCallState] = field(default_factory=dict)
    call_order: list[str] = field(default_factory=list)
    model_calls: int = 0
    events: list[Event] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    def pending_approvals(self) -> list[ToolCallState]:
        return [
            self.calls[cid]
            for cid in self.call_order
            if self.calls[cid].status == "requested" and self.calls[cid].side_effect
        ]

    def unfinished_calls(self) -> list[ToolCallState]:
        """Calls with no terminal event yet, in request order."""
        return [
            self.calls[cid]
            for cid in self.call_order
            if self.calls[cid].status in ("requested", "approved")
        ]

    def executed_tools(self) -> list[str]:
        return [
            self.calls[cid].tool
            for cid in self.call_order
            if self.calls[cid].status == "executed"
        ]


def project(evts: list[Event]) -> RunState:
    if not evts:
        raise ValueError("empty event log")
    state = RunState(run_id=evts[0].run_id, events=list(evts))
    for e in evts:
        p = e.payload
        if e.type == ev.RUN_CREATED:
            state.agent = p.get("agent", "")
            state.request = p.get("request", "")
            state.meta = p.get("meta", {})
        elif e.type == ev.MODEL_REQUESTED:
            state.model_calls += 1
        elif e.type == ev.TOOL_REQUESTED:
            cid = p["call_id"]
            state.calls[cid] = ToolCallState(
                call_id=cid,
                tool=p["tool"],
                args=p.get("args", {}),
                side_effect=bool(p.get("side_effect")),
                requested_seq=e.seq,
            )
            state.call_order.append(cid)
        elif e.type == ev.TOOL_APPROVED:
            state.calls[p["call_id"]].status = "approved"
        elif e.type == ev.TOOL_DENIED:
            c = state.calls[p["call_id"]]
            c.status = "denied"
            c.deny_reason = p.get("reason")
        elif e.type == ev.TOOL_EXECUTED:
            c = state.calls[p["call_id"]]
            c.status = "executed"
            c.result = p.get("result")
        elif e.type == ev.TOOL_FAILED:
            c = state.calls[p["call_id"]]
            c.status = "failed"
            c.error = p.get("error")
        elif e.type == ev.RUN_FINISHED:
            state.status = FINISHED
            state.final_answer = p.get("final_answer")
        elif e.type == ev.RUN_FAILED:
            state.status = FAILED
            state.failure_cause = p.get("cause")

    if state.status == RUNNING and state.pending_approvals():
        state.status = WAITING_APPROVAL
    return state


def build_messages(state: RunState, system_prompt: str) -> list[dict[str, Any]]:
    """Reconstruct the model conversation from the event log.

    Deterministic: same log, same messages. Used by live adapters and by
    the runtime when resuming, so a resumed run sends exactly what an
    uninterrupted run would have sent.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.request},
    ]
    for e in state.events:
        p = e.payload
        if e.type == ev.MODEL_RESPONDED and not p.get("malformed"):
            turn = p["turn"]
            if turn.get("tool_calls"):
                messages.append(
                    {"role": "assistant", "content": turn.get("text", ""),
                     "tool_calls": turn["tool_calls"]}
                )
            else:
                messages.append({"role": "assistant", "content": turn.get("text", "")})
        elif e.type == ev.MODEL_RESPONDED and p.get("malformed"):
            messages.append(
                {"role": "user",
                 "content": "Your previous output could not be parsed. "
                            "Reply with either tool calls or a final answer."}
            )
        elif e.type == ev.TOOL_EXECUTED:
            messages.append(
                {"role": "tool", "call_id": p["call_id"], "content": p["result"]}
            )
        elif e.type == ev.TOOL_FAILED:
            messages.append(
                {"role": "tool", "call_id": p["call_id"],
                 "content": {"error": p.get("error", "tool failed")}}
            )
        elif e.type == ev.TOOL_DENIED:
            messages.append(
                {"role": "tool", "call_id": p["call_id"],
                 "content": {"denied": True,
                             "reason": p.get("reason") or "denied by operator"}}
            )
    return messages
