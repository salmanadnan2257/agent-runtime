"""The agent loop: plan, tool call, observe, continue or finish.

Everything the loop does is recorded as events before or immediately
after it happens, so a run can be killed at any point and resumed, or
replayed byte for byte from its log.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from . import events as ev
from .adapters.base import AdapterError, MalformedOutputError, ModelAdapter
from .projection import RunState, ToolCallState, build_messages, project
from .store import EventStore
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry

# Approval policy return values: None (wait for a human),
# ("approve", approver) or ("deny", approver, reason).
ApprovalDecision = tuple[str, ...] | None
ApprovalPolicy = Callable[[ToolCallState], ApprovalDecision]


def auto_approve_policy(call: ToolCallState) -> ApprovalDecision:
    return ("approve", "policy:auto")


def _digest(messages: list[dict[str, Any]]) -> str:
    blob = json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Runtime:
    def __init__(
        self,
        store: EventStore,
        registry: ToolRegistry,
        adapter: ModelAdapter,
        executor: ToolExecutor,
        system_prompt: str,
        approval_policy: ApprovalPolicy | None = None,
        max_model_calls: int = 25,
        max_adapter_retries: int = 2,
        max_malformed: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        adapter_backoff: float = 0.5,
    ):
        self.store = store
        self.registry = registry
        self.adapter = adapter
        self.executor = executor
        self.system_prompt = system_prompt
        self.approval_policy = approval_policy
        self.max_model_calls = max_model_calls
        self.max_adapter_retries = max_adapter_retries
        self.max_malformed = max_malformed
        self.sleep = sleep
        self.adapter_backoff = adapter_backoff

    # -- public API ---------------------------------------------------------

    def start(
        self, agent: str, request: str, meta: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunState:
        run_id = run_id or self.store.new_run_id()
        self.store.append(run_id, ev.RUN_CREATED, {
            "agent": agent, "request": request, "meta": meta or {},
        })
        return self.drive(run_id)

    def drive(self, run_id: str) -> RunState:
        """Advance the run until it finishes, fails, or needs approval."""
        while True:
            state = project(self.store.events(run_id))
            if state.status in ("finished", "failed"):
                return state

            paused = self._process_tool_calls(run_id, state)
            if paused:
                return project(self.store.events(run_id))

            state = project(self.store.events(run_id))
            if state.status in ("finished", "failed"):
                return state

            if state.model_calls >= self.max_model_calls:
                self.store.append(run_id, ev.RUN_FAILED,
                                  {"cause": "model_call_limit"})
                return project(self.store.events(run_id))

            self._checkpoint(run_id, state)
            done = self._model_step(run_id, state)
            if done:
                return project(self.store.events(run_id))

    # -- internals -----------------------------------------------------------

    def _checkpoint(self, run_id: str, state: RunState) -> None:
        # One checkpoint per completed observe phase; skip the first turn
        # and avoid duplicates when re-driving a resumed run.
        if state.model_calls == 0:
            return
        if state.events and state.events[-1].type == ev.CHECKPOINT:
            return
        self.store.append(run_id, ev.CHECKPOINT, {
            "model_calls": state.model_calls,
            "calls_resolved": len(state.executed_tools()),
        })

    def _process_tool_calls(self, run_id: str, state: RunState) -> bool:
        """Resolve pending tool calls. Returns True if paused for approval."""
        for call in state.unfinished_calls():
            if call.status == "requested" and call.side_effect:
                decision = self.approval_policy(call) if self.approval_policy else None
                if decision is None:
                    return True  # waiting for a human
                if decision[0] == "approve":
                    self.store.append(run_id, ev.TOOL_APPROVED, {
                        "call_id": call.call_id, "approver": decision[1],
                    })
                    self._execute(run_id, call)
                else:
                    self.store.append(run_id, ev.TOOL_DENIED, {
                        "call_id": call.call_id, "approver": decision[1],
                        "reason": decision[2] if len(decision) > 2 else "denied",
                    })
            else:
                # Read-only call, or one already approved (e.g. via CLI/web).
                self._execute(run_id, call)
        return False

    def _execute(self, run_id: str, call: ToolCallState) -> None:
        if call.tool not in self.registry:
            self.store.append(run_id, ev.TOOL_FAILED, {
                "call_id": call.call_id,
                "error": f"unknown tool: {call.tool}", "attempts": 0,
            })
            return
        spec = self.registry.get(call.tool)
        outcome = self.executor.execute(spec, call.args, run_id, call.requested_seq)
        if outcome.ok:
            self.store.append(run_id, ev.TOOL_EXECUTED, {
                "call_id": call.call_id,
                "result": outcome.result,
                "attempts": outcome.attempts,
                "replayed": outcome.replayed,
            })
        else:
            self.store.append(run_id, ev.TOOL_FAILED, {
                "call_id": call.call_id,
                "error": outcome.error,
                "attempts": outcome.attempts,
            })

    def _model_step(self, run_id: str, state: RunState) -> bool:
        """One model call. Returns True if the run reached a terminal state."""
        messages = build_messages(state, self.system_prompt)
        self.store.append(run_id, ev.MODEL_REQUESTED, {
            "call_number": state.model_calls + 1,
            "messages_digest": _digest(messages),
            "message_count": len(messages),
        })

        turn = None
        for attempt in range(self.max_adapter_retries + 1):
            try:
                turn = self.adapter.complete(messages, self.registry.specs_for_model())
                break
            except AdapterError as exc:
                self.store.append(run_id, ev.MODEL_ERROR, {
                    "error": str(exc), "attempt": attempt + 1,
                })
                if attempt < self.max_adapter_retries:
                    self.sleep(self.adapter_backoff * (2 ** attempt))
            except MalformedOutputError as exc:
                self.store.append(run_id, ev.MODEL_RESPONDED, {
                    "malformed": True, "error": str(exc), "raw": exc.raw[:2000],
                })
                malformed = sum(
                    1 for e in self.store.events(run_id)
                    if e.type == ev.MODEL_RESPONDED and e.payload.get("malformed")
                )
                if malformed > self.max_malformed:
                    self.store.append(run_id, ev.RUN_FAILED, {
                        "cause": f"malformed_model_output: {exc}",
                    })
                    return True
                return False  # corrective observation goes back to the model
        if turn is None:
            self.store.append(run_id, ev.RUN_FAILED, {
                "cause": "adapter_error: retries exhausted",
            })
            return True

        self.store.append(run_id, ev.MODEL_RESPONDED, {
            "turn": turn.to_dict(),
            "usage": turn.usage.to_dict(),
            "model": turn.model,
        })
        if turn.is_final:
            self.store.append(run_id, ev.RUN_FINISHED, {"final_answer": turn.text})
            return True
        for tc in turn.tool_calls:
            side_effect = (
                tc["tool"] in self.registry
                and self.registry.get(tc["tool"]).side_effect
            )
            self.store.append(run_id, ev.TOOL_REQUESTED, {
                "call_id": tc["call_id"],
                "tool": tc["tool"],
                "args": tc["args"],
                "side_effect": side_effect,
            })
        return False


def approve_call(store: EventStore, run_id: str, call_id: str,
                 approver: str = "cli") -> None:
    _assert_pending(store, run_id, call_id)
    store.append(run_id, ev.TOOL_APPROVED, {"call_id": call_id, "approver": approver})


def deny_call(store: EventStore, run_id: str, call_id: str,
              reason: str, approver: str = "cli") -> None:
    _assert_pending(store, run_id, call_id)
    store.append(run_id, ev.TOOL_DENIED, {
        "call_id": call_id, "approver": approver, "reason": reason,
    })


def _assert_pending(store: EventStore, run_id: str, call_id: str) -> None:
    state = project(store.events(run_id))
    pending = {c.call_id for c in state.pending_approvals()}
    if call_id not in pending:
        raise ValueError(f"call {call_id} is not pending approval in run {run_id}")
