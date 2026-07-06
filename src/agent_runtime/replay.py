"""Deterministic replay of a recorded run.

Replay rebuilds the run using only its event log: recorded model turns,
recorded tool outcomes, recorded approvals, recorded timestamps. No
adapter, no tool handler and no network is touched. If the runtime is
deterministic, the replayed log is byte-identical to the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import events as ev
from .adapters.base import AdapterError, MalformedOutputError, ModelAdapter, ModelTurn, Usage
from .events import Event, canonical_log
from .projection import ToolCallState, project
from .runtime import ApprovalDecision, Runtime
from .store import EventStore
from .tools.executor import ExecutionOutcome
from .tools.registry import ToolRegistry, ToolSpec


class _RecordedClock:
    """Replays the original timestamps in append order."""

    def __init__(self, timestamps: list[float]):
        self._ts = list(timestamps)
        self._i = 0

    def __call__(self) -> float:
        if self._i >= len(self._ts):
            raise RuntimeError("replay produced more events than the original run")
        t = self._ts[self._i]
        self._i += 1
        return t


class _RecordedAdapter(ModelAdapter):
    """Yields the recorded outcome of each adapter attempt, in order."""

    name = "replay"

    def __init__(self, source_events: list[Event]):
        self._queue: list[tuple[str, dict[str, Any]]] = []
        for e in source_events:
            if e.type == ev.MODEL_ERROR:
                self._queue.append(("error", e.payload))
            elif e.type == ev.MODEL_RESPONDED:
                kind = "malformed" if e.payload.get("malformed") else "turn"
                self._queue.append((kind, e.payload))

    def complete(self, messages, tools) -> ModelTurn:
        if not self._queue:
            raise RuntimeError("replay adapter exhausted: log has no more model turns")
        kind, payload = self._queue.pop(0)
        if kind == "error":
            raise AdapterError(payload["error"])
        if kind == "malformed":
            raise MalformedOutputError(payload["error"], raw=payload.get("raw", ""))
        turn = payload["turn"]
        u = payload["usage"]
        return ModelTurn(
            text=turn.get("text", ""),
            tool_calls=turn.get("tool_calls", []),
            usage=Usage(**u),
            model=payload.get("model", ""),
        )


class _RecordedExecutor:
    """Returns recorded tool outcomes keyed by the tool_requested seq."""

    def __init__(self, source_events: list[Event]):
        seq_by_call: dict[str, int] = {}
        self._outcomes: dict[int, ExecutionOutcome] = {}
        for e in source_events:
            p = e.payload
            if e.type == ev.TOOL_REQUESTED:
                seq_by_call[p["call_id"]] = e.seq
            elif e.type == ev.TOOL_EXECUTED:
                self._outcomes[seq_by_call[p["call_id"]]] = ExecutionOutcome(
                    ok=True, result=p["result"], attempts=p["attempts"],
                    replayed=p.get("replayed", False),
                )
            elif e.type == ev.TOOL_FAILED:
                self._outcomes[seq_by_call.get(p["call_id"], -1)] = ExecutionOutcome(
                    ok=False, error=p["error"], attempts=p.get("attempts", 0),
                )

    def execute(self, spec: ToolSpec, args: dict[str, Any], run_id: str,
                requested_seq: int) -> ExecutionOutcome:
        if requested_seq not in self._outcomes:
            raise RuntimeError(
                f"replay has no recorded outcome for tool call at seq {requested_seq}")
        return self._outcomes[requested_seq]


class _RecordedApprovals:
    """Replays approve/deny decisions exactly as recorded."""

    def __init__(self, source_events: list[Event]):
        self._decisions: dict[str, ApprovalDecision] = {}
        for e in source_events:
            p = e.payload
            if e.type == ev.TOOL_APPROVED:
                self._decisions[p["call_id"]] = ("approve", p["approver"])
            elif e.type == ev.TOOL_DENIED:
                self._decisions[p["call_id"]] = (
                    "deny", p["approver"], p.get("reason", "denied"))

    def __call__(self, call: ToolCallState) -> ApprovalDecision:
        return self._decisions.get(call.call_id)


@dataclass
class ReplayResult:
    replay_run_id: str
    identical: bool
    original_events: int
    replayed_events: int
    first_divergence: int | None  # seq of first differing event, if any
    original_canonical: str
    replay_canonical: str


def replay_run(
    store: EventStore,
    registry: ToolRegistry,
    run_id: str,
    system_prompt: str,
    until: int | None = None,
) -> ReplayResult:
    source = store.events(run_id)
    if not source:
        raise ValueError(f"unknown run: {run_id}")
    if until is not None:
        source = [e for e in source if e.seq <= until]

    replay_id = f"{run_id}-replay-{store.new_run_id()[:6]}"
    clock = _RecordedClock([e.ts for e in source])
    # A store view with the recorded clock, same database file.
    replay_store = EventStore(store.path, clock=clock)
    try:
        rt = Runtime(
            store=replay_store,
            registry=registry,
            adapter=_RecordedAdapter(source),
            executor=_RecordedExecutor(source),  # type: ignore[arg-type]
            system_prompt=system_prompt,
            approval_policy=_RecordedApprovals(source),
            sleep=lambda _s: None,
        )
        created = source[0].payload
        try:
            rt.start(agent=created.get("agent", ""),
                     request=created.get("request", ""),
                     meta=created.get("meta", {}),
                     run_id=replay_id)
        except RuntimeError:
            # A truncated log (--until) can leave the loop without a next
            # recorded turn. Whatever was appended so far IS the state at
            # that point; stop there.
            pass
    finally:
        replay_store.close()

    replayed = store.events(replay_id)
    if until is not None:
        replayed = replayed[: len(source)]
    orig_c = canonical_log(source)
    repl_c = canonical_log(replayed)
    divergence: int | None = None
    if orig_c != repl_c:
        for o, r in zip(source, replayed):
            if o.canonical() != r.canonical():
                divergence = o.seq
                break
        else:
            divergence = min(len(source), len(replayed))
    return ReplayResult(
        replay_run_id=replay_id,
        identical=orig_c == repl_c,
        original_events=len(source),
        replayed_events=len(replayed),
        first_divergence=divergence,
        original_canonical=orig_c,
        replay_canonical=repl_c,
    )


def replay_state(store: EventStore, run_id: str, until: int | None = None):
    """Project the state of a run at (or up to) a given event seq."""
    return project(store.events(run_id, until=until))
