"""Event types and canonical serialization.

Every run is an append-only sequence of events. The full state of a run
(status, conversation, pending approvals, side effects, cost) is a pure
function of its event log. Nothing else is authoritative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Event types, in the order they typically appear in a run.
RUN_CREATED = "run_created"
MODEL_REQUESTED = "model_requested"
MODEL_RESPONDED = "model_responded"
MODEL_ERROR = "model_error"
TOOL_REQUESTED = "tool_requested"
TOOL_APPROVED = "tool_approved"
TOOL_DENIED = "tool_denied"
TOOL_EXECUTED = "tool_executed"
TOOL_FAILED = "tool_failed"
CHECKPOINT = "checkpoint"
RUN_FINISHED = "run_finished"
RUN_FAILED = "run_failed"

ALL_TYPES = frozenset({
    RUN_CREATED, MODEL_REQUESTED, MODEL_RESPONDED, MODEL_ERROR,
    TOOL_REQUESTED, TOOL_APPROVED, TOOL_DENIED, TOOL_EXECUTED,
    TOOL_FAILED, CHECKPOINT, RUN_FINISHED, RUN_FAILED,
})

TERMINAL_TYPES = frozenset({RUN_FINISHED, RUN_FAILED})


@dataclass(frozen=True)
class Event:
    """One immutable entry in a run's log."""

    run_id: str
    seq: int
    type: str
    ts: float
    payload: dict[str, Any]

    def canonical(self, include_run_id: bool = False) -> str:
        """Deterministic JSON encoding used for replay comparison."""
        body: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type,
            "ts": self.ts,
            "payload": self.payload,
        }
        if include_run_id:
            body["run_id"] = self.run_id
        return json.dumps(body, sort_keys=True, separators=(",", ":"))


def canonical_log(events: list[Event]) -> str:
    """Byte-comparable encoding of a whole log (run id excluded)."""
    return "\n".join(e.canonical() for e in events)
