"""Config-driven failure injection.

A FaultPlan describes faults to fire at specific points: model calls
(malformed output, provider 500s) and tool executions (timeouts,
exceptions). Plans are plain dicts so they can live in YAML:

    model:
      - {at_call: 2, kind: malformed}
      - {at_call: 3, kind: adapter_500}
    tools:
      http_get:
        - {at_attempt: 1, kind: timeout}
        - {at_attempt: 2, kind: exception, message: "connection reset"}

Tool fault entries fire on matching attempt numbers, so a fault at
attempt 1 with a clean attempt 2 exercises the retry path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools.executor import ToolTimeout

MODEL_FAULTS = ("malformed", "adapter_500")
TOOL_FAULTS = ("timeout", "exception")


@dataclass
class FaultPlan:
    model: list[dict[str, Any]] = field(default_factory=list)
    tools: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FaultPlan":
        data = data or {}
        plan = cls(model=list(data.get("model", [])),
                   tools={k: list(v) for k, v in data.get("tools", {}).items()})
        for entry in plan.model:
            if entry.get("kind") not in MODEL_FAULTS:
                raise ValueError(f"unknown model fault: {entry.get('kind')}")
        for name, entries in plan.tools.items():
            for entry in entries:
                if entry.get("kind") not in TOOL_FAULTS:
                    raise ValueError(
                        f"unknown tool fault for {name}: {entry.get('kind')}")
        return plan

    def model_fault(self, call_number: int) -> str | None:
        for entry in self.model:
            if entry.get("at_call") == call_number:
                return entry["kind"]
        return None

    def tool_fault(self, tool_name: str, attempt: int) -> Exception | None:
        """Used as the executor's fault_hook."""
        for entry in self.tools.get(tool_name, []):
            if entry.get("at_attempt") == attempt:
                if entry["kind"] == "timeout":
                    return ToolTimeout(f"injected timeout on {tool_name}")
                return RuntimeError(entry.get("message", "injected tool exception"))
        return None
