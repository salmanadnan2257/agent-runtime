"""Deterministic scripted adapter used for offline runs and tests.

The script is a list of turns taken from a scenario fixture. Usage
numbers are fabricated deterministically from message sizes and are
labeled simulated everywhere they surface.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AdapterError, MalformedOutputError, ModelAdapter, ModelTurn, Usage

MOCK_MODEL = "mock-1"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockModelAdapter(ModelAdapter):
    name = "mock"

    def __init__(self, script: list[dict[str, Any]], faults: Any = None):
        """script: list of turn dicts, each either
        {"tool_calls": [{"tool": ..., "args": {...}}, ...]} or
        {"final": "answer text"} or {"malformed": "raw garbage"}.
        faults: optional FaultPlan consulted per model call.
        """
        self.script = script
        self.faults = faults
        self.calls = 0        # every complete() invocation, incl. injected faults
        self.script_i = 0     # script turns actually consumed

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.calls += 1
        if self.faults is not None:
            # Injected faults do not consume a script turn: after the fault
            # clears, the script continues where it left off.
            fault = self.faults.model_fault(self.calls)
            if fault == "adapter_500":
                raise AdapterError("simulated provider HTTP 500")
            if fault == "malformed":
                raise MalformedOutputError(
                    "simulated unparseable model output", raw="{not json"
                )

        if self.script_i >= len(self.script):
            # Scripts must cover every model call; running past the end is a
            # scenario bug, surfaced as a hard error rather than a guess.
            raise MalformedOutputError(
                f"mock script exhausted after {len(self.script)} turns"
            )
        step = self.script[self.script_i]
        self.script_i += 1
        input_text = json.dumps(messages, sort_keys=True, default=str)

        if "malformed" in step:
            raise MalformedOutputError("scripted malformed output", raw=step["malformed"])

        if "tool_calls" in step:
            calls = [
                {
                    "call_id": f"call-{self.calls}-{i}",
                    "tool": tc["tool"],
                    "args": tc.get("args", {}),
                }
                for i, tc in enumerate(step["tool_calls"])
            ]
            out_text = step.get("text", "")
            output_len = len(json.dumps(calls)) + len(out_text)
        else:
            calls = []
            out_text = step["final"]
            output_len = len(out_text)

        usage = Usage(
            input_tokens=_estimate_tokens(input_text),
            output_tokens=max(1, output_len // 4),
            latency_ms=float(180 + 7 * (output_len % 40)),  # deterministic, plausible
            simulated=True,
        )
        return ModelTurn(text=out_text, tool_calls=calls, usage=usage, model=MOCK_MODEL)
