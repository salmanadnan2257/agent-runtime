"""ModelAdapter interface and shared turn/usage types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class AdapterError(Exception):
    """Transport or provider failure (e.g. HTTP 500). Retryable."""


class MalformedOutputError(Exception):
    """The model produced output the runtime could not parse."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "simulated": self.simulated,
        }


@dataclass
class ModelTurn:
    """One parsed model response: tool calls, or a final text answer."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def is_final(self) -> bool:
        return not self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "tool_calls": self.tool_calls}


class ModelAdapter(ABC):
    """Turns a conversation plus tool specs into the next ModelTurn."""

    name = "base"

    @abstractmethod
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        """May raise AdapterError (transport) or MalformedOutputError (parse)."""
