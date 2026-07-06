"""Typed tool registry.

Each tool declares a JSON-schema for its arguments, whether it has side
effects (which routes it through the approval gate), a timeout and a
retry budget. Handlers receive validated args plus a ToolContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jsonschema


class ToolError(Exception):
    """Raised by handlers for expected failures (bad path, denied host)."""


@dataclass
class ToolContext:
    """Per-run environment handed to tool handlers."""

    workspace: Path
    http_allowlist: tuple[str, ...] = ()
    http_client: Any = None  # injectable for tests; None means real httpx

    def resolve(self, rel_path: str) -> Path:
        """Resolve a path inside the workspace sandbox, rejecting escapes."""
        candidate = (self.workspace / rel_path).resolve()
        root = self.workspace.resolve()
        if candidate != root and root not in candidate.parents:
            raise ToolError(f"path escapes workspace sandbox: {rel_path}")
        return candidate


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for args
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]
    side_effect: bool = False
    timeout: float = 10.0
    retries: int = 2
    backoff: float = 0.5  # base seconds, doubled per retry

    def validate_args(self, args: dict[str, Any]) -> None:
        jsonschema.validate(instance=args, schema=self.parameters)

    def public(self) -> dict[str, Any]:
        """Shape given to model adapters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        jsonschema.Draft202012Validator.check_schema(spec.parameters)
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def specs_for_model(self) -> list[dict[str, Any]]:
        return [t.public() for t in self.tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        return ToolRegistry(tools={n: self.get(n) for n in names})
