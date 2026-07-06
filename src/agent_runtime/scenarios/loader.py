"""Load and validate scenario YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

SCENARIO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "agent": {"type": "string"},
        "request": {"type": "string", "minLength": 1},
        "workspace_files": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "approvals": {
            "oneOf": [
                {"type": "string", "enum": ["auto"]},
                {
                    "type": "object",
                    "properties": {
                        "deny": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["deny"],
                    "additionalProperties": False,
                },
            ]
        },
        "faults": {"type": "object"},
        "behaviors": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string"},
                                    "args": {"type": "object"},
                                },
                                "required": ["tool"],
                                "additionalProperties": False,
                            },
                        },
                        "text": {"type": "string"},
                        "final": {"type": "string"},
                        "malformed": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "expect": {
            "type": "object",
            "properties": {
                "status": {"type": "string",
                           "enum": ["finished", "failed", "waiting_approval"]},
                "tools_executed": {"type": "array", "items": {"type": "string"}},
                "tools_executed_contains": {"type": "array",
                                            "items": {"type": "string"}},
                "tools_not_executed": {"type": "array", "items": {"type": "string"}},
                "denied_tools": {"type": "array", "items": {"type": "string"}},
                "final_contains": {"type": "array", "items": {"type": "string"}},
                "final_not_contains": {"type": "array", "items": {"type": "string"}},
                "failure_cause_contains": {"type": "string"},
                "files_exist": {"type": "array", "items": {"type": "string"}},
                "min_events": {"type": "integer", "minimum": 1},
                "no_side_effect_before_approval": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["name", "agent", "request", "behaviors", "expect"],
    "additionalProperties": False,
}


@dataclass
class Scenario:
    name: str
    agent: str
    request: str
    behaviors: dict[str, list[dict[str, Any]]]
    expect: dict[str, Any]
    workspace_files: dict[str, str] = field(default_factory=dict)
    approvals: Any = "auto"
    faults: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "Scenario":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=SCENARIO_SCHEMA)
        for version, turns in data["behaviors"].items():
            for i, turn in enumerate(turns):
                keys = {"tool_calls", "final", "malformed"} & set(turn)
                if len(keys) != 1:
                    raise ValueError(
                        f"{path}: behaviors.{version}[{i}] must have exactly one "
                        f"of tool_calls / final / malformed")
        return cls(
            name=data["name"],
            agent=data["agent"],
            request=data["request"],
            behaviors=data["behaviors"],
            expect=data["expect"],
            workspace_files=data.get("workspace_files", {}),
            approvals=data.get("approvals", "auto"),
            faults=data.get("faults", {}),
            path=str(path),
        )


def load_pack(directory: str | Path) -> list[Scenario]:
    root = Path(directory)
    if root.is_file():
        return [Scenario.from_file(root)]
    files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    if not files:
        raise FileNotFoundError(f"no scenario YAML files under {root}")
    scenarios = [Scenario.from_file(f) for f in files]
    names = [s.name for s in scenarios]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate scenario names: {sorted(dupes)}")
    return scenarios
