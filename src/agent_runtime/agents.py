"""Example agent definitions: a system prompt plus an allowed tool set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDef:
    name: str
    system_prompt: str
    tools: tuple[str, ...]


AGENTS: dict[str, AgentDef] = {
    "ops_assistant": AgentDef(
        name="ops_assistant",
        system_prompt=(
            "You are an operations assistant. You chase overdue invoices, "
            "draft reminder emails, and keep the invoice sheet current. "
            "Read data before acting. Draft emails, never send them."
        ),
        tools=("read_file", "list_files", "csv_update", "draft_email",
               "calendar_add", "calendar_list"),
    ),
    "data_entry": AgentDef(
        name="data_entry",
        system_prompt=(
            "You are a data-entry agent. You transcribe records into CSV "
            "sheets accurately, one operation at a time, and verify by "
            "reading back what you wrote."
        ),
        tools=("read_file", "list_files", "write_file", "csv_update"),
    ),
    "research_summarizer": AgentDef(
        name="research_summarizer",
        system_prompt=(
            "You are a research summarizer. You fetch allowlisted pages and "
            "local notes, then write a concise summary file with sources."
        ),
        tools=("read_file", "list_files", "http_get", "write_file"),
    ),
}


def get_agent(name: str) -> AgentDef:
    if name not in AGENTS:
        raise KeyError(f"unknown agent: {name} (have: {', '.join(sorted(AGENTS))})")
    return AGENTS[name]
