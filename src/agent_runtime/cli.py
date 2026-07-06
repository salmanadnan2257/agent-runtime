"""agentrun command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .accounting import account
from .agents import AGENTS, get_agent
from .bootstrap import DEFAULT_DB, resume_run, start_scenario_run
from .projection import project
from .replay import replay_run, replay_state
from .runtime import approve_call as _approve, deny_call as _deny
from .scenarios.loader import Scenario, load_pack
from .scenarios.report import html_report, terminal_diff, terminal_report
from .scenarios.runner import run_pack
from .store import EventStore


@click.group()
@click.option("--db", default=DEFAULT_DB, show_default=True,
              help="Path to the event store database.")
@click.pass_context
def main(ctx: click.Context, db: str) -> None:
    """Event-sourced agent runtime: run, replay, approve, regression-test."""
    ctx.obj = db


def _store(ctx: click.Context) -> EventStore:
    return EventStore(ctx.obj)


def _print_state(state) -> None:
    click.echo(f"run {state.run_id}: {state.status}")
    if state.status == "waiting_approval":
        for call in state.pending_approvals():
            click.echo(f"  pending approval: {call.call_id} -> {call.tool} "
                       f"{call.args}")
        click.echo(f"  approve with: agentrun approvals approve {state.run_id} "
                   f"<call-id>")
    if state.final_answer:
        click.echo(f"  final answer: {state.final_answer}")
    if state.failure_cause:
        click.echo(f"  failure cause: {state.failure_cause}")


@main.command()
@click.option("--scenario", "scenario_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Scenario YAML providing the request and scripted model.")
@click.option("--behavior", default="v1", show_default=True)
@click.option("--workspace", default="workspace", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--auto-approve", is_flag=True,
              help="Approve side-effecting tools without pausing.")
@click.pass_context
def run(ctx: click.Context, scenario_path: str, behavior: str,
        workspace: str, auto_approve: bool) -> None:
    """Start a run with the mock adapter, scripted by a scenario file."""
    store = _store(ctx)
    try:
        scenario = Scenario.from_file(scenario_path)
        policy = (lambda call: ("approve", "policy:auto")) if auto_approve else None
        state = start_scenario_run(store, scenario, behavior,
                                   Path(workspace), policy)
        costs = account(state.events)
        _print_state(state)
        sim = " (simulated)" if costs.simulated else ""
        click.echo(f"  cost: ${costs.cost_usd:.4f}{sim}, "
                   f"{costs.input_tokens}+{costs.output_tokens} tokens, "
                   f"avg latency {costs.avg_latency_ms:.0f} ms")
    finally:
        store.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def resume(ctx: click.Context, run_id: str) -> None:
    """Resume an interrupted run from its event log."""
    store = _store(ctx)
    try:
        _print_state(resume_run(store, run_id))
    finally:
        store.close()


@main.command()
@click.argument("run_id")
@click.option("--until", type=int, default=None,
              help="Replay only up to this event sequence number.")
@click.pass_context
def replay(ctx: click.Context, run_id: str, until: int | None) -> None:
    """Re-execute a run from recorded events. No network, no tools."""
    store = _store(ctx)
    try:
        result = replay_run(store, _registry_for(store, run_id), run_id,
                            _system_prompt_for(store, run_id), until=until)
        click.echo(f"replayed {result.replayed_events}/{result.original_events} "
                   f"events into {result.replay_run_id}")
        if result.identical:
            click.echo("byte-identical: yes")
        else:
            click.echo(f"byte-identical: NO, first divergence at event "
                       f"{result.first_divergence}")
            sys.exit(1)
        if until is not None:
            state = replay_state(store, run_id, until=until)
            click.echo(f"state at event {until}: status={state.status}, "
                       f"model_calls={state.model_calls}, "
                       f"tools_executed={state.executed_tools()}")
    finally:
        store.close()


def _registry_for(store: EventStore, run_id: str):
    from .tools.builtin import build_default_registry

    state = project(store.events(run_id))
    return build_default_registry().subset(list(get_agent(state.agent).tools))


def _system_prompt_for(store: EventStore, run_id: str) -> str:
    state = project(store.events(run_id))
    return get_agent(state.agent).system_prompt


@main.group()
def runs() -> None:
    """Inspect runs."""


@runs.command("list")
@click.pass_context
def runs_list(ctx: click.Context) -> None:
    store = _store(ctx)
    try:
        for rid in store.run_ids():
            s = project(store.events(rid))
            c = account(s.events)
            sim = " simulated" if c.simulated else ""
            click.echo(f"{rid}  {s.status:<17} {s.agent:<20} "
                       f"{len(s.events):>3} events  ${c.cost_usd:.4f}{sim}  "
                       f"{s.request[:50]}")
    finally:
        store.close()


@runs.command("show")
@click.argument("run_id")
@click.pass_context
def runs_show(ctx: click.Context, run_id: str) -> None:
    store = _store(ctx)
    try:
        evts = store.events(run_id)
        if not evts:
            raise click.ClickException(f"unknown run: {run_id}")
        for e in evts:
            click.echo(e.canonical())
        _print_state(project(evts))
    finally:
        store.close()


@main.group()
def approvals() -> None:
    """List, approve or deny pending side-effecting tool calls."""


@approvals.command("list")
@click.pass_context
def approvals_list(ctx: click.Context) -> None:
    store = _store(ctx)
    try:
        found = False
        for rid in store.run_ids():
            s = project(store.events(rid))
            for call in s.pending_approvals():
                found = True
                click.echo(f"{rid}  {call.call_id}  {call.tool}  {call.args}")
        if not found:
            click.echo("no pending approvals")
    finally:
        store.close()


@approvals.command("approve")
@click.argument("run_id")
@click.argument("call_id")
@click.pass_context
def approvals_approve(ctx: click.Context, run_id: str, call_id: str) -> None:
    store = _store(ctx)
    try:
        try:
            _approve(store, run_id, call_id, approver="cli")
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        _print_state(resume_run(store, run_id))
    finally:
        store.close()


@approvals.command("deny")
@click.argument("run_id")
@click.argument("call_id")
@click.option("--reason", default="denied via cli", show_default=True)
@click.pass_context
def approvals_deny(ctx: click.Context, run_id: str, call_id: str,
                   reason: str) -> None:
    store = _store(ctx)
    try:
        try:
            _deny(store, run_id, call_id, reason=reason, approver="cli")
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        _print_state(resume_run(store, run_id))
    finally:
        store.close()


@main.command()
@click.argument("pack", type=click.Path(exists=True))
@click.option("--behavior", default="v1", show_default=True,
              help="Behavior version to test.")
@click.option("--compare", default=None,
              help="Second behavior version to diff against.")
@click.option("--html", "html_path", type=click.Path(dir_okay=False),
              default=None, help="Also write an HTML report here.")
def test(pack: str, behavior: str, compare: str | None,
         html_path: str | None) -> None:
    """Run a scenario pack and report pass/fail (and a version diff)."""
    scenarios = load_pack(pack)
    results_a = run_pack(scenarios, behavior)
    click.echo(terminal_report(behavior, results_a))
    results_b = None
    if compare:
        results_b = run_pack(scenarios, compare)
        click.echo("")
        click.echo(terminal_report(compare, results_b))
        click.echo("")
        click.echo(terminal_diff(behavior, compare, results_a, results_b))
    if html_path:
        Path(html_path).write_text(
            html_report(behavior, results_a, compare, results_b),
            encoding="utf-8")
        click.echo(f"\nhtml report: {html_path}")
    failed = [r for r in results_a if not r.passed]
    if compare and results_b:
        failed += [r for r in results_b if not r.passed]
    if failed:
        sys.exit(1)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8321, show_default=True)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Serve the approval / timeline web UI."""
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(ctx.obj), host=host, port=port, log_level="warning")


@main.command()
def agents() -> None:
    """List available agent definitions."""
    for name, a in AGENTS.items():
        click.echo(f"{name}: tools = {', '.join(a.tools)}")


if __name__ == "__main__":
    main()
