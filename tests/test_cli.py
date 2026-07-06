import re

from click.testing import CliRunner

from agent_runtime.cli import main

from conftest import SCENARIO_DIR

SCN = str(SCENARIO_DIR / "ops_assistant" / "chase_overdue_acme.yaml")
SCN_PLAIN = str(SCENARIO_DIR / "research" / "empty_notes_graceful.yaml")


def invoke(tmp_path, *args, **kw):
    runner = CliRunner()
    return runner.invoke(main, ["--db", str(tmp_path / "cli.db"), *args], **kw)


def run_id_from(output: str) -> str:
    return re.search(r"run (\S+):", output).group(1)


def test_run_auto_approve_finishes(tmp_path):
    r = invoke(tmp_path, "run", "--scenario", SCN, "--auto-approve",
               "--workspace", str(tmp_path / "ws"))
    assert r.exit_code == 0, r.output
    assert "finished" in r.output
    assert "INV-1001" in r.output
    assert "simulated" in r.output  # cost line labels mock usage


def test_run_pauses_then_cli_approval_flow(tmp_path):
    r = invoke(tmp_path, "run", "--scenario", SCN,
               "--workspace", str(tmp_path / "ws"))
    assert r.exit_code == 0, r.output
    assert "waiting_approval" in r.output
    rid = run_id_from(r.output)

    lst = invoke(tmp_path, "approvals", "list")
    assert rid in lst.output and "draft_email" in lst.output
    call_id = re.search(rf"{rid}\s+(\S+)\s+draft_email", lst.output).group(1)

    ok = invoke(tmp_path, "approvals", "approve", rid, call_id)
    assert ok.exit_code == 0, ok.output
    assert "finished" in ok.output

    empty = invoke(tmp_path, "approvals", "list")
    assert "no pending approvals" in empty.output


def test_cli_denial_feeds_back(tmp_path):
    scn = str(SCENARIO_DIR / "ops_assistant" / "deny_email_fallback.yaml")
    # Run without auto-approve so the CLI, not scenario policy, decides.
    r = invoke(tmp_path, "run", "--scenario", scn,
               "--workspace", str(tmp_path / "ws"))
    rid = run_id_from(r.output)
    lst = invoke(tmp_path, "approvals", "list")
    call_id = re.search(rf"{rid}\s+(\S+)\s+draft_email", lst.output).group(1)
    denied = invoke(tmp_path, "approvals", "deny", rid, call_id,
                    "--reason", "recipient looks wrong")
    assert denied.exit_code == 0, denied.output
    assert "finished" in denied.output
    assert "denied" in denied.output.lower()


def test_replay_command_byte_identical(tmp_path):
    r = invoke(tmp_path, "run", "--scenario", SCN_PLAIN, "--auto-approve",
               "--workspace", str(tmp_path / "ws"))
    rid = run_id_from(r.output)
    rep = invoke(tmp_path, "replay", rid)
    assert rep.exit_code == 0, rep.output
    assert "byte-identical: yes" in rep.output


def test_replay_until(tmp_path):
    r = invoke(tmp_path, "run", "--scenario", SCN_PLAIN, "--auto-approve",
               "--workspace", str(tmp_path / "ws"))
    rid = run_id_from(r.output)
    rep = invoke(tmp_path, "replay", rid, "--until", "4")
    assert rep.exit_code == 0, rep.output
    assert "state at event 4" in rep.output


def test_runs_list_and_show(tmp_path):
    r = invoke(tmp_path, "run", "--scenario", SCN_PLAIN, "--auto-approve",
               "--workspace", str(tmp_path / "ws"))
    rid = run_id_from(r.output)
    lst = invoke(tmp_path, "runs", "list")
    assert rid in lst.output and "research_summarizer" in lst.output
    shown = invoke(tmp_path, "runs", "show", rid)
    assert shown.exit_code == 0
    assert '"type":"run_created"' in shown.output
    missing = invoke(tmp_path, "runs", "show", "nope")
    assert missing.exit_code != 0


def test_test_command_pass_and_fail_exit_codes(tmp_path):
    ops = str(SCENARIO_DIR / "ops_assistant")
    ok = invoke(tmp_path, "test", ops, "--behavior", "v1")
    assert ok.exit_code == 0, ok.output
    assert "8/8 scenarios passed" in ok.output

    html_path = tmp_path / "report.html"
    cmp = invoke(tmp_path, "test", ops, "--behavior", "v1",
                 "--compare", "v2", "--html", str(html_path))
    assert cmp.exit_code == 1  # v2 has a regression in this pack
    assert "v2 fails 1 of 8 scenarios" in cmp.output
    assert "first divergence at event" in cmp.output
    assert html_path.is_file() and "REGRESSION" in html_path.read_text()


def test_agents_command(tmp_path):
    r = invoke(tmp_path, "agents")
    assert "ops_assistant" in r.output and "draft_email" in r.output
