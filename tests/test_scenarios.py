import pytest
import yaml

from agent_runtime.scenarios.loader import Scenario, load_pack
from agent_runtime.scenarios.report import (
    diff_results, html_report, terminal_diff, terminal_report,
)
from agent_runtime.scenarios.runner import run_pack, run_scenario

from conftest import SCENARIO_DIR

EXPECTED_V2_FAILURES = {
    "chase-overdue-acme", "transcribe-two-records", "summarize-notes",
}


@pytest.fixture(scope="module")
def pack():
    return load_pack(SCENARIO_DIR)


@pytest.fixture(scope="module")
def results_v1(pack, tmp_path_factory):
    return run_pack(pack, "v1", workdir=tmp_path_factory.mktemp("v1"))


@pytest.fixture(scope="module")
def results_v2(pack, tmp_path_factory):
    return run_pack(pack, "v2", workdir=tmp_path_factory.mktemp("v2"))


def test_pack_has_24_scenarios_across_3_agents(pack):
    assert len(pack) == 24
    agents = {s.agent for s in pack}
    assert agents == {"ops_assistant", "data_entry", "research_summarizer"}
    for agent in agents:
        assert sum(1 for s in pack if s.agent == agent) == 8


def test_v1_all_pass(results_v1):
    failing = [(r.name, r.failures, r.error) for r in results_v1 if not r.passed]
    assert not failing, failing


def test_v2_fails_exactly_the_known_regressions(results_v2):
    failing = {r.name for r in results_v2 if not r.passed}
    assert failing == EXPECTED_V2_FAILURES


def test_diff_reports_first_divergence(results_v1, results_v2):
    divs = {d.name: d for d in diff_results(results_v1, results_v2)}
    d = divs["chase-overdue-acme"]
    assert d.a_passed and not d.b_passed
    assert d.first_divergence is not None
    assert "read_file" in d.expected and "draft_email" in d.got


def test_terminal_reports_render(results_v1, results_v2):
    text = terminal_report("v1", results_v1)
    assert "24/24 scenarios passed" in text
    diff = terminal_diff("v1", "v2", results_v1, results_v2)
    assert "v2 fails 3 of 24 scenarios" in diff
    assert "first divergence at event" in diff


def test_html_report_renders(results_v1, results_v2):
    doc = html_report("v1", results_v1, "v2", results_v2)
    assert "chase-overdue-acme" in doc
    assert "REGRESSION" in doc
    assert "simulated" in doc


def test_unknown_behavior_version_is_reported(pack):
    r = run_scenario(pack[0], "v99")
    assert not r.passed and "v99" in r.error


def test_loader_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "x", "agent": "a"}))
    with pytest.raises(Exception):
        Scenario.from_file(bad)


def test_loader_rejects_ambiguous_turns(tmp_path):
    bad = tmp_path / "ambiguous.yaml"
    bad.write_text(yaml.safe_dump({
        "name": "x", "agent": "data_entry", "request": "r",
        "behaviors": {"v1": [{"final": "a", "malformed": "b"}]},
        "expect": {"status": "finished"},
    }))
    with pytest.raises(ValueError, match="exactly one"):
        Scenario.from_file(bad)


def test_loader_rejects_duplicate_names(tmp_path):
    doc = yaml.safe_dump({
        "name": "same", "agent": "data_entry", "request": "r",
        "behaviors": {"v1": [{"final": "ok"}]},
        "expect": {"status": "finished"},
    })
    (tmp_path / "a.yaml").write_text(doc)
    (tmp_path / "b.yaml").write_text(doc)
    with pytest.raises(ValueError, match="duplicate scenario names"):
        load_pack(tmp_path)


def test_no_side_effect_before_approval_invariant(results_v1):
    # Every scenario that asserts the invariant passed it; spot-check one
    # scenario's signature contains an approval before execution.
    r = next(x for x in results_v1 if x.name == "chase-overdue-acme")
    sig = r.event_sig
    approved = next(i for i, s in enumerate(sig) if s.startswith("tool_approved"))
    executed = [i for i, s in enumerate(sig) if s.startswith("tool_executed")]
    draft_exec = max(executed)
    assert approved < draft_exec
