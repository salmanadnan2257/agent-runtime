"""Terminal and HTML reports for scenario packs and version diffs."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from .runner import ScenarioResult


@dataclass
class Divergence:
    name: str
    a_passed: bool
    b_passed: bool
    first_divergence: int | None  # event index where the logs part ways
    expected: str = ""  # signature of version A at that index
    got: str = ""       # signature of version B at that index
    b_failures: list[str] | None = None


def diff_results(
    a: list[ScenarioResult], b: list[ScenarioResult]
) -> list[Divergence]:
    by_name = {r.name: r for r in b}
    out: list[Divergence] = []
    for ra in a:
        rb = by_name.get(ra.name)
        if rb is None:
            continue
        idx: int | None = None
        expected = got = ""
        for i, (sa, sb) in enumerate(zip(ra.event_sig, rb.event_sig)):
            if sa != sb:
                idx, expected, got = i, sa, sb
                break
        else:
            if len(ra.event_sig) != len(rb.event_sig):
                idx = min(len(ra.event_sig), len(rb.event_sig))
                expected = ra.event_sig[idx] if idx < len(ra.event_sig) else "(end)"
                got = rb.event_sig[idx] if idx < len(rb.event_sig) else "(end)"
        out.append(Divergence(
            name=ra.name, a_passed=ra.passed, b_passed=rb.passed,
            first_divergence=idx, expected=expected, got=got,
            b_failures=rb.failures or ([rb.error] if rb.error else []),
        ))
    return out


def terminal_report(version: str, results: list[ScenarioResult]) -> str:
    lines = [f"behavior {version}: "
             f"{sum(r.passed for r in results)}/{len(results)} scenarios passed"]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.name} ({r.event_count} events, "
                     f"status={r.status}, cost=${r.cost_usd:.4f} simulated)")
        for f in r.failures:
            lines.append(f"         - {f}")
        if r.error:
            lines.append(f"         ! {r.error}")
    return "\n".join(lines)


def terminal_diff(
    version_a: str, version_b: str,
    a: list[ScenarioResult], b: list[ScenarioResult],
) -> str:
    divs = diff_results(a, b)
    regressions = [d for d in divs if d.a_passed and not d.b_passed]
    lines = [
        f"comparing {version_a} -> {version_b}: "
        f"{version_b} fails {sum(1 for r in b if not r.passed)} of {len(b)} scenarios "
        f"({len(regressions)} regressions vs {version_a})"
    ]
    for d in divs:
        if d.a_passed == d.b_passed and d.first_divergence is None:
            continue
        verdict = (
            "REGRESSION" if d.a_passed and not d.b_passed
            else "FIXED" if not d.a_passed and d.b_passed
            else "DIVERGED" if d.first_divergence is not None
            else "BOTH FAIL"
        )
        lines.append(f"  [{verdict}] {d.name}")
        if d.first_divergence is not None:
            lines.append(
                f"      first divergence at event {d.first_divergence}: "
                f"expected {d.expected}, got {d.got}")
        for f in d.b_failures or []:
            lines.append(f"      {version_b} failure: {f}")
    if len(lines) == 1:
        lines.append("  no divergences: both versions behave identically")
    return "\n".join(lines)


_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:60rem;
     padding:0 1rem;color:#1a1a1a;background:#fafafa}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2rem}
table{border-collapse:collapse;width:100%;background:#fff}
td,th{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;
      font-size:.85rem;vertical-align:top}
.pass{color:#1a7f37;font-weight:600}.fail{color:#b42318;font-weight:600}
.mono{font-family:ui-monospace,monospace;font-size:.8rem}
.note{color:#666;font-size:.8rem}
"""


def html_report(
    version_a: str, results_a: list[ScenarioResult],
    version_b: str | None = None, results_b: list[ScenarioResult] | None = None,
) -> str:
    def rows(results: list[ScenarioResult]) -> str:
        out = []
        for r in results:
            cls = "pass" if r.passed else "fail"
            detail = "<br>".join(html.escape(f) for f in r.failures)
            if r.error:
                detail += f"<br>! {html.escape(r.error)}"
            out.append(
                f"<tr><td>{html.escape(r.name)}</td>"
                f"<td class='{cls}'>{'PASS' if r.passed else 'FAIL'}</td>"
                f"<td>{r.status}</td><td>{r.event_count}</td>"
                f"<td>${r.cost_usd:.4f} <span class='note'>simulated</span></td>"
                f"<td>{detail or ''}</td></tr>")
        return "".join(out)

    parts = [
        f"<style>{_CSS}</style>",
        "<h1>agentrun scenario report</h1>",
        f"<h2>behavior {html.escape(version_a)}: "
        f"{sum(r.passed for r in results_a)}/{len(results_a)} passed</h2>",
        "<table><tr><th>scenario</th><th>result</th><th>status</th>"
        "<th>events</th><th>cost</th><th>failures</th></tr>",
        rows(results_a), "</table>",
    ]
    if version_b is not None and results_b is not None:
        parts += [
            f"<h2>behavior {html.escape(version_b)}: "
            f"{sum(r.passed for r in results_b)}/{len(results_b)} passed</h2>",
            "<table><tr><th>scenario</th><th>result</th><th>status</th>"
            "<th>events</th><th>cost</th><th>failures</th></tr>",
            rows(results_b), "</table>",
            f"<h2>divergences {html.escape(version_a)} &rarr; "
            f"{html.escape(version_b)}</h2><table>"
            "<tr><th>scenario</th><th>verdict</th><th>first divergence</th></tr>",
        ]
        for d in diff_results(results_a, results_b):
            if d.a_passed == d.b_passed and d.first_divergence is None:
                continue
            verdict = ("REGRESSION" if d.a_passed and not d.b_passed
                       else "FIXED" if not d.a_passed and d.b_passed else "DIVERGED")
            where = ("" if d.first_divergence is None else
                     f"event {d.first_divergence}: expected "
                     f"<span class='mono'>{html.escape(d.expected)}</span>, got "
                     f"<span class='mono'>{html.escape(d.got)}</span>")
            parts.append(f"<tr><td>{html.escape(d.name)}</td>"
                         f"<td class='fail'>{verdict}</td><td>{where}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def write_html_report(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")
