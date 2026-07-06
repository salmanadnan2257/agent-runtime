"""Minimal FastAPI UI: run list, run timeline, approval queue with diffs."""

from __future__ import annotations

import html
import json
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from ..accounting import account
from ..bootstrap import resume_run
from ..projection import project
from ..runtime import approve_call, deny_call
from ..store import EventStore
from .preview import preview

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:64rem;
     padding:0 1rem;color:#1a1a1a;background:#fafafa}
h1{font-size:1.3rem}a{color:#0b5cad}
table{border-collapse:collapse;width:100%;background:#fff;margin:.5rem 0}
td,th{border:1px solid #ddd;padding:.35rem .6rem;font-size:.85rem;
      text-align:left;vertical-align:top}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:.6rem;
       font-size:.75rem;font-weight:600}
.finished{background:#d3f3df;color:#1a7f37}.failed{background:#fde2dd;color:#b42318}
.waiting_approval{background:#fdf0d3;color:#8a6100}.running{background:#e3ecfd;color:#1d4fa1}
pre{background:#f2f2f2;padding:.5rem;overflow-x:auto;font-size:.78rem;margin:.2rem 0}
.diff{background:#0e1116;color:#d0d7de}
form{display:inline}button{padding:.25rem .8rem;margin-right:.4rem;cursor:pointer}
.approve{background:#d3f3df;border:1px solid #1a7f37}
.deny{background:#fde2dd;border:1px solid #b42318}
.note{color:#666;font-size:.78rem}
input[type=text]{padding:.25rem;font-size:.8rem;width:16rem}
"""


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}"
        f"<p class='note'><a href='/'>all runs</a></p></body></html>")


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="agentrun", docs_url=None, redoc_url=None)

    def store() -> EventStore:
        return EventStore(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        st = store()
        try:
            rows = []
            for rid in st.run_ids():
                s = project(st.events(rid))
                costs = account(s.events)
                sim = " (simulated)" if costs.simulated else ""
                rows.append(
                    f"<tr><td><a href='/runs/{rid}'>{rid}</a></td>"
                    f"<td>{html.escape(s.agent)}</td>"
                    f"<td><span class='badge {s.status}'>{s.status}</span></td>"
                    f"<td>{len(s.events)}</td>"
                    f"<td>${costs.cost_usd:.4f}{sim}</td>"
                    f"<td>{html.escape(s.request[:80])}</td></tr>")
            body = ("<table><tr><th>run</th><th>agent</th><th>status</th>"
                    "<th>events</th><th>cost</th><th>request</th></tr>"
                    + "".join(rows) + "</table>") if rows else "<p>no runs yet</p>"
            return _page("agentrun: runs", body)
        finally:
            st.close()

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_view(run_id: str) -> HTMLResponse:
        st = store()
        try:
            evts = st.events(run_id)
            if not evts:
                raise HTTPException(404, f"unknown run: {run_id}")
            s = project(evts)
            costs = account(evts)
            ws = Path(s.meta.get("workspace", "."))

            parts = [
                f"<p><span class='badge {s.status}'>{s.status}</span> "
                f"agent <b>{html.escape(s.agent)}</b> &middot; "
                f"{costs.model_calls} model calls &middot; "
                f"{costs.input_tokens}+{costs.output_tokens} tokens &middot; "
                f"${costs.cost_usd:.4f}"
                + (" <span class='note'>(simulated usage)</span>"
                   if costs.simulated else "")
                + f" &middot; avg latency {costs.avg_latency_ms:.0f} ms</p>",
                f"<p class='note'>request: {html.escape(s.request)}</p>",
            ]

            pending = s.pending_approvals()
            if pending:
                parts.append("<h2>pending approvals</h2>")
                for call in pending:
                    diff = preview(call.tool, call.args, ws)
                    parts.append(
                        f"<table><tr><th>{html.escape(call.tool)} "
                        f"<span class='note'>{call.call_id}</span></th></tr>"
                        f"<tr><td><pre class='diff'>{html.escape(diff)}</pre>"
                        f"<form method='post' "
                        f"action='/runs/{run_id}/calls/{call.call_id}/approve'>"
                        f"<button class='approve'>approve</button></form>"
                        f"<form method='post' "
                        f"action='/runs/{run_id}/calls/{call.call_id}/deny'>"
                        f"<input type='text' name='reason' placeholder='reason'>"
                        f"<button class='deny'>deny</button></form>"
                        f"</td></tr></table>")

            parts.append("<h2>timeline</h2><table>"
                         "<tr><th>#</th><th>event</th><th>detail</th></tr>")
            for e in evts:
                p = dict(e.payload)
                detail = html.escape(json.dumps(p, sort_keys=True)[:400])
                parts.append(f"<tr><td>{e.seq}</td><td>{e.type}</td>"
                             f"<td><pre>{detail}</pre></td></tr>")
            parts.append("</table>")
            if s.final_answer:
                parts.append(f"<h2>final answer</h2>"
                             f"<pre>{html.escape(s.final_answer)}</pre>")
            return _page(f"run {run_id}", "".join(parts))
        finally:
            st.close()

    @app.post("/runs/{run_id}/calls/{call_id}/approve")
    def approve(run_id: str, call_id: str) -> RedirectResponse:
        st = store()
        try:
            try:
                approve_call(st, run_id, call_id, approver="web")
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            resume_run(st, run_id)
        finally:
            st.close()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/calls/{call_id}/deny")
    def deny(run_id: str, call_id: str, reason: str = Form("")) -> RedirectResponse:
        st = store()
        try:
            try:
                deny_call(st, run_id, call_id,
                          reason=reason or "denied via web", approver="web")
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            resume_run(st, run_id)
        finally:
            st.close()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    return app
