# agent-runtime

An event-sourced runtime for tool-using LLM agents, with the operations
layer that agent demos usually skip: deterministic replay, crash-safe
resume, an approval gate for side effects, config-driven failure
injection, and a YAML scenario suite that regression-tests agent behavior
between model versions.

## Why

The agent loop itself (plan, call a tool, observe, continue) is a
weekend's work. What makes agents deployable is everything around it: what
exactly happened in run 4187, can I reproduce it, did the agent email a
customer before anyone approved it, and does the new model version still
pass the 24 flows that worked last week. This project treats those as the
core problem. Every run is an append-only event log, and every feature
(replay, resume, approvals, cost accounting, regression diffs) is a
function of that log.

## Features

- **Agent loop over a ModelAdapter interface.** Three adapters: a
  deterministic `MockModelAdapter` scripted from scenario fixtures, plus
  `AnthropicAdapter` (Messages API) and `OpenAIAdapter` (Chat Completions),
  both env-key-driven.
- **Typed tool registry.** Tools declare JSON-schema parameters, a
  `side_effect` flag, timeout, retries with exponential backoff. Shipped
  tools: sandboxed filesystem read/write/list, allowlisted HTTP GET, CSV
  updater, email drafter (writes `.eml` files, never sends), SQLite
  calendar.
- **Event sourcing.** Append-only SQLite log per run; the projection in
  `projection.py` is the single interpreter of what a log means.
- **Deterministic replay.** `agentrun replay <run-id>` re-drives the loop
  purely from recorded model turns, tool results, approvals and
  timestamps. No network. The result is compared byte for byte;
  `--until <seq>` gives stepwise state for debugging.
- **Checkpoint/resume without duplicate side effects.** Idempotency keys
  derived from the log let a run killed mid-tool-execution resume cleanly;
  tests simulate the crash at the worst possible instant.
- **Approval queue.** Side-effecting calls pause the run. Approve or deny
  from the CLI (`agentrun approvals list|approve|deny`) or a small FastAPI
  UI that renders the run timeline and diff-style previews of proposed
  actions. Denials go back to the model as observations.
- **Failure injection.** Fault plans (tool timeout, tool exception,
  malformed model output, provider 500s) prove the loop degrades cleanly:
  retries, error observations, a `run_failed` with a cause, never an
  unhandled exception.
- **Scenario regression suite.** 24 YAML scenarios across 3 example agents
  (invoice-chasing ops assistant, data-entry agent, research summarizer).
  `agentrun test scenarios/ --behavior v1 --compare v2` prints pass/fail,
  a first-divergence diff per scenario ("expected tool_requested:read_file,
  got tool_requested:draft_email at event 3"), and writes an HTML report.
- **Cost and latency accounting** per run from adapter usage metadata. The
  mock adapter fabricates deterministic usage numbers, labeled "simulated"
  in the CLI and UI.

Honest scope note: the Anthropic and OpenAI adapters are implemented
against their current APIs and covered by offline contract tests (request
building, response parsing, error mapping via injected HTTP transports),
but **no live API calls were made here because no API keys were
available**. Everything else, including the full test suite, the CLI and
the web UI, runs and was verified entirely offline.

## Architecture in one paragraph

`store.py` is an append-only SQLite event table plus an idempotency
ledger. `runtime.py` drives the loop as a fixed-point iteration: project
the log, resolve pending tool calls (through the approval gate for side
effects), otherwise ask the adapter for the next turn, append what
happened, repeat. Because pending work is always re-derived from the log,
crash recovery and post-approval continuation are the normal code path,
not special cases. Replay swaps in recorded stand-ins for the clock,
adapter, executor and approval policy and lets the same loop run. Details
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); scenario authoring in
[docs/WRITING_SCENARIOS.md](docs/WRITING_SCENARIOS.md).

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # provides the `agentrun` command
pip install pytest        # for the test suite
pytest                    # 93 tests, all offline
```

Copy `.env.example` to `.env` and fill in keys only if you want live
adapter runs; nothing else reads them.

## Usage

```bash
# Run a scenario through the mock adapter, pausing at the approval gate
agentrun run --scenario scenarios/ops_assistant/chase_overdue_acme.yaml \
             --workspace ./ws
# -> run 1f0c2a...: waiting_approval
#    pending approval: call-2-0 -> draft_email {...}

agentrun approvals list
agentrun approvals approve <run-id> <call-id>     # run continues, finishes
agentrun approvals deny <run-id> <call-id> --reason "wrong recipient"

# Inspect and replay
agentrun runs list
agentrun runs show <run-id>                        # full event log
agentrun replay <run-id>                           # byte-identical: yes
agentrun replay <run-id> --until 6                 # state mid-run

# Resume an interrupted run (no duplicated side effects)
agentrun resume <run-id>

# Regression-test agent behavior and diff two versions
agentrun test scenarios/ --behavior v1 --compare v2 --html report.html

# Web UI: timelines, approval queue with diff previews
agentrun serve --port 8321
```

## Challenges

- **Byte-identical replay fought every source of nondeterminism.** Wall
  clocks, retry sleeps, and "request sent at" metadata all leak into an
  event log. The fix was structural: the store takes an injectable clock
  (replay feeds back recorded timestamps in append order), sleeps are
  injectable no-ops, and `model_requested` records a digest of the
  messages rather than anything time-dependent. The replay comparator then
  gets to demand exact equality, and `tests/test_replay.py` holds it to
  that across happy paths, denials, faults and failed runs.
- **The mock adapter's script index and its call counter had to be
  different numbers.** First design used one counter for both, so an
  injected 500 on call 1 silently consumed script turn 1 and every later
  turn was off by one. Splitting `calls` (fault matching) from `script_i`
  (script consumption) fixed fault plans, and turned out to be exactly
  what resume needed: reposition `script_i` by counting parsed
  `model_responded` events, `calls` by counting all adapter outcomes.
- **Closing the duplicate-side-effect window took ordering, not locks.**
  A crash between "handler ran" and "tool_executed appended" is where
  resume would naively re-execute. The executor writes the result to an
  `INSERT OR IGNORE` ledger keyed by `(run_id, tool_requested seq)` before
  the runtime appends the event; resume finds the ledger entry and emits
  `tool_executed` with `replayed: true`. `test_resume.py` crashes a run at
  precisely that point (via an executor wrapper that raises after
  execution) and asserts a side-effect counter stays at 1.
- **Replay with `--until` ends where the log ends, and the loop hates
  that.** A truncated log leaves the runtime mid-iteration asking for a
  model turn that was never recorded. Rather than special-casing the loop,
  the recorded clock and adapter raise when exhausted and replay treats
  that as the stop condition; whatever was appended by then is exactly the
  state at that seq, which the test verifies.
- **The first full-suite run failed with 5 errors in the web tests only:**
  FastAPI's `Form` parsing needs `python-multipart`, which nothing else
  imports, so every non-web test passed and the gap only surfaced when the
  deny-with-reason endpoint was exercised through `TestClient`. Pinned it
  as a real dependency rather than a test extra since the deny form is a
  runtime feature.
- **Writing the exhaustive project documentation honestly.** Getting the documentation right meant not trusting the README's own numbers at face value: cross-checking the stated test and scenario counts against the actual files with wc and grep caught a real discrepancy (the scenario count was off by one) before it went into the PDF.

## What I learned

- Deriving all state from a projection collapses feature count: resume
  after crash, continue after approval, and a plain next loop iteration
  became literally the same function, and the recovery tests exercise the
  normal path rather than a bespoke one.
- Determinism is a budget you spend field by field. Any payload value not
  a pure function of the log (a timestamp, an attempt count that depends
  on real sleeps, a random id) is a future replay divergence; designing
  each event payload meant asking "can replay recompute this?" every time.
- Separating expected tool failures (`ToolError`) from unexpected ones
  buys a retry policy for free: deterministic refusals (sandbox escape,
  allowlist miss, missing file) fail fast, while transient exceptions and
  timeouts consume the retry budget with backoff.
- `concurrent.futures` timeouts abandon the worker thread rather than kill
  it; a timeout is an answer to "how long do I wait", not "how do I stop
  the tool". That limitation is documented rather than papered over.
- A per-event "type:detail" signature is enough for useful regression
  diffs; comparing two signature lists gives "first divergence at event N,
  expected X, got Y" with no diffing library.

## What I'd do differently

- Approvals via the web UI resume the run synchronously inside the request
  handler. A slow tool blocks the HTTP worker for its whole timeout. A real
  deployment needs a worker process consuming a resume queue, with the UI
  only appending decision events.
- Resume repositions the mock adapter by counting events. It is correct
  for the states the tests cover, but a run resumed in the middle of an
  injected model-fault sequence could realign fault triggers off by one.
  The honest fix is recording the adapter cursor in a checkpoint payload
  instead of inferring it.
- Event payloads have no schema version. Replay detects divergence when
  the runtime changes, which is a tripwire, not a migration story; old
  logs should carry a version field and projectors per version.
- Tool timeouts do not actually kill the handler thread (see above), so a
  truly hung tool leaks a thread per attempt. Subprocess isolation for
  tools would fix both this and the fact that tools currently share the
  runtime's memory space.
- The in-house neutral message format keeps adapters thin, but it ignores
  streaming and provider-specific parallel tool-call semantics. The price
  table in `accounting.py` is hardcoded and will drift.
