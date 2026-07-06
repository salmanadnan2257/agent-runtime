# Architecture

## The one rule

Every run is an append-only event log in SQLite. The full state of a run
(status, conversation, pending approvals, executed tools, cost) is a pure
function of that log. Nothing else is authoritative: no in-memory session
object survives a process, and none needs to.

This single decision buys the three features that matter operationally:
crash recovery (rebuild state from the log), deterministic replay (re-drive
the loop from recorded outcomes), and behavioral regression testing (assert
on the log, not on printouts).

## Event model

Table `events(run_id, seq, type, ts, payload)` with `PRIMARY KEY (run_id,
seq)`. There is no UPDATE or DELETE path in the codebase. Appends compute
`MAX(seq)+1` and retry on the unique-constraint violation, so two writers
racing on one run cannot both claim a sequence number.

Event types, in the order a typical run emits them:

| type | payload highlights |
|---|---|
| `run_created` | agent, request, meta (workspace, adapter kind, scenario path) |
| `model_requested` | call number, digest and count of the messages sent |
| `model_responded` | parsed turn (text or tool calls), usage, model; or `malformed: true` with the raw output |
| `model_error` | transport/provider failure per attempt (e.g. HTTP 500) |
| `tool_requested` | call_id, tool, args, side_effect flag |
| `tool_approved` / `tool_denied` | call_id, approver, deny reason |
| `tool_executed` / `tool_failed` | call_id, result or error, attempt count |
| `checkpoint` | model calls and resolved tool calls so far |
| `run_finished` / `run_failed` | final answer, or failure cause |

`projection.py` is the only module that interprets ordering. The runtime,
CLI, web UI, replay comparator and scenario assertions all consume its
`RunState`, so they cannot disagree about what a log means.

## The loop

`Runtime.drive(run_id)` is a fixed point iteration over the log:

1. Project the log. Terminal status: stop.
2. Resolve unfinished tool calls in request order. A side-effecting call
   with no decision either gets one from the approval policy or pauses the
   run (`waiting_approval`); read-only calls execute directly.
3. If nothing is pending, append `model_requested`, call the adapter,
   append what happened, and loop.

Because step 2 re-derives pending work from the log, "resume after crash"
and "continue after approval" are the same code path as a normal iteration.
There is no special-case recovery logic.

## Conversation reconstruction

`build_messages()` folds the log into a provider-neutral message list:
system prompt, user request, assistant turns, tool observations. Denials
become tool observations (`{"denied": true, "reason": ...}`), which is how
"the operator said no" reaches the model as feedback. Malformed model
output becomes a corrective user message. Adapters translate this neutral
format into the Anthropic Messages or OpenAI Chat Completions wire shape.

## Idempotency and the crash window

Side effects must not repeat when a run is resumed. The executor derives an
idempotency key from `(run_id, seq of the tool_requested event)`, both of
which are identical on resume. Execution order is:

1. run the handler,
2. write the result into the `executions` ledger (`INSERT OR IGNORE`),
3. the runtime appends `tool_executed`.

If the process dies between 2 and 3 (the worst window), the resumed run
finds the ledger entry and appends `tool_executed` with `replayed: true`
without touching the handler. `tests/test_resume.py` simulates exactly this
crash point and asserts a side-effect counter stays at 1.

## Deterministic replay

`agentrun replay <run-id>` rebuilds a run using four recorded stand-ins:

- a clock that replays the original timestamps in append order,
- an adapter that replays recorded turns, malformed outputs and transport
  errors, in order,
- an executor that returns recorded outcomes keyed by `tool_requested` seq,
- an approval policy that replays recorded approve/deny decisions.

The real runtime then drives the loop normally. No network, no tool
handlers, no key needed. The replayed log is compared to the original via a
canonical JSON encoding of every event; any divergence reports the first
differing seq. `--until <seq>` truncates the source log for stepwise
debugging and also prints the projected state at that point.

This doubles as a regression tripwire: if a code change alters the loop's
event sequence, replaying an old run diverges and says where.

## Failure injection

`FaultPlan` is a plain dict (YAML-friendly) naming faults at positions:
model faults per adapter call (`malformed`, `adapter_500`) and tool faults
per attempt (`timeout`, `exception`). The mock adapter and the executor's
`fault_hook` consult it. The runtime's contract under faults: adapter
errors retry with backoff then fail the run with a cause; malformed output
gets a corrective observation twice, then fails the run; tool failures
surface to the model as error observations. Nothing raises out of
`drive()`.

## Cost accounting

Every `model_responded` carries usage metadata (tokens, latency, a
`simulated` flag for the mock adapter). `accounting.py` folds these into
per-run totals with a static price table. The UI and CLI label simulated
usage as such.

## Package map

```
src/agent_runtime/
  events.py        event types + canonical serialization
  store.py         SQLite append-only store + idempotency ledger
  projection.py    log -> RunState, log -> messages
  runtime.py       the loop, approval gating
  replay.py        recorded clock/adapter/executor/approvals + comparator
  faults.py        fault plans
  accounting.py    usage -> cost/latency
  agents.py        example agent definitions
  bootstrap.py     build runtimes for new/existing runs (CLI + web share it)
  adapters/        base types, mock, anthropic, openai
  tools/           registry (JSON-schema, side_effect, timeout, retries),
                   executor, builtin tools
  scenarios/       YAML loader, runner, terminal/HTML reports
  web/             FastAPI timeline + approval UI, diff previews
  cli.py           agentrun entry point
```
