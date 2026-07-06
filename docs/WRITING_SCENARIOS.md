# Writing scenarios

A scenario is one YAML file: an initial request, one or more scripted model
behavior versions, and assertions on the resulting event log. The pack in
`scenarios/` runs entirely offline through the mock adapter.

## Minimal example

```yaml
name: mark-invoice-paid
agent: ops_assistant
request: "Cetus GmbH paid INV-1003. Update the invoice sheet."
workspace_files:
  invoices.csv: |
    invoice,client,amount,due_date,status
    INV-1003,Cetus GmbH,890.00,2026-06-01,overdue
behaviors:
  v1:
    - tool_calls:
        - tool: read_file
          args: {path: invoices.csv}
    - tool_calls:
        - tool: csv_update
          args: {path: invoices.csv, op: set_cell,
                 row_index: 1, col_index: 4, value: paid}
    - final: "Marked INV-1003 as paid."
expect:
  status: finished
  tools_executed: [read_file, csv_update]
  final_contains: ["INV-1003", "paid"]
  no_side_effect_before_approval: true
```

Run it:

```
agentrun test path/to/file.yaml
agentrun test scenarios/ --behavior v1 --compare v2 --html report.html
```

## Fields

### `name`, `agent`, `request`

`name` must be unique across the pack. `agent` is one of the definitions in
`agent_runtime/agents.py` (`ops_assistant`, `data_entry`,
`research_summarizer`); it fixes the system prompt and the allowed tools.

### `workspace_files`

Files created in the run's sandbox workspace before the run starts. Keys
are relative paths, values are file contents.

### `behaviors`

A map of version name to a script: the exact sequence of model turns the
mock adapter will play. Each turn is exactly one of:

- `tool_calls`: a list of `{tool, args}` entries (one turn may hold several
  calls),
- `final`: the final answer text, which ends the run,
- `malformed`: raw unparseable output, to exercise the recovery path.

The script must cover every model call the run will make, including the
extra call after a tool failure or a denial. Running past the end of the
script fails the run loudly rather than guessing.

Version names are free-form. The convention in this repo: `v1` is the
baseline contract, `v2` is a candidate behavior you diff against it. The
diff report pinpoints the first event where two versions part ways.

### `approvals`

- `auto` (default): side-effecting calls are approved by policy.
- `{deny: [tool names], reason: "..."}`: listed tools are denied with that
  reason (the model sees it as an observation); everything else is
  approved.

### `faults`

A fault plan applied to the run (see `docs/ARCHITECTURE.md`):

```yaml
faults:
  model:
    - {at_call: 1, kind: adapter_500}   # or kind: malformed
  tools:
    read_file:
      - {at_attempt: 1, kind: exception, message: "disk error"}
      - {at_attempt: 2, kind: timeout}
```

`at_call` counts every adapter invocation including failed ones.
`at_attempt` counts executor attempts for that tool, so a fault at attempt
1 with a clean attempt 2 exercises the retry path. Injected model faults do
not consume a script turn.

## Assertions (`expect`)

| key | meaning |
|---|---|
| `status` | `finished`, `failed` or `waiting_approval` |
| `tools_executed` | exact ordered list of successfully executed tools |
| `tools_executed_contains` | these tools ran, order and extras ignored |
| `tools_not_executed` | these tools must not have run |
| `denied_tools` | these tools were denied at the approval gate |
| `final_contains` / `final_not_contains` | case-insensitive substring checks on the final answer |
| `failure_cause_contains` | substring of the recorded failure cause |
| `files_exist` | workspace-relative paths that must exist afterwards |
| `min_events` | minimum event count in the log |
| `no_side_effect_before_approval` | every executed side-effect call had a prior `tool_approved` event |

All assertions are evaluated; a failing scenario reports every miss, and
the runner distinguishes assertion failures from infrastructure errors.

## Tips

- Assert the contract, not the incidentals. `tools_executed` pins the
  action sequence; use `tools_executed_contains` when order is legitimately
  flexible.
- Give failure-path scenarios explicit scripts for the recovery turns. A
  denied tool means one extra model call.
- Keep `workspace_files` minimal but real: the tools actually read them, so
  a `read_file` result in the transcript reflects the fixture exactly.
- Scripted `final` text is what `final_contains` runs against; write it the
  way you would accept from a live model.
