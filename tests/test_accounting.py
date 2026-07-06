from agent_runtime import events as ev
from agent_runtime.accounting import account
from agent_runtime.events import Event


def responded(seq, model, inp, out, latency, simulated=False):
    return Event(run_id="r", seq=seq, type=ev.MODEL_RESPONDED, ts=0.0, payload={
        "turn": {"text": "x", "tool_calls": []},
        "model": model,
        "usage": {"input_tokens": inp, "output_tokens": out,
                  "latency_ms": latency, "simulated": simulated},
    })


def test_account_sums_usage_and_prices():
    evts = [
        responded(1, "mock-1", 1000, 500, 200.0, simulated=True),
        responded(2, "mock-1", 2000, 1000, 400.0, simulated=True),
    ]
    c = account(evts)
    assert c.model_calls == 2
    assert c.input_tokens == 3000 and c.output_tokens == 1500
    # mock-1 pricing: $3/M input, $15/M output
    assert abs(c.cost_usd - (3000 * 3 + 1500 * 15) / 1_000_000) < 1e-9
    assert c.avg_latency_ms == 300.0
    assert c.simulated


def test_account_skips_malformed_and_handles_empty():
    malformed = Event(run_id="r", seq=1, type=ev.MODEL_RESPONDED, ts=0.0,
                      payload={"malformed": True, "error": "x"})
    c = account([malformed])
    assert c.model_calls == 0 and c.cost_usd == 0.0
    assert account([]).avg_latency_ms == 0.0


def test_unknown_model_uses_default_pricing():
    c = account([responded(1, "some-new-model", 1_000_000, 0, 1.0)])
    assert c.cost_usd == 3.00
    assert not c.simulated
