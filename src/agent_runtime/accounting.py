"""Cost and latency accounting derived from adapter usage metadata."""

from __future__ import annotations

from dataclasses import dataclass

from . import events as ev
from .events import Event

# USD per million tokens (input, output). The mock model gets a nominal
# price so simulated runs still produce a readable cost figure.
PRICING: dict[str, tuple[float, float]] = {
    "mock-1": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_PRICE = (3.00, 15.00)


@dataclass
class RunCosts:
    model: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0
    simulated: bool = False

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.model_calls if self.model_calls else 0.0


def account(events: list[Event]) -> RunCosts:
    costs = RunCosts()
    for e in events:
        if e.type != ev.MODEL_RESPONDED or e.payload.get("malformed"):
            continue
        usage = e.payload.get("usage", {})
        model = e.payload.get("model", "")
        price_in, price_out = PRICING.get(model, DEFAULT_PRICE)
        costs.model = model or costs.model
        costs.model_calls += 1
        costs.input_tokens += usage.get("input_tokens", 0)
        costs.output_tokens += usage.get("output_tokens", 0)
        costs.total_latency_ms += usage.get("latency_ms", 0.0)
        costs.cost_usd += (
            usage.get("input_tokens", 0) * price_in
            + usage.get("output_tokens", 0) * price_out
        ) / 1_000_000
        costs.simulated = costs.simulated or bool(usage.get("simulated"))
    return costs
