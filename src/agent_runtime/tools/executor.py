"""Tool execution with timeout, retries with backoff, and idempotency.

The idempotency key is derived from (run_id, seq of the tool_requested
event), so a resumed run computes the same key and finds the recorded
result instead of executing the side effect twice.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..store import EventStore
from .registry import ToolContext, ToolError, ToolSpec


class ToolTimeout(ToolError):
    pass


def idempotency_key(run_id: str, requested_seq: int) -> str:
    return hashlib.sha256(f"{run_id}:{requested_seq}".encode()).hexdigest()[:32]


@dataclass
class ExecutionOutcome:
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    replayed: bool = False  # result came from the idempotency ledger


class ToolExecutor:
    def __init__(
        self,
        store: EventStore,
        ctx: ToolContext,
        sleep: Callable[[float], None] = time.sleep,
        fault_hook: Callable[[str, int], Exception | None] | None = None,
    ):
        """fault_hook(tool_name, attempt) may return an exception to inject."""
        self.store = store
        self.ctx = ctx
        self.sleep = sleep
        self.fault_hook = fault_hook

    def execute(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        run_id: str,
        requested_seq: int,
    ) -> ExecutionOutcome:
        key = idempotency_key(run_id, requested_seq)
        prior = self.store.get_execution(key)
        if prior is not None:
            return ExecutionOutcome(ok=True, result=prior, replayed=True)

        try:
            spec.validate_args(args)
        except Exception as exc:  # jsonschema.ValidationError and friends
            return ExecutionOutcome(ok=False, error=f"invalid arguments: {exc}")

        last_error = "unknown error"
        attempts = 0
        for attempt in range(spec.retries + 1):
            attempts = attempt + 1
            try:
                if self.fault_hook is not None:
                    injected = self.fault_hook(spec.name, attempts)
                    if injected is not None:
                        raise injected
                result = self._run_with_timeout(spec, args)
                # Record the result BEFORE the caller appends tool_executed:
                # if we crash between these two writes, resume finds the
                # ledger entry and does not repeat the side effect.
                self.store.claim_execution(key, run_id, result)
                return ExecutionOutcome(ok=True, result=result, attempts=attempts)
            except ToolTimeout as exc:
                last_error = f"timeout after {spec.timeout}s: {exc}"
            except ToolError as exc:
                # Expected, deterministic failure: retrying will not help.
                return ExecutionOutcome(ok=False, error=str(exc), attempts=attempts)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < spec.retries:
                self.sleep(spec.backoff * (2 ** attempt))
        return ExecutionOutcome(ok=False, error=last_error, attempts=attempts)

    def _run_with_timeout(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(spec.handler, args, self.ctx)
            try:
                return future.result(timeout=spec.timeout)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise ToolTimeout(spec.name) from exc
