"""Adapter for the OpenAI Chat Completions API.

POST https://api.openai.com/v1/chat/completions with function tools.
Needs OPENAI_API_KEY. Request construction is covered by offline
contract tests; live calls were not exercised here (no key available).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from .base import AdapterError, MalformedOutputError, ModelAdapter, ModelTurn, Usage

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"


class OpenAIAdapter(ModelAdapter):
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = client

    def build_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role, content = m["role"], m.get("content", "")
            if role in ("system", "user"):
                out.append({"role": role, "content": content})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": content or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["call_id"],
                            "type": "function",
                            "function": {
                                "name": tc["tool"],
                                "arguments": json.dumps(tc["args"], sort_keys=True),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m["call_id"],
                    "content": json.dumps(m["content"], sort_keys=True),
                })
        body: dict[str, Any] = {"model": self.model, "messages": out}
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        return body

    def parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelTurn:
        try:
            choice = data["choices"][0]["message"]
            calls: list[dict[str, Any]] = []
            for tc in choice.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as exc:
                    raise MalformedOutputError(
                        f"tool arguments are not valid JSON: {exc}",
                        raw=tc["function"]["arguments"][:2000],
                    ) from exc
                calls.append({
                    "call_id": tc["id"],
                    "tool": tc["function"]["name"],
                    "args": args,
                })
            usage = Usage(
                input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
            )
            return ModelTurn(text=choice.get("content") or "", tool_calls=calls,
                             usage=usage, model=data.get("model", self.model))
        except (KeyError, IndexError, TypeError) as exc:
            raise MalformedOutputError(f"unexpected response shape: {exc}",
                                       raw=json.dumps(data)[:2000]) from exc

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        if not self.api_key:
            raise AdapterError("OPENAI_API_KEY is not set")
        body = self.build_request(messages, tools)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        client = self._client or httpx.Client(timeout=120)
        start = time.monotonic()
        try:
            resp = client.post(API_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AdapterError(f"openai transport error: {exc}") from exc
        finally:
            if self._client is None:
                client.close()
        latency = (time.monotonic() - start) * 1000
        if resp.status_code >= 500:
            raise AdapterError(f"openai HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise AdapterError(f"openai HTTP {resp.status_code}: {resp.text[:500]}")
        return self.parse_response(resp.json(), latency)
